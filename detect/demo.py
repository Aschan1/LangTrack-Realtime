import cv2
import dspy
import io
import queue
import threading
import requests
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly
import torch
from typing import List, Optional
from ultralytics import YOLOE
from LLMs.get_prompt import PromptProcessor # Import your high-dimensional parsing class

# ==========================================
# 1. Configuration Parameters
# ==========================================
WHISPER_API_URL = "http://localhost:5000/transcribe"

# Audio stream configuration
MF_S = 44100       
TF_S = 16000       
CHUNK_SIZE_MS = 32.0 
NSM = int(MF_S * CHUNK_SIZE_MS / 1000)
NST = int(TF_S * CHUNK_SIZE_MS / 1000)

VAD_THRESHOLD = 0.4
SILENCE_TIME = 0.8  
MAX_DURATION = 20.0 
MIN_SPEECH = 0.25   

#Implementing a wait and lock mechanism to esnure output can be parsed even before speech is fully completed.
WAIT_K = 3                                              # commit ≥k stable words before first parse
PARSE_STRIDE_CHUNKS = int(0.4 * (1000 / CHUNK_SIZE_MS)) # re-transcribe+parse ~every 400ms of speech
LOCK_AGREEMENT_N = 2                                    # field locks after N consecutive identical parses

SB_CHUNKS = int(SILENCE_TIME * (1000 / CHUNK_SIZE_MS))
SPB_CHUNKS = int(MAX_DURATION * (1000 / CHUNK_SIZE_MS))
MIN_SPB_CHUNKS = int(MIN_SPEECH * (1000 / CHUNK_SIZE_MS))

# ==========================================
# 2. Main Encapsulated Class
# ==========================================
class VoiceControlledYOLO:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.audio_queue = queue.Queue()
        
        # Thread communication variables
        self.global_search_query = ""
        self.last_search_query = ""
        self.query_lock = threading.Lock()#The lock ensures that only one thread can access or modify the global_search_query at a time, preventing multiple threads from simultaneously reading or writing to the variable, which could lead to inconsistent or unexpected behavior.
        
        self._initialize_models()

    def _initialize_models(self):
        """Encapsulated model initialization process"""
        print("[1/3] Loading Qwen Brain (DSPy)...")
        self.main_lm = dspy.LM(model="openai/qwen3.5", api_base="http://127.0.0.1:8080/v1", api_key="unused", cache=False)
        dspy.configure(lm=self.main_lm)
        self.prompt_processor = dspy.Predict(PromptProcessor)
        
        try:
            self.prompt_processor.load("./LLMs/optimized_new_prompt.json")
            print("[Qwen3.5]: Loaded optimized_new_prompt.json successfully!")
        except Exception as e:
            print(f"[Qwen3.5]: Warning: Using default prompt. Error: {e}")
            print("DSPy Qwen3.5 Client is ready!")

        print("[2/3] Loading YOLO Eyes...")
        self.yolo_model = YOLOE("yoloe-26l-seg.pt").to(self.device) 

        print("[3/3] Loading Silero VAD (Auditory Nerve)...")
        torch.set_num_threads(1)
        self.vad_model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
        self.vad_model.eval()
        self.vad_iterator = self.utils[3](self.vad_model, sampling_rate=TF_S, threshold=VAD_THRESHOLD)

        print("-" * 50)
        print("All systems initialized and ready!")
        print("-" * 50)

    # ==========================================
    # 3. Audio & LLM Core (Background Thread)
    # ==========================================
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"[Audio]: Audio error: {status}")
        chunk = np.asarray(indata, dtype=np.float32)
        if chunk.ndim > 1:
            chunk = chunk[:, 0]
        self.audio_queue.put(chunk)

    def transcribe_via_api(self, speech_segments):
        audio_data = np.concatenate(speech_segments, axis=0)
        wav_io = io.BytesIO()
        sf.write(wav_io, audio_data, TF_S, format='WAV', subtype='PCM_16')
        wav_io.seek(0)
        
        try:
            response = requests.post(
                WHISPER_API_URL, 
                files={"file": ("capture.wav", wav_io, "audio/wav")},
                timeout=5.0
            )
            if response.status_code == 200:
                return response.json().get("text", "")
        except Exception as e:
            print(f"[Whisper]: API Request failed: {e}")
        return ""

    # Function to reset the utterance state for new speech input
    def _reset_utterance_state(self):
        self.prev_partial_words = []
        self.committed_words = []
        self.locked  = {"target": None, "attributes": None,
                    "anchor_object": None, "spatial_relation": None}
        self.pending = {k: (None, 0) for k in self.locked}   # (value, agreement_count)
        self.last_parsed_prefix = ""

    # Function to find the longests stable prefix between two sequences of words
    # Used to determine when to commit a partial parse to the final output.
    @staticmethod
    def stable_prefix(prev_words, curr_words):
        out = []
        for a, b in zip(prev_words, curr_words):
            if a == b: out.append(a)
            else: break
        return out
    
    # Function to update the locks based on the latest parse results
    def update_locks(self, res):
        #Iterate over each field in the locked dictionary to determine if it can be locked based on the latest parse results
        for field in self.locked:
            #If the field is already locked, skip further processing for that field
            if self.locked[field] is not None:
                continue  # Already locked, skip
            new_value = getattr(res, field, None)#getattr the field value from the parse result
            if str(new_value).strip().lower() in ("none", "null","","[]","unknown"):
                self.pending[field] = (None, 0) # Reset pending sequence to none if the new value is invalid. Target is not set to None, but instead pending is, because we want to keep track of the last valid value for potential locking.
                continue  # Ignore invalid values
            prev_value, count = self.pending[field]#Extracting the previous value and count for the field from the pending dictionary
            count = count + 1 if new_value == prev_value else 1#Counting consecutive identical parses for lock agreement
            #count = count +1 was used vs count +=1 to ensure that the count is incremented only when the new value matches the previous value, otherwise it resets to 1 for a new value.
            self.pending[field] = (new_value, count)
            if count >= LOCK_AGREEMENT_N:
                self.locked[field]=new_value #Lock the field if agreement count is reached
    
    #Function to push the locked target to the global search query for YOLO tracking
    def push_target(self):
        attributes = self.locked["attributes"] or []
        query = (" ".join(attributes) + " " + self.locked["target"]).strip()
        with self.query_lock:
            if query != self.global_search_query:#Only update the global search query if it has changed to avoid unnecessary updates
                self.global_search_query = query#Update the global search query with the new target
    #Function to handle the interim parse of speech input, stabilizing the output and committing to final output if enough stable words are detected
    def interim_parse(self, speech_buffer):
        text = self.transcribe_via_api(speech_buffer)
        curr_words = text.split()
        #Stabilize the output by checking the longest common prefix of the previous and current words
        self.committed_words = self.stable_prefix(self.prev_partial_words, curr_words)
        self.prev_partial_words = curr_words
        #Wait-K mechanism: Only commit to the final output if we have at least K stable words
        if len(self.committed_words) < WAIT_K:
            return #Not enough stable words to commit, wait for more input
        #Commit the stable words to the final output
        committed=" ".join(self.committed_words) #Join the stable words into a single string
        if committed == self.last_parsed_prefix:
            return #No new committed words, nothing to do
        self.last_parsed_prefix = committed #Update the last parsed prefix to the newly committed words
        #Parse the committed words using DSPy to extract structured information
        res = self.prompt_processor(speech=committed)
        #Update the pending fields based on the new parse results
        self.update_locks(res)
        if self.locked["target"]: #If have notyet pushed locked target, push it
            self.push_target()
    
    #Function to handle the final parse when speech input is complete
    def final_parse(self, speech_buffer):
        text = self.transcribe_via_api(speech_buffer)#Transcribe the final speech buffer to text
        if not text.strip():
            print("[Whisper]: No valid speech detected in the final parse.")
            return
        res = self.prompt_processor(speech=text)#Parse the final transcribed text using DSPy to extract structured information
        #Update the pending fields based on the final parse results
        valid = str(getattr(res,'is_valid_command','False')).strip().lower() in ('true','yes','1')
        target = str(getattr(res,'target','')).strip().lower()
        with self.query_lock:
            if valid and target not in ('none','null','','unknown'):
                self.global_search_query = (" ".join(res.attributes) + " " + res.target).strip() if res.attributes else res.target
            else:
                self.global_search_query = ""#Reset the global search query if the command is invalid or the target is not defined
    #Function to process audio input in a separate thread, handling VAD, speech buffering, and parsing
    def process_audio_llm_thread(self):
        is_speaking = False
        silence_counter = 0
        chunks_since_last_parse = 0
        speech_buffer = []
        self._reset_utterance_state()  # Reset the utterance state for the next speech input
        while True:
            audio = self.audio_queue.get()
            if audio is None: 
                break
            
            # sample wanted length between TF_S and MF_S, then run VAD on it
            resampled_16k = resample_poly(audio, TF_S, MF_S).astype(np.float32)
            if len(resampled_16k) < NST:
                resampled_16k = np.pad(resampled_16k, (0, NST - len(resampled_16k)))
            elif len(resampled_16k) > NST:
                resampled_16k = resampled_16k[:NST]
                
            with torch.no_grad():
                audio_tensor = torch.from_numpy(resampled_16k).float().unsqueeze(0)
                self.vad_iterator(audio_tensor, return_seconds=True)
                triggered = self.vad_iterator.triggered
               
            # if triggered, add the audio chunk to the buffer and check for minimum length
            if triggered:
                speech_buffer.append(resampled_16k)
                silence_counter = 0
                is_speaking = True
                chunks_since_last_parse += 1
                if len(speech_buffer) >= MIN_SPB_CHUNKS and chunks_since_last_parse >= PARSE_STRIDE_CHUNKS:
                    chunks_since_last_parse = 0
                    self.interim_parse(speech_buffer)
            else:
                # Here means we got silence after some speech, we check if the speech buffer has enough content to be a valid command or reached the max duration
                if is_speaking:
                    silence_counter += 1
                    if silence_counter >= SB_CHUNKS or len(speech_buffer) >= SPB_CHUNKS:
                        if len(speech_buffer) >= MIN_SPB_CHUNKS:
                            self.final_parse(speech_buffer)
                            # print(f"\n[Whisper]: Speech captured, sending to Whisper API...")
                            # text = self.transcribe_via_api(speech_buffer)
                            
                            # if text.strip():
                            #     print(f"[Whisper]: '{text}'")
                                
                            #     # DSPy structured extraction
                            #     dspy_res = self.prompt_processor(speech=text)
                                
                            #     is_valid_str = str(getattr(dspy_res, 'is_valid_command', 'False')).strip().lower()
                            #     is_valid = is_valid_str in ['true', 'yes', '1']
                            #     target_str = str(getattr(dspy_res, 'target', 'None')).strip().lower()

                            #     # only update the search query if the command is valid and has a tangible target
                            #     if is_valid and target_str not in ['none', 'null', '', 'unknown']:
                            #         print(f"[Qwen Reasoning]: Target: {dspy_res.target}, Attributes: {dspy_res.attributes}")
                                    
                            #         search_query = " ".join(dspy_res.attributes) + " " + dspy_res.target if dspy_res.attributes else dspy_res.target
                            #         search_query = search_query.strip()
                            #         print(f"[Qwen Reasoning] New visual target locked: '{search_query}'")
                                    
                            #         # Update global variable across threads
                            #         with self.query_lock:
                            #             self.global_search_query = search_query
                            #     else:
                            #         print(f"[Qwen Reasoning]: Ignored invalid command or background noise -> '{text}'")
                        
                        speech_buffer = []
                        silence_counter = 0
                        is_speaking = False
                        chunks_since_last_parse = 0
                        self._reset_utterance_state()  # Reset the utterance state for the next speech input

    # ==========================================
    # 4. Visual Hub & Main Loop (Main Thread)
    # ==========================================
    def run(self):
        # Start mic stream
        stream = sd.InputStream(samplerate=MF_S, channels=1, callback=self.audio_callback, blocksize=NSM)
        stream.start()

        # Start background processing thread
        audio_thread = threading.Thread(target=self.process_audio_llm_thread, daemon=True)
        audio_thread.start()

        # Open camera
        cap = cv2.VideoCapture(0)
        print("\nCamera is ON! Please start speaking (e.g., 'Find the red pepper')...")
        print("Press ESC to exit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read camera frame.")
                break
                
            # Safely get the latest search query
            with self.query_lock:
                current_query = self.global_search_query
                
            if current_query:
                if current_query != self.last_search_query:#Only update YOLO target if the search query has changed
                    print(f"Updating YOLO target to: '{current_query}'")
                    self.yolo_model.set_classes([current_query])
                    self.last_search_query = current_query
                
                # Run tracking (no longer stuttering!)
                results = self.yolo_model.track(frame, conf=0.1, persist=True, verbose=False)
                annotated_frame = results[0].plot()
                
                # UI text for active search
                cv2.putText(annotated_frame, f"Searching: {current_query}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # Default UI before any command is given
                annotated_frame = frame.copy()
                cv2.putText(annotated_frame, "Listening for commands...", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Multimodal Voice-Controlled YOLO", annotated_frame)
            
            # Press ESC to exit
            if cv2.waitKey(1) & 0xFF == 27:
                break

        # Graceful cleanup
        stream.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Application exited gracefully.")

if __name__ == "__main__":
    app = VoiceControlledYOLO()
    app.run()