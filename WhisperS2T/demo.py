import whisper_s2t

model = whisper_s2t.load_model(model_identifier="large-v2", backend='TensorRT-LLM')

files = ['../audio_data/test.m4a']
lang_codes = ['en']
tasks = ['transcribe']
initial_prompts = [None]

out = model.transcribe_with_vad(files,
                                lang_codes=lang_codes,
                                tasks=tasks,
                                initial_prompts=initial_prompts,
                                batch_size=24)

print(out[0]) # Print first utterance for first file