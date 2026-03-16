# api_server.py (runs inside a Docker container)
# Before running, install dependencies inside the container:
# pip install fastapi uvicorn python-multipart

from fastapi import FastAPI, UploadFile, File
import whisper_s2t
import os

app = FastAPI()

print("🔄 Loading TensorRT-LLM Whisper engine...")
# Load the model, ensuring backend is TensorRT-LLM
model = whisper_s2t.load_model(model_identifier="large-v2", backend='TensorRT-LLM')
print("✅ Whisper service ready!")

lang_codes = ['en']
tasks = ['transcribe']
initial_prompts = [None]

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    temp_file_path = f"temp_{file.filename}"
    
    # 1. Save received audio to a temp file (whisper_s2t requires a file path)
    with open(temp_file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        # 2. Run inference using TensorRT acceleration
        out = model.transcribe([temp_file_path],
                                lang_codes=lang_codes,
                                tasks=tasks,
                                initial_prompts=initial_prompts,
                                batch_size=2)
        
        # out is typically [[{'text': 'hello', 'start_time': 0.0, ...}]]
        print(f"🎉 API transcription result: {out}")
        recognized_text = out[0][0]['text'].strip() if out and out[0] else ""
        
        return {"status": "success", "text": recognized_text}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)