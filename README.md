# LangTrack-Realtime

Better real-time open-vocabulary detection system in home

# Demo

https://github.com/user-attachments/assets/6e758b35-858d-4790-b34a-80e0f0792f55

## Project Description

LangTrack-Realtime is a real-time open-vocabulary detection system based on advanced AI technology, designed for home environments. It integrates models like YOLOv8-World, Whisper, Qwen2.5, and SigLIP to achieve real-time object detection, speech recognition, and multimodal understanding at once.

## Key Features

- **Real-time Detection**: Fast open-vocabulary object detection based on YOLOv8-World
- **Speech Integration**: Integrated Whisper for speech-to-text
- **Multimodal Understanding**: Uses SigLIP/Qwen2.5 for image-text matching
- **Home Optimization**: Detection algorithms optimized for home scenarios
- **Easy to Use**: Provides complete toolchain and example code

## Getting Started

### Environment Requirements

- Python 3.13
- torch 2.4.1

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/your-username/LangTrack-Realtime.git
cd LangTrack-Realtime
```

2. Install dependencies by pip:
```bash
pip install -r requirements.txt
```
### Dataset

Please put the dataset `GHome638` in `yolo_dataset/`, containing annotated data for home scenarios. You can download it from [here](https://drive.google.com/file/d/1bLyoBie4FxnPyaa63QKMi-zgh3uZhIkl/view?usp=drive_link).

### Run YOLO-Siglip Detection

```bash
python ultralytics/run_world.py
```

### DEMO

#### 1. Run the Whisper service
1. Run the Whisper Service Container (More details please refer to [WhisperS2T](https://github.com/shashikg/WhisperS2T?tab=readme-ov-file))
```bash
docker pull shashikg/whisper_s2t:dev-trtllm
docker run -it --rm \
  --gpus all \
  -p 5000:5000 \
  -v ${PWD}/WhisperS2T:/workspace/whisper \
  shashikg/whisper_s2t:dev-trtllm \
  /bin/bash
```
- Tips: Please run the commands below in the container.
2. Install necessary libraries:
```bash
pip install fastapi uvicorn python-multipart
cd whisper
```
3. Run the API server:
```bash
python api_server.py
```
OR
```bash
python3 api_server.py
```

#### 2. Run Qwen3.5
```bash
# Lower -ngl and -c if you have tight budget on RAM
llama-server -m <YOUR_MODEL_PATH> -ngl 99 --parallel 4 -c 16482 --port 8080 --reasoning-budget 0 --temp 0.1 --top-p 0.9 --top-k 40 --repeat-penalty 1.1
```

#### 3. Run the Demo
```bash
python ultralytics/demo.py
```
## Project Structure

```
LangTrack-Realtime/
├── LLMs/                 # Pre-trained models
├── models/                 # Pre-trained models
├── tools/                  # Data processing tools
├── ultralytics/           # YOLO library
├── examples/              # Example code
├── yolo_dataset/          # Dataset
├── WhisperS2T/               # Speech processing
└── README.md
```


## License

This project uses the MIT license. See LICENSE file for details.

## Acknowledgments

This project uses the following open-source libraries and tools:

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [SigLIP](https://github.com/google-research/big_vision)
- [WhisperS2T](https://github.com/shashikg/WhisperS2T?tab=readme-ov-file)
- [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
