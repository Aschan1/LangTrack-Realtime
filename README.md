# LangTrack-Realtime

Better real-time open-vocabulary detection system in home

## Project Description

LangTrack-Realtime is a real-time open-vocabulary detection system based on advanced AI technology, designed for home environments. It integrates models like YOLOv8-World, Whisper, Qwen2.5, and SigLIP to achieve real-time object detection, speech recognition, and multimodal understanding at once.

## Key Features

- **Real-time Detection**: Fast open-vocabulary object detection based on YOLOv8-World
- **Speech Integration**: Integrated Whisper for speech-to-text
- **Multimodal Understanding**: Uses SigLIP/Qwen2.5 for image-text matching
- **Home Optimization**: Detection algorithms optimized for home scenarios
- **Easy to Use**: Provides complete toolchain and example code

## Installation

### Environment Requirements

- Python 3.13
- CUDA (recommended for GPU acceleration)

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/your-username/LangTrack-Realtime.git
cd LangTrack-Realtime
```

2. Install dependencies(Whisper not included yet):
```bash
conda env create -f environment.yml
```
3. Or try to install dependencies by pip(Whisper not included yet):
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

Use tool scripts to process data(Cured data are published in [Google Cloud](https://drive.google.com/file/d/16L68W-vvtYAvxRCTGMgGVkV32QO7XKjb/view?usp=drive_link)):
```bash
python tools/cure_data.py
python tools/YOLO_data.py 
```
`YOLO_data.py` converts json and picture into Yolov8 format, but it is useless now. Because we are not using offical API `model.val` to evaluate.

### Visualize Data

```bash
python tools/visualize_data.py #Support YOLOv8 format data only.
```

### Run Detection

```bash
python ultralytics/run_world.py
```

## Dataset

Please put the dataset `GHome727` in `yolo_dataset/`, containing annotated data for home scenarios. You can download it from [here](https://drive.google.com/file/d/16L68W-vvtYAvxRCTGMgGVkV32QO7XKjb/view?usp=drive_link).

## Project Structure

```
LangTrack-Realtime/
├── models/                 # Pre-trained models
├── tools/                  # Data processing tools
├── ultralytics/           # YOLO library
├── examples/              # Example code
├── yolo_dataset/          # Dataset
├── Whisper/               # Speech processing
├── SigLIP/                # Multimodal model
└── README.md
```


## License

This project uses the MIT license. See LICENSE file for details.

## Acknowledgments

This project uses the following open-source libraries and tools:

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [SigLIP](https://github.com/google-research/big_vision)
