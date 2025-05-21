# 🔍 Weapon Detection System

![Weapon Detection](https://img.shields.io/badge/AI-Weapon%20Detection-red)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)


A robust deep learning-based weapon detection system using YOLOv8 for real-time identification of weapons in images and videos. This project includes three specialized models for different weapon detection scenarios.

## 🚀 Features

- **Real-time weapon detection** in images, videos, and webcam feeds
- **Three specialized models**:
  - 🔫 **Gun Detection Model**: Specialized for firearms detection
  - 🗡️ **Stick/Knife/Sword Model**: Focused on melee weapons
  - 🛡️ **All Weapons Model**: Comprehensive detection of various weapon types
- **High accuracy** with optimized YOLOv8 architecture
- **Easy-to-use** command line interface
- **Customizable confidence thresholds** for detection sensitivity

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for real-time performance)

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/weapon-detection.git
   cd weapon-detection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🔧 Models

The project includes three pre-trained models:

1. **gun.pt** - Specialized for detecting firearms
2. **stick_knife_sword.pt** - Focused on detecting melee weapons
3. **All_weapon.pt** - Comprehensive model for detecting various weapon types

All models are available on [HuggingFace](https://huggingface.co/yourusername/weapon-detection-models).

## 🎮 Usage

### Image Detection

```bash
python infer.py --model gun.pt --source path/to/image.jpg
```

### Video Detection

```bash
python infer.py --model All_weapon.pt --source path/to/video.mp4 --output results.mp4
```

### Webcam Detection

```bash
python infer.py --model stick_knife_sword.pt --source webcam
```

### Adjusting Confidence Threshold

```bash
python infer.py --model gun.pt --source path/to/image.jpg --conf 0.6
```

### Running Tests

You can test the models using the `infer.py` script with various inputs:

```bash
# Test with sample images
python infer.py --model gun.pt --source test_images/gun1.jpg

# Test with a video file
python infer.py --model All_weapon.pt --source test_videos/surveillance.mp4 --output Result/detection_result.mp4

# Test with webcam
python infer.py --model stick_knife_sword.pt --source webcam
```

All detection results (processed images and videos) are saved in the `Result/` folder.

## 🔄 Training Your Own Model

You can fine-tune the models on your own dataset using the provided `finetune.py` script:

```bash
python finetune.py
```

The script handles:
- Dataset preparation and organization
- Training configuration
- Model training with optimized hyperparameters
- Model export to various formats

## 📊 Dataset

The models were trained on a custom dataset of weapon images collected from various sources:
- [Roboflow](https://roboflow.com) datasets
- Custom dataset available [here](https://drive.google.com/drive/folders/179q_MNjx0ipzybhdjpQTxVu3IbI-5lWl)

## 📁 Project Structure

```
Weapon_detection/
├── finetune.py            # Script for fine-tuning models
├── infer.py               # Script for testing/inference of models
├── paths.txt              # Example command paths
├── requirements.txt       # Project dependencies
├── Models/       # Add all models here after download
└── Result/                # Folder where detection results are saved
```

## 📈 Performance

The models achieve:
- Average inference speed: ~20-30 FPS on GPU
- mAP@0.5: 0.85+ (varies by model)
- Real-time detection capability on modern hardware

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


##  Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv8 implementation
- [Roboflow](https://roboflow.com) for dataset resources
