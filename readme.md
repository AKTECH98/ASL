
# ASL Gesture Recognition and Text Translation System

This project implements a deep learning-based solution for recognizing American Sign Language (ASL) gestures and translating them into text. It leverages pose, hand, and face landmarks extracted from video frames and uses sequence modeling techniques such as LSTM, GRU, and attention mechanisms.

---

## 📁 Dataset

- **Source**: MS-ASL Dataset  
- **Size**: 25,000+ annotated videos  
- **Labels**: 1,000 different ASL signs  
- **Diversity**: 200+ signers recorded in real-life, unconstrained environments  

---

## 🎯 Objective

To recognize ASL gestures from video and convert them into text using deep learning models that can process spatiotemporal data.

---

## 🧠 Model Architectures

### 1. LSTM Model

- **Type**: Recurrent Neural Network (RNN)
- **Layers**:
  - One or more LSTM layers
  - Final Dense layer with softmax activation

### 2. GRU Model

- **Type**: RNN with Gated Recurrent Units
- **Layers**:
  - GRU layers
  - Final Dense classification layer

### 3. Attention + GRU Model

- **Type**: Hybrid RNN with Attention
- **Layers**:
  - GRU layers
  - Attention mechanism
  - Dense output layer

### 4. Attention-Only Model

- **Type**: Transformer-like Self-Attention Model
- **Layers**:
  - Self-attention layers
  - Positional encoding
  - Feedforward classification network

---

## 🏗️ Pipeline

```
Video Input → MediaPipe Holistic → Frame Windowing → Sequence Model (LSTM/GRU/Attention) → Predicted Sign → Text Translation
```

**MediaPipe Holistic Extraction:**
- Pose Landmarks: 33 points
- Face Landmarks: 468 points
- Hand Landmarks: 21 points per hand

**Frame Windowing:**
- Typical size: 30–40 consecutive frames per sample

---

## 📂 Project Structure

```
ASL/
├── Models/                # Model architecture files and trained weights
├── Output/                # Trained models and evaluation metrics
├── config/                # YAML configuration files for models and preprocessing
├── DataProcessor/         # Data processing scripts and utilities
├── Modules/               # Core deep learning and attention modules
```

---

## ⚙️ Setup

1. **Create and activate a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## 🧹 Data Preprocessing

1. **Run Data Cleaning and Preparation:**

```bash
python data_cleaning.py --raw_data <raw_data_path> --cleaned_data <output_path> --checkpoint --show_video
```

2. **Run Preprocessing:**

```bash
python preprocess.py --config config/preprocess.yaml
```

---

## 🧪 Run Experiment

```bash
python run_experiment.py --config <model_config_file>
```

---

## 🏋️ Training (Manual Script)

```bash
python train.py --config <config_file> --exp_name <experiment_name>
```

---

## 📺 Live Prediction

```bash
python live_predict.py --model_path Models/experiment_1.pt
```

> Press `q` to quit the live prediction mode.

---

## 👤 Author

**Anshul Kiyawat** (`ak3748`)
