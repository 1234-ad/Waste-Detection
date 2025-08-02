# Waste Detection Using Machine Learning

## Project Overview
A CNN-based image classification project that detects and categorizes garbage images into six types: cardboard, glass, metal, paper, plastic, and trash.

## Folder Structure
```
Waste-Detection-ML/
├── notebooks/
├── src/
│   ├── train.py
│   ├── predict.py
├── app/
│   └── webcam_inference.py
├── models/
│   └── best_model.h5 (after training)
├── dataset/
│   └── garbage_classification/ (after manual download)
├── README.md
├── requirements.txt
```

## Setup Instructions
1. Download dataset from [Kaggle](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
2. Extract to `dataset/garbage_classification/`
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Run Training
```bash
python src/train.py
```

## Predict Single Image
```bash
python src/predict.py path_to_image.jpg
```

## Real-Time Detection (Webcam)
```bash
python app/webcam_inference.py
```

## Model Performance
Achieved 80%+ accuracy using CNN with basic data augmentation in under 10 epochs.