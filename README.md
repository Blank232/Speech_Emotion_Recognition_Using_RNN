# Speech Emotion Recognition

## Overview
This project is a **Speech Emotion Recognition (SER) system** using deep learning. It extracts **MFCC features** from audio files and classifies emotions using a **Bidirectional LSTM** model. The system also includes **data augmentation**, a **confusion matrix**, and **real-time predictions on new audio files**.

## Features
- **Multi-class emotion detection** based on MFCC features
- **Deep learning model** using BiLSTM
- **Data Augmentation** (Adding Noise, Time Shifting)
- **Confusion matrix & accuracy visualization**
- **Real-time emotion prediction on new audio files**

## Emotion Labels
The system classifies speech into the following emotions:
- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

## Dataset
- **Dataset Path:** `C:/Users/ishir/Speech_Emotion_Recognition/dataset mini project`
- The dataset contains `.wav` files stored in actor-wise directories.

## Dependencies
Install the required libraries using:
```bash
pip install numpy tensorflow librosa matplotlib scikit-learn keras
```

## Model Architecture
- **Input Layer:** MFCC features
- **BiLSTM Layers:** Two Bidirectional LSTM layers
- **Batch Normalization & Dropout:** Prevent overfitting
- **Dense Layers:** Fully connected layers for classification
- **Softmax Activation:** Multi-class classification

## Training
The model is trained using **categorical crossentropy loss** with an **Adam optimizer** and an **exponential learning rate decay**.
```python
history = model.fit(X_train, y_train, validation_split=0.2, epochs=200, batch_size=16, callbacks=[checkpoint])
```

## Evaluation
- The model achieves **76.5% accuracy** on the test set.
- A **confusion matrix** is generated to visualize classification performance.

## Real-time Prediction
You can predict the emotion of a new `.wav` file:
```python
predicted_emotion = predict_emotion_from_audio("path_to_audio.wav")
print(f"Predicted Emotion: {predicted_emotion}")
```

## Visualization
- **Training vs Validation Accuracy & Loss**
- **Confusion Matrix**
```python
ConfusionMatrixDisplay(conf_matrix, display_labels=label_encoder.classes_).plot(cmap='Blues')
plt.show()
```

## Usage
1. **Train the model**: Run the script to preprocess data and train the model.
2. **Evaluate performance**: Check the test accuracy and confusion matrix.
3. **Predict on new audio files**: Provide an audio file path for real-time emotion recognition.

## Future Improvements
- Integrating with a real-time audio input system
- Expanding dataset diversity
- Implementing more augmentation techniques

## Author
Ishir Srivats
