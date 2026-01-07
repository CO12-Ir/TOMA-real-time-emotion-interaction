# 🍅 TOMA — Tomato AI Pet

🌏 Language: English | [中文](README_zh.md)

TOMA is a real-time, voice-driven virtual pet that reacts to the user's
**emotional tone of speech** rather than speech content.

Instead of focusing on speech recognition or dialogue intelligence,
TOMA emphasizes **affective perception and feedback**, forming a closed-loop
human–computer interaction system based on speech emotion signals.

This is an experimental prototype focusing on interaction experience,
and many behaviors are intentionally simplified.

---

## Overview

TOMA continuously listens to microphone input, detects valid speech segments,
analyzes emotional cues from speech, and updates the pet's visual expression
accordingly.

The system is designed as a lightweight affective interaction pipeline,
combining a pretrained speech emotion recognition (SER) model with a
locally trained response classifier and a real-time graphical interface.

---

## System Pipeline

### 1. Real-Time Audio Monitoring

The system continuously monitors microphone input in the background.

- When the audio volume exceeds a predefined threshold, recording starts.
- Recording automatically stops when:
  - the input remains silent for a short period, or
  - the maximum recording duration is reached.

This mechanism ensures that only **meaningful speech segments** are recorded,
avoiding long silence or unnecessary audio data.

---

### 2. Speech Segment Storage

Each detected speech segment is temporarily saved as an audio file.
Once saved, the segment is immediately sent for emotion analysis, while
the system resumes listening for new speech input.

---

### 3. Speech Emotion Recognition (SER)

Recorded speech segments are processed by a pretrained SER model to extract:

- A **discrete emotion label** (e.g. Neutral, Happy, Sad, Angry, Fearful)
- A **continuous VAD vector**:
  - Valence (emotional positivity/negativity)
  - Arousal (activation level)
  - Dominance (sense of control)

To improve temporal stability, VAD outputs are smoothed over time using
exponential moving averages and a cooldown mechanism.

---

### 4. Custom Response Classifier

Emotion recognition outputs are not used directly as pet behaviors.

Instead, the emotion label and VAD values are passed to a **custom response
classifier**, trained on a small, manually labeled dataset created by the author.

- Input: emotion label + VAD values
- Output: discrete pet response categories (e.g. calm, happy, sad, fear)

This layer decouples **affective perception** from **behavioral decision-making**.

---

### 5. Visual Feedback with Pygame

The predicted pet response updates a global state variable.
A continuously running Pygame loop monitors this state and dynamically updates:

- the pet’s facial expression
- the displayed emotion image

From the user’s perspective, the virtual pet reacts in real time to
changes in vocal emotion.

---

## Model and Data

### Speech Emotion Recognition

TOMA uses a pretrained SER model for inference only:

- **MERaLiON-SER: Robust Speech Emotion Recognition Model for English and SEA Languages**

The original model is not modified or retrained in this project.

### Response Classifier

- Trained locally using a small, manually annotated dataset
- Uses emotion labels and VAD features as input
- Implemented with LightGBM
- Intended for experimental and interaction purposes only

Due to limited data size, the response classifier is **not designed for
broad generalization**.

---

## Design Philosophy

- Emotion recognition is treated as a **signal extraction layer**, not a final decision.
- Behavioral responses are determined by a lightweight, interpretable classifier.
- The system prioritizes **responsiveness, stability, and interaction feel**
  over raw classification accuracy.

---

## Requirements

- Python 3.9+
- torch
- transformers
- sounddevice
- pygame
- numpy, pandas, scikit-learn, lightgbm

---

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

## References

MERaLiON Team.
MERaLiON-SER: Robust Speech Emotion Recognition Model for English and SEA Languages
arXiv:2511.04914, 2025

Wang et al.
Benchmarking Contextual and Paralinguistic Reasoning in Speech-LLMs
Findings of EMNLP 2025

Wang et al.
Contextual Paralinguistic Data Creation for Multi-Modal Speech-LLM
Interspeech 2025

Wang et al.
Incorporating Contextual Paralinguistic Understanding in Large Speech-Language Models
IEEE ASRU 2025