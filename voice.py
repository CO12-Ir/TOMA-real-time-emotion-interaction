from transformers import AutoProcessor, AutoModelForAudioClassification
import torch, soundfile as sf, torchaudio
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
import joblib

vad_state = None
last_update_time = None

NEUTRAL_VAD = np.array([0.45, 0.35, 0.45])
COOLDOWN_BETA = 0.01
COOLDOWN_INTERVAL = 1.0
alpha = 0.6
# ===== EMA 情绪状态（全局）=====

modelr = joblib.load('toma_response_model.pkl')

repo = "MERaLiON/MERaLiON-SER-v1"
processor = AutoProcessor.from_pretrained(repo)
model = AutoModelForAudioClassification.from_pretrained(repo, trust_remote_code=True).cpu().eval()


def update_vad_ema(vad):
    global vad_state, last_update_time
    vad = np.array(vad, dtype=float)

    if vad_state is None:
        vad_state = vad
    else:
        vad_state = alpha * vad + (1 - alpha) * vad_state

    last_update_time = time.time()
    return vad_state

def cooldown_vad():
    global vad_state, last_update_time

    if vad_state is None or last_update_time is None:
        return None

    now = time.time()
    elapsed = now - last_update_time
    steps = int(elapsed // COOLDOWN_INTERVAL)

    if steps <= 0:
        return vad_state

    for _ in range(steps):
        vad_state = vad_state + COOLDOWN_BETA * (NEUTRAL_VAD - vad_state)

    last_update_time = now
    return vad_state


def fromVoice(dir):
        

    wav, sr = sf.read(dir)
    if wav.ndim > 1: wav = wav.mean(axis=1)
    if sr != 16000:
        wav = torchaudio.functional.resample(torch.tensor(wav).unsqueeze(0), sr, 16000).squeeze(0).numpy()

    inputs = processor(wav, sampling_rate=16000, return_tensors="pt")
    with torch.inference_mode():
        out = model(**inputs)
    logits, dims = out["logits"], out["dims"]
    emo_idx = torch.argmax(logits, dim=1).item()
    emo_map = ["Neutral","Happy","Sad","Angry","Fearful","Disgusted","Surprised"]
    vad_raw = dims.squeeze().tolist()
    vad_smooth = update_vad_ema(vad_raw)

    #print("Predicted idx: ",emo_idx)
    #print("Predicted Emotion:", emo_map[emo_idx])
    #print("VAD raw:     ", [round(x, 3) for x in vad_raw])
    #print("VAD EMA:     ", [round(x, 3) for x in vad_smooth.tolist()])

    X_single = pd.DataFrame([[emo_idx] + vad_smooth.tolist()], columns=["label", "valence", "arousal", "dominance"])
    y_pred = modelr.predict(X_single)
    y = y_pred[0]
    res =["😊舒服~","😄开心！","😟伤心……","😰怕怕"]
    print(res[y])

    return emo_map[emo_idx], vad_smooth.tolist()


