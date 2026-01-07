

import sounddevice as sd
import numpy as np
import soundfile as sf
import time

from voice import *

from concurrent.futures import ThreadPoolExecutor

from collections import deque
import os
import joblib
import pandas as pd

modelr = joblib.load('toma_response_model2.pkl')
res0 = 0
toma_face = "neutral"

# 初始化线程池，最多同时处理2段分析
executor = ThreadPoolExecutor(max_workers=2)

samplerate = 16000
chunk_duration = 0.5  # 秒
chunk_size = int(samplerate * chunk_duration)
silence_threshold = 0.008 #可以调整，

max_record_sec = 10
min_record_sec = 0.5

silence_cut_sec = 0.25  # 安静多久才截断
silence_start_time = None


def analyze_and_cleanup(filename):
    try:
        return fromVoice(filename)
    finally:
        try:
            os.remove(filename)
            #print(f"🗑️ 分析完成，删除: {filename}")
        except FileNotFoundError:
            pass

def rms(chunk):
    chunk = chunk.astype(np.float32) / 32768.0 #发现声音太大录不进去，做的归一化
    return np.sqrt(np.mean(chunk ** 2))

def start_toma():
    print("🎤我来啦！")
    global res0, toma_face, silence_start_time
    try:
        with sd.InputStream(samplerate=samplerate, channels=1, dtype='int16') as stream:
            buffer = []
            recording = False
            start_time = None
            temp = "neutral"
            while True:
                chunk, overflowed = stream.read(chunk_size)
                volume = rms(chunk)
                
                if volume > silence_threshold:
                    silence_start_time = None
                    if not recording:
                        recording = True
                        start_time = time.time()
                        buffer = []
                        temp = toma_face
                        toma_face = "listening"
                        print("🟢我在听……")

                    buffer.append(chunk)
                    if time.time() - start_time >= max_record_sec:
                        filename = f"segment_{int(time.time())}.wav"
                        audio = np.concatenate(buffer)
                        sf.write(filename, audio, samplerate)
                    
                        # 异步执行情绪识别
                        
                        executor.submit(analyze_and_cleanup, filename)
                        toma_face = temp
                        recording = False
                        buffer = []

                elif recording:
                    if silence_start_time is None:
                        silence_start_time = time.time()

                    silence_duration = time.time() - silence_start_time

                    if silence_duration >= silence_cut_sec:
                        duration = time.time() - start_time

                        if duration >= min_record_sec:
                            filename = f"segment_{int(time.time())}.wav"
                            audio = np.concatenate(buffer)
                            sf.write(filename, audio, samplerate)
                            executor.submit(analyze_and_cleanup, filename)
                            toma_face = temp
                        else:
                            print("⚠️唔。没听清")
                            toma_face = temp

                        recording = False
                        buffer = []
                        silence_start_time = None
                else:
                    
                    state = cooldown_vad()
                    if state is not None:
                        v, a, d = state
                        #print(f"❄️ 冷却中 | V={v:.3f} A={a:.3f} D={d:.3f}")
                        X_single = pd.DataFrame([[v,a,d]], columns=["valence", "arousal", "dominance"])
                        y_pred = modelr.predict(X_single)
                        y = y_pred[0]

                        if res0 != y:
                            if y == 0:
                                toma_face = "neutral"
                            elif y == 1:
                                toma_face = "happy"
                            elif y == 2:
                                toma_face = "sad"
                            elif y == 3:
                                toma_face = "fear"

                            res = ["😊舒服~","😄开心！","😟伤心……","😰怕怕"]
                            print(res[y])

                        res0 = y
                        
                    


            

    except KeyboardInterrupt:
        print("\n🛑再见！")

#start_toma()