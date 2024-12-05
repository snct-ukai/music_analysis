import librosa
import numpy as np
import os
from segment_separate import segment_separate_from_data
import sys
import wave
import soundfile as sf

def wav_separate_from_data(data, sr, filePath):
    segment_time = segment_separate_from_data(data, sr, filePath, saveFig=True)

    dirname = "segments"
    filename = os.path.basename(filePath)
    filename = os.path.splitext(filename)[0]
    output_dir = f'./{dirname}/{filename}'
    if(os.path.exists(output_dir) == True):
        # delete files
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))
    os.makedirs(output_dir, exist_ok=True)

    # オーディオファイルの分割
    segments_file = []
    for i in range(len(segment_time) - 1):
        segment = data[int(segment_time[i] * sr):int(segment_time[i + 1] * sr)]
        rms = librosa.feature.rms(y = segment)
        if np.mean(rms) < 0.01:
            del segment
            continue
        segment = np.array(segment)
        segments_file.append(segment)
        sf.write(f"{output_dir}/segment_{i}.wav", segment, sr)
        del segment
    
    return segments_file


def wav_separate(filePath):
    # オーディオファイルの最高サンプリングレートを取得
    with wave.open(filePath, "rb") as wf:
        sampleRate = wf.getframerate()

    # オーディオファイルの読み込み
    y, _ = librosa.load(filePath, sr=sampleRate)
    return wav_separate_from_data(y, sampleRate, filePath)

if __name__ == "__main__":
    try:
        filePath = sys.argv[1]
        wav_separate(filePath)
    except IndexError:
        print("Usage: python wav_separate.py <audio file path>")
        sys.exit(1)