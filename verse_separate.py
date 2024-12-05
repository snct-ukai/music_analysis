import numpy as np
import librosa
import matplotlib.pyplot as plt

def verse_separate(vocal : np.ndarray, data : np.ndarray, sr : int):
    hop_length = 512
    frame_length = 2048
    vocal_rms = np.array([
        sum(abs(vocal[i:i+frame_length]**2))
        for i in range(0, len(vocal), hop_length)
    ])
    vocal_rms = vocal_rms / np.max(vocal_rms)
    vocal_rms = np.exp(vocal_rms) - 1
    vocal_rms = vocal_rms / np.max(vocal_rms)
    vocal_rms[vocal_rms < 0.1] = 0
    tempo, beats = librosa.beat.beat_track(y=vocal, sr=sr)
    one_beat = int(sr * 60 / tempo / hop_length)
    verse = []

    count = 0
    for i in range(len(vocal_rms) - one_beat):
        if(vocal_rms[i] < 0.1):
            count += 1
        else:
            if count > one_beat * 4:
                verse.append(i - count)
                verse.append(i)
            count = 0

    if __name__ == '__main__':
        plt.plot(vocal_rms)
        for i in verse:
            plt.axvline(i, color='red')
        plt.show()
        
    verse = np.array(verse) * hop_length
    data_array = [
        np.array(data[verse[j]:verse[j+1]]) for j in range(0, len(verse) - 1, 1)
    ]

    return data_array

if __name__ == '__main__':
    import sys
    from spleeter.separator import Separator
    import soundfile as sf
    file = sys.argv[1]
    sr = sf.info(file).samplerate
    data, sr = librosa.load(file, sr = sr)
    Separator('spleeter:2stems').separate_to_file(file, 'tmp')
    import re
    filename = re.split(r'[/\\]', file)[-1].split('.')[0]
    sr = sf.info(f'tmp/{filename}/vocals.wav').samplerate
    vocal = librosa.load(f'tmp/{filename}/vocals.wav', sr=sr)[0]
    verse_separate(vocal, data, sr)
