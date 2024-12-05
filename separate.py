from verse_separate import verse_separate
from wav_separate import wav_separate_from_data
from spleeter.separator import Separator
import librosa
import soundfile as sf
import os
import sys

def separate(filepath):
    sr = sf.info(filepath).samplerate
    data, _ = librosa.load(filepath, sr=sr)
    model = Separator('spleeter:2stems')
    model.separate_to_file(filepath, 'tmp')
    del model
    
    import re
    filename = re.split(r'[/\\]', filepath)[-1].split('.')[0]
    sr = sf.info(f'tmp/{filename}/vocals.wav').samplerate
    vocal = librosa.load(f'tmp/{filename}/vocals.wav', sr=sr)[0]
    verses = verse_separate(vocal, data, sr)

    savedir = f"./separate_data/{filename}"
    if(os.path.exists(savedir) == True):
        # delete files
        for file in os.listdir(savedir):
            os.remove(os.path.join(savedir, file))
            
    os.makedirs(savedir, exist_ok=True)
    j = 0
    for i, verse in enumerate(verses):
        segments_in_verse = wav_separate_from_data(verse, sr, filepath)
        verses[i] = [0]
        for segment in segments_in_verse:
            sf.write(f"{savedir}/segment_{j}.wav", segment, sr)
            j += 1
        del segments_in_verse

if __name__ == "__main__":
    filepath = sys.argv[1]
    separate(filepath)