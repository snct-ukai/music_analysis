import librosa
import chord_estimation as ce

audioFileDir = "./wav"
# get audio file paths
audioFiles = librosa.util.find_files(audioFileDir)
# load audio files
audioData = [librosa.load(audioFile, sr=None) for audioFile in audioFiles]

# estimate chords for each audio file
for i, (y, sr) in enumerate(audioData):
    y, _ = librosa.effects.trim(y)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    chord_name = ce.chord_estimate(chroma, tempo, len(y) / sr)
    # print estimated chords 4 chord at  one line
    print(audioFiles[i])
    for j, chord in enumerate(chord_name):
        print(chord, end="\t")
        if (j + 1) % 4 == 0:
            print()