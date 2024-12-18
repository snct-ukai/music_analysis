import numpy as np
import librosa
import soundfile as sf
import os
import sys
from spleeter.separator import Separator as SSeparator
import re
from util.matrix import cos_sim

class Separator():
    def __init__(self, filepath):
        self.filepath = filepath
        self.spleeterModel = SSeparator('spleeter:2stems')
        self.spleeterModel.separate_to_file(filepath, 'tmp')
        self.filename = re.split(r'[/\\]', filepath)[-1].split('.')[0]
        self.sr = sf.info(self.filepath).samplerate
        self.data = librosa.load(self.filepath, sr=self.sr)[0]
        self.verse_time = None
        self.hop_length = 512
        self.frame_length = 2048
        self.tempo, _ = librosa.beat.beat_track(y=self.data, sr=self.sr)
        self.segment_time : np.ndarray
        self.combined_score : np.ndarray = np.array([])

    def verse_separate(self, vocal : np.ndarray, sr : int):
        vocal_rms = np.array([
            sum(abs(vocal[i:i+self.frame_length]**2))
            for i in range(0, len(vocal), self.hop_length)
        ])
        vocal_rms = vocal_rms / np.max(vocal_rms)
        vocal_rms = np.exp(vocal_rms) - 1
        vocal_rms = vocal_rms / np.max(vocal_rms)
        vocal_rms[vocal_rms < 0.1] = 0
        tempo, _ = librosa.beat.beat_track(y=vocal, sr=sr)
        one_beat = int(sr * 60 / tempo / self.hop_length)
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
        
        verse.append(len(vocal_rms))

        self.verse_time = np.array(verse) * self.hop_length / sr

    def calc_feat(self, data, sr):
        rms = np.array([
            sum(abs(data[i:i+2048]**2))
            for i in range(0, len(data), 512)
        ])
        zcr = librosa.feature.zero_crossing_rate(data, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        stft = librosa.stft(data, n_fft=self.frame_length, hop_length=self.hop_length)
        spectral_flux = np.sqrt(
            np.mean(np.abs(np.diff(np.abs(stft))**2), axis=0)
        )
        mfcc = librosa.feature.mfcc(y = data, sr=sr, n_mfcc=13, hop_length=self.hop_length)
        chroma = librosa.feature.chroma_stft(y = data, sr=sr, hop_length=self.hop_length)
        spectral_contrast = librosa.feature.spectral_contrast(y = data, sr=sr, hop_length=self.hop_length)
        tonnetz = librosa.feature.tonnetz(y = data, sr=sr, hop_length=self.hop_length)
        min_length = min(len(rms), len(zcr), stft.shape[1], mfcc.shape[1], chroma.shape[1], spectral_contrast.shape[1], tonnetz.shape[1])
        # trim
        rms = rms[:min_length]
        zcr = zcr[:min_length]
        spectral_flux = spectral_flux[:min_length]
        mfcc = mfcc[:, :min_length]
        chroma = chroma[:, :min_length]
        spectral_contrast = spectral_contrast[:, :min_length]
        tonnetz = tonnetz[:, :min_length]
        rms_diff = np.abs(np.diff(rms))
        zcr_diff = np.abs(np.diff(zcr))
        spectral_flux_diff = np.abs(np.diff(spectral_flux))
        # trim
        diff_min_length = min(len(rms_diff), len(zcr_diff), len(spectral_flux_diff))
        rms_diff = rms_diff[:diff_min_length]
        zcr_diff = zcr_diff[:diff_min_length]
        spectral_flux_diff = spectral_flux_diff[:diff_min_length]
        return dict(
            rms=rms,
            zcr=zcr,
            spectral_flux=spectral_flux,
            mfcc=mfcc,
            chroma=chroma,
            spectral_contrast=spectral_contrast,
            tonnetz=tonnetz,
            rms_diff=rms_diff,
            zcr_diff=zcr_diff,
            spectral_flux_diff=spectral_flux_diff
        )
    @staticmethod
    def valid_convolve(xx, size):
        import math
        if len(xx) == 0:
            raise ValueError("Input array 'xx' cannot be empty")
        b = np.ones(size)/size
        xx_mean = np.convolve(xx, b, mode="same")
        n_conv = math.ceil(size/2)
        xx_mean[0] *= size/n_conv
        for i in range(1, n_conv):
            xx_mean[i] *= size/(i+n_conv)
            xx_mean[-i] *= size/(i + n_conv - (size % 2)) 
        return xx_mean
    
    threshold = 0.94
    @staticmethod
    def should_merge_segments(seg1_start, seg1_end, seg2_start, seg2_end, mfcc, chroma, spectral_contrast, tonnetz, 
                                  mfcc_threshold=threshold, chroma_threshold=threshold, spectral_contrast_threshold=threshold, 
                                  tonnetz_threshold=threshold):
            
        mfcc_bin_seg1 = mfcc[:, seg1_start:seg1_end]
        mfcc_bin_seg2 = mfcc[:, seg2_start:seg2_end]
        chroma_bin_seg1 = chroma[:, seg1_start:seg1_end]
        chroma_bin_seg2 = chroma[:, seg2_start:seg2_end]
        spectral_contrast_bin_seg1 = spectral_contrast[:, seg1_start:seg1_end]
        spectral_contrast_bin_seg2 = spectral_contrast[:, seg2_start:seg2_end]
        tonnetz_bin_seg1 = tonnetz[:, seg1_start:seg1_end]
        tonnetz_bin_seg2 = tonnetz[:, seg2_start:seg2_end]

        mfcc_cos_sim = cos_sim(mfcc_bin_seg1, mfcc_bin_seg2)
        chroma_cos_sim = cos_sim(chroma_bin_seg1, chroma_bin_seg2)
        spectral_contrast_cos_sim = cos_sim(spectral_contrast_bin_seg1, spectral_contrast_bin_seg2)
        tonnetz_cos_sim = cos_sim(tonnetz_bin_seg1, tonnetz_bin_seg2)

        return (
            mfcc_cos_sim > mfcc_threshold and
            chroma_cos_sim > chroma_threshold and
            spectral_contrast_cos_sim > spectral_contrast_threshold and
            tonnetz_cos_sim > tonnetz_threshold
        )
            
        

    def detect_segment(self):
        if self.verse_time is None:
            raise Exception("Verse time is not detected")
        
        sr = sf.info(f'tmp/{self.filename}/accompaniment.wav').samplerate
        data, _ = librosa.load(f'tmp/{self.filename}/accompaniment.wav', sr=sr)

        measure_length = int((self.tempo * sr) / 60 * 4)
        
        
        segment_time = []
        from scipy.stats import boxcox
        for i in range(0, len(self.verse_time) -1, 1):
            start = int(self.verse_time[i] * sr)
            end = int(self.verse_time[i+1] * sr)

            feat = self.calc_feat(data[start:end], sr)
            combined_score = np.abs(feat['rms_diff']) + np.abs(feat['zcr_diff']) + np.abs(feat['spectral_flux_diff'])
            if not np.all(combined_score == combined_score[0]):
                combined_score = boxcox(combined_score + 1)[0]
                
            window_size = int(len(combined_score) / (len(data) / sr) * (self.tempo / 60 * 8))
            if window_size == 0:
                seg_time = ((np.array([0, len(data[start:end])]) + start) / sr)
                segment_time.extend(seg_time)
                self.combined_score = np.concatenate([self.combined_score, combined_score])
                continue
            mean = self.valid_convolve(combined_score, window_size)
            std = self.valid_convolve((combined_score - mean)**2, window_size)**0.5
            n = 1.5
            threshold = mean + n * std
            threshold_low = mean - n * std

            change_point = np.where(combined_score > threshold[:len(combined_score)])[0]
            change_point_low = np.where(combined_score < threshold_low[:len(combined_score)])[0]
            change_point = np.concatenate([change_point, change_point_low])
            change_point = np.sort(change_point)
            change_samples = change_point * self.hop_length
            change_samples = np.concatenate(([0], change_samples, [len(data[start:end])]))

            min_segment_length = measure_length
            filtered_change_samples = [change_samples[0]]
            for i in range(1, len(change_samples)):
                prev_start = filtered_change_samples[-1] // self.hop_length
                prev_end = change_samples[i] // self.hop_length

                if (change_samples[i] - filtered_change_samples[-1] < min_segment_length or
                    (i < len(change_samples) - 1 and
                     self.should_merge_segments(
                        prev_start,
                        prev_end,
                        change_samples[i] // self.hop_length,
                        change_samples[i + 1] // self.hop_length,
                        feat['mfcc'], feat['chroma'], feat['spectral_contrast'],
                        feat['tonnetz']))):
                    continue
                
                filtered_change_samples.append(change_samples[i])
            
            seg_time = ((np.array(filtered_change_samples) + start) / sr)
            seg_time = np.concatenate([[seg_time[0]], seg_time[1:][np.diff(seg_time) > measure_length / sr]])
            segment_time.extend(seg_time)

            self.combined_score = np.concatenate([self.combined_score, combined_score])

        self.segment_time = np.array(segment_time)
        self.segment_time = np.unique(self.segment_time)

        print(self.segment_time)

    def separate(self):
        sr = sf.info(f'tmp/{self.filename}/vocals.wav').samplerate
        vocal = librosa.load(f'tmp/{self.filename}/vocals.wav', sr=sr)[0]
        self.verse_separate(vocal, sr)
        self.detect_segment()

        save_dir = f'./output/{self.filename}'

        os.makedirs(save_dir, exist_ok=True)

        for file in os.listdir(save_dir):
            os.remove(os.path.join(save_dir, file))

        for i, seg in enumerate(self.segment_time):
            start = int(seg * sr)
            if i == len(self.segment_time) - 1:
                end = len(self.data)
            else:
                end = int(self.segment_time[i+1] * sr)
            save_file_path = os.path.join(save_dir, f'segment_{i}.wav')
            sf.write(save_file_path, self.data[start:end], sr)

        print("Separation is done")

        import matplotlib.pyplot as plt
        plt.plot(self.combined_score)
        for verse in self.verse_time:
            plt.axvline(verse * sr / self.hop_length, color='green')
        for seg in self.segment_time:
            plt.axvline(seg * sr / self.hop_length, color='red')
        plt.savefig(os.path.join(save_dir, 'combined_score.png'), dpi=600)


if __name__ == "__main__":
    filepath = sys.argv[1]
    separator = Separator(filepath)
    separator.separate()