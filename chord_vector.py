import numpy as np

root = 1.0
third = 1.0
fifth = 1.0
seventh = 1.0

template_major  = np.array([root, 0, 0, 0, third, 0, 0, fifth, 0, 0, 0, 0])
template_minor  = np.array([root, 0, 0, third, 0, 0, 0, fifth, 0, 0, 0, 0])
template_dim = np.array([root, 0, 0, third, 0, 0, fifth, 0, 0, 0, 0, 0])
template_aug = np.array([root, 0, 0, 0, third, 0, 0, 0, fifth, 0, 0, 0])
template_sus4 = np.array([root, 0, 0, 0, 0, third, 0, fifth, 0, 0, 0, 0])
template_sus2 = np.array([root, 0, third, 0, 0, 0, 0, fifth, 0, 0, 0, 0])

templates = np.array([np.roll(template_major, i) for i in range(12)]
                     + [np.roll(template_minor, i) for i in range(12)]
                    + [np.roll(template_dim, i) for i in range(12)]
                    + [np.roll(template_aug, i) for i in range(12)]
                    + [np.roll(template_sus4, i) for i in range(12)]
                    + [np.roll(template_sus2, i) for i in range(12)])

chord_dic = {
    0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F", 6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B",
    12: "Cm", 13: "C#m", 14: "Dm", 15: "D#m", 16: "Em", 17: "Fm", 18: "F#m", 19: "Gm", 20: "G#m", 21: "Am", 22: "A#m", 23: "Bm",
    24: "Cdim", 25: "C#dim", 26: "Ddim", 27: "D#dim", 28: "Edim", 29: "Fdim", 30: "F#dim", 31: "Gdim", 32: "G#dim", 33: "Adim", 34: "A#dim", 35: "Bdim",
    36: "Caug", 37: "C#aug", 38: "Daug", 39: "D#aug", 40: "Eaug", 41: "Faug", 42: "F#aug", 43: "Gaug", 44: "G#aug", 45: "Aaug", 46: "A#aug", 47: "Baug",
    48: "Csus4", 49: "C#sus4", 50: "Dsus4", 51: "D#sus4", 52: "Esus4", 53: "Fsus4", 54: "F#sus4", 55: "Gsus4", 56: "G#sus4", 57: "Asus4", 58: "A#sus4", 59: "Bsus4",
    60: "Csus2", 61: "C#sus2", 62: "Dsus2", 63: "D#sus2", 64: "Esus2", 65: "Fsus2", 66: "F#sus2", 67: "Gsus2", 68: "G#sus2", 69: "Asus2", 70: "A#sus2", 71: "Bsus2"
    }