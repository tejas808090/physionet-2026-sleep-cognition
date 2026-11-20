"""Write one synthetic PSG + CAISR annotation pair to check the signal features end to end.
Known content: 10 Hz alpha in wake, 13 Hz spindle band + delta in NREM, theta in REM, 60 bpm ECG, SpO2 96 with dips.
"""
import sys
from pathlib import Path

import edfio
import numpy as np

out = Path(sys.argv[1])
rng = np.random.default_rng(0)
H, FS = 6, 200
n_ep = H * 120
stages = np.full(n_ep, 2)
stages[:40] = 5; stages[200:260] = 1; stages[300:360] = 4; stages[500:520] = 5; stages[600:680] = 4; stages[-10:] = 9
t = np.arange(H * 3600 * FS) / FS
st_s = np.repeat(stages, 30 * FS)
pink = np.cumsum(rng.standard_normal(len(t))) ; pink -= np.convolve(pink, np.ones(FS * 2) / (FS * 2), mode="same")
eeg = 3 * pink + 5 * rng.standard_normal(len(t))
eeg += np.where(st_s == 5, 20 * np.sin(2 * np.pi * 10 * t), 0)
eeg += np.where(np.isin(st_s, (1, 2)), 8 * np.sin(2 * np.pi * 13 * t) + 30 * np.sin(2 * np.pi * 1 * t), 0)
eeg += np.where(st_s == 4, 12 * np.sin(2 * np.pi * 6 * t), 0)
ecg = np.zeros(len(t)); beat = np.arange(0, len(t), FS)  # 60 bpm
jit = rng.integers(-6, 7, len(beat)); idx = np.clip(beat + jit, 0, len(t) - 1); ecg[idx] = 1.0
ecg = np.convolve(ecg, np.hanning(12), mode="same") * 1000 + 20 * rng.standard_normal(len(t))
spo2 = np.full(H * 3600, 96.0); 
for s in range(600, H * 3600 - 60, 600): spo2[s:s + 30] = 91
mk = lambda x, label, fs, unit: edfio.EdfSignal(np.ascontiguousarray(x), sampling_frequency=fs, label=label, physical_dimension=unit)
m2 = 2 * rng.standard_normal(len(t))
edfio.Edf([mk(eeg + m2, "C3", FS, "uV"), mk(m2, "M2", FS, "uV"), mk(eeg, "F4-M1", FS, "uV"), mk(ecg, "EKG", FS, "uV"),
           mk(spo2, "SaO2", 1, "%")]).write(out / "physiological_data/X/sub-X1_ses-1.edf")
z = np.zeros
edfio.Edf([mk(z(H * 7200), "arousal_caisr", 2, ""), mk(z(H * 3600), "resp_caisr", 1, ""), mk(z(H * 3600), "limb_caisr", 1, ""),
           mk(stages.astype(float), "stage_caisr", 1 / 30, "")]).write(out / "algorithmic_annotations/X/sub-X1_ses-1_caisr_annotations.edf")
print("ok")
