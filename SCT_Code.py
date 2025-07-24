import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# Define EEG bands
BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 45)
}

# ---- Chirplet Transform Function ----
def chirplet_transform(signal, fs, f0, alpha, sigma=0.1):
    n = len(signal)
    t = np.linspace(-n / (2 * fs), n / (2 * fs), n)
    g = np.exp(-t**2 / (2 * sigma**2))
    chirp = np.exp(2j * np.pi * (f0 * t + 0.5 * alpha * t**2))
    kernel = g * chirp
    analytic = hilbert(signal)
    ct = np.abs(np.convolve(analytic, np.conj(kernel[::-1]), mode='same'))
    return ct

# ---- Band-wise Energy Calculation ----
def compute_band_energy_chirplet(signal, fs):
    energy_by_band = {}
    for band, (fmin, fmax) in BANDS.items():
        f0 = (fmin + fmax) / 2
        ct_pos = chirplet_transform(signal, fs, f0, +25)
        ct_neg = chirplet_transform(signal, fs, f0, -25)

        ct_fused = (ct_pos / np.max(ct_pos)) + (ct_neg / np.max(ct_neg))
        ct_fused /= 2
        energy_by_band[band] = np.mean(ct_fused**2)  # Power
    return energy_by_band

# ---- Extract Chirplet Features from One File ----
def extract_chirplet_features(file_path, label):
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    raw.pick("eeg")
    raw.set_montage("standard_1020")
    raw.resample(128)  # Optional downsample
    sfreq = raw.info['sfreq']
    data, _ = raw[:, :int(sfreq * 10)]  # Use first 10s

    features = {}
    for i, ch_name in enumerate(raw.ch_names):
        signal = data[i, :]
        band_feats = compute_band_energy_chirplet(signal, sfreq)
        for band, val in band_feats.items():
            features[f"{ch_name}_{band}"] = val
    features['label'] = label
    return features

# ---- Process a Folder of EEG Files ----
def process_folder(folder_path, label, output_csv_path):
    all_features = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".set"):
            full_path = os.path.join(folder_path, file)
            try:
                print(f"✅ Processing: {file}")
                feats = extract_chirplet_features(full_path, label)
                print(f"➡️  {file} features: {len(feats)} keys")
                all_features.append(feats)
            except Exception as e:
                print(f"❌ Failed on {file}: {e}")

    df = pd.DataFrame(all_features)

    # Normalize numeric features
    features_only = df.drop(columns=['label'])
    df_normalized = (features_only - features_only.mean()) / features_only.std()
    df_normalized['label'] = df['label']

    df_normalized.to_csv(output_csv_path, index=False)
    print(f"💾 Saved normalized chirplet features to {output_csv_path}")
