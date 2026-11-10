"""Ten-second acoustic measures and video-level aggregation."""
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import librosa

def extract_audio_from_video(video_path, audio_path):
    command = ['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_path, '-y']
    subprocess.run(command, check=True, capture_output=True)

def calculate_mfcc(y, sr, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs


def analyze_audio_features(segment, sr):
    # Calculate MFCCs and take the mean across time frames
    mfccs = calculate_mfcc(segment, sr)
    mean_mfccs = np.mean(mfccs, axis=1)
    
    # Calculate the frequency component of the volume contour around 4 Hz
    hop_length = 512
    rms = librosa.feature.rms(y=segment, hop_length=hop_length)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
    vol_contour_4hz = np.nan
    if len(times) > 1:
        rms_fft = np.fft.fft(rms)
        freqs = np.fft.fftfreq(len(rms), d=times[1] - times[0])
        idx_4hz = np.argmin(np.abs(freqs - 4.0))
        vol_contour_4hz = np.abs(rms_fft[idx_4hz])

    # Calculate the frequency centroid
    centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)[0]
    freq_centroid = np.mean(centroid)
    
    # Calculate the frequency bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)[0]
    freq_bandwidth = np.mean(bandwidth)
    
    # Calculate the energy ratio in the range [0–630) Hz
    stft = np.abs(librosa.stft(y=segment))
    freqs = librosa.fft_frequencies(sr=sr)
    total_energy = np.sum(stft)
    low_energy = np.sum(stft[freqs < 630])
    energy_ratio_0_630 = low_energy / total_energy if total_energy > 0 else 0

    # Calculate the energy ratio in the range [630-1720) Hz
    mid_energy = np.sum(stft[(freqs >= 630) & (freqs < 1720)])
    energy_ratio_630_1720 = mid_energy / total_energy if total_energy > 0 else 0
    
    # Calculate the energy ratio in the range [1720-4400) Hz
    high_energy = np.sum(stft[(freqs >= 1720) & (freqs < 4400)])
    energy_ratio_1720_4400 = high_energy / total_energy if total_energy > 0 else 0
    
    # Calculate non-silence ratio
    amplitude_threshold = 0.02  # Amplitude threshold for silence
    non_silence = np.sum(np.abs(segment) > amplitude_threshold)
    non_silence_ratio = non_silence / len(segment)
    
    # Calculate volume dynamic range
    volume_dynamic_range = np.max(rms) - np.min(rms)
    
    # Calculate zero crossing rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(segment)[0]
    mean_zero_crossing_rate = np.mean(zero_crossing_rate)
    
    # Calculate volume standard deviation
    volume_std_dev = np.std(rms)
    
    return {
        "mean_mfccs": mean_mfccs,
        "vol_contour_4hz": vol_contour_4hz,
        "freq_centroid": freq_centroid,
        "freq_bandwidth": freq_bandwidth,
        "energy_ratio_0_630": energy_ratio_0_630,
        "energy_ratio_630_1720": energy_ratio_630_1720,
        "energy_ratio_1720_4400": energy_ratio_1720_4400,
        "non_silence_ratio": non_silence_ratio,
        "volume_dynamic_range": volume_dynamic_range,
        "mean_zero_crossing_rate": mean_zero_crossing_rate,
        "volume_std_dev": volume_std_dev
    }

def analyze_audio_segments(video_path, audio_path, segment_length=10):
    # Extract the audio from the video using ffmpeg

    extract_audio_from_video(video_path, audio_path)
    
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    
    # Determine the number of samples in each segment
    segment_samples = int(segment_length * sr)
    
    # Check if the audio is shorter than one segment
    if len(y) < segment_samples:
        # Handle this case by processing the entire audio as one segment
        segments = [y]
    else:
        # Frame the audio into segments
        segments = librosa.util.frame(y, frame_length=segment_samples, hop_length=segment_samples).T
        
    all_features = []
    for segment in segments:
        features = analyze_audio_features(segment, sr)
        all_features.append(features)
    
    return all_features


def calculate_volatility(values):
    differences = np.diff(values)
    volatility = np.std(differences) if not (len(differences) == 0 or np.all(differences == 0)) else np.nan
    return volatility

def aggregate_features(features_list):
    aggregated_features = {}
    # Initialize lists for MFCCs
    mfccs_list = []
    
    # Collect MFCCs separately
    for features in features_list:
        mfccs_list.append(features["mean_mfccs"])
    
    # Convert list of arrays to 2D array
    mfccs_array = np.array(mfccs_list)
    # Compute mean and std across segments for each MFCC coefficient
    aggregated_features["mean_mfccs"] = np.mean(mfccs_array, axis=0)
    aggregated_features["std_mfccs"] = np.std(mfccs_array, axis=0)
    # Compute volatility for MFCCs
    mfccs_diffs = np.diff(mfccs_array, axis=0)
    aggregated_features["volatility_mfccs"] = np.std(mfccs_diffs, axis=0)
    
    # Other scalar features
    scalar_keys = [key for key in features_list[0].keys() if key != "mean_mfccs"]
    for key in scalar_keys:
        all_values = [features[key] for features in features_list]
        mean_value = np.mean(all_values)
        std_value = np.std(all_values)
        volatility_value = calculate_volatility(all_values)
        aggregated_features[f"mean_{key}"] = mean_value
        aggregated_features[f"std_{key}"] = std_value
        aggregated_features[f"volatility_{key}"] = volatility_value
    
    return aggregated_features



def measure(path):
    with tempfile.TemporaryDirectory() as folder:
        segments = analyze_audio_segments(path, str(Path(folder) / "audio.wav"))
    result = {}
    for name, value in aggregate_features(segments).items():
        if isinstance(value, np.ndarray):
            result.update({f"{name}_{index}": v for index, v in enumerate(value, 1)})
        else:
            result[name] = value
    return result, {"observations": len(segments)}
