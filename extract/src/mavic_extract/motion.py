import cv2
import numpy as np
from math import atan2, pi
from scipy.stats import entropy
from scenedetect import detect, AdaptiveDetector


def circular_mean_std(angles):
    sin_sum = np.sum(np.sin(angles))
    cos_sum = np.sum(np.cos(angles))
    mean_angle = atan2(sin_sum, cos_sum)
    mean_angle = mean_angle if mean_angle >= 0 else mean_angle + 2 * pi
    angular_deviation = np.sqrt(-2 * np.log(np.sqrt(sin_sum**2 + cos_sum**2) / len(angles)))
    return mean_angle, angular_deviation

def circular_volatility(angles):
    diffs = np.diff(angles)
    diffs = np.arctan2(np.sin(diffs), np.cos(diffs))  # Normalize differences
    return np.std(diffs)

def compute_entropy(image):
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256), density=True)
    return entropy(hist, base=2)

def detect_shots(video_path):
    detector = AdaptiveDetector()
    scene_list = detect(str(video_path), detector)
    cap = cv2.VideoCapture(str(video_path))
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return len(scene_list) / (duration / 60) if duration > 0 else 0

def process_video(video_path, sampling_interval=10):
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_frames = np.arange((fps * sampling_interval) // 2, frame_count, fps * sampling_interval)
    
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError('Cannot decode first frame')
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    magnitudes, angles, spatial_entropies, info_entropies = [], [], [], []
    
    pairs = []
    previous = 0
    for idx in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        magnitudes.append(np.mean(mag))
        angles.append(np.mean(ang))
        spatial_entropies.append(compute_entropy(gray))
        info_entropies.append(compute_entropy(mag))
        
        pairs.append([previous, int(idx)])
        previous = int(idx)
        prev_gray = gray
    
    cap.release()
    
    if not magnitudes:
        return {"avg_shots_per_min": detect_shots(video_path)}, {"observations": 0, "frame_pairs": []}
    
    angle_mean, angle_std = circular_mean_std(np.array(angles))
    avg_shots_per_min = detect_shots(video_path)
    
    return {

        'motion_magnitude_mean': np.mean(magnitudes),
        'motion_magnitude_std': np.std(magnitudes),
        'motion_magnitude_volatility': np.std(np.diff(magnitudes)),
        'motion_direction_mean': angle_mean,
        'motion_direction_std': angle_std,
        'motion_direction_volatility': circular_volatility(np.array(angles)),
        'spatial_entropy_mean': np.mean(spatial_entropies),
        'spatial_entropy_std': np.std(spatial_entropies),
        'spatial_entropy_volatility': np.std(np.diff(spatial_entropies)),
        'info_entropy_mean': np.mean(info_entropies),
        'info_entropy_std': np.std(info_entropies),
        'info_entropy_volatility': np.std(np.diff(info_entropies)),
        'avg_shots_per_min': avg_shots_per_min
    } , {"observations": len(pairs), "frame_pairs": pairs}
