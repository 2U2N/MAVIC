from . import appearance_measures as metrics
measures = [
    ("color_warmth", metrics.calculate_color_warmth, ["mean", "sd", "vol"]),
    ("valence", metrics.calculate_valence_statistics, ["mean", "sd", "vol"]),
    ("dominance", metrics.calculate_dominance_statistics, ["mean", "sd", "vol"]),
    ("arousal", metrics.calculate_arousal_statistics, ["mean", "sd", "vol"]),
    ("colorfulness", metrics.calculate_colorfulness_statistics, ["mean", "sd", "vol"]),
    ("clarity", metrics.calculate_clarity_statistics, ["mean", "sd", "vol"]),
    ("gray_distribution_entropy", metrics.calculate_gray_distribution_entropy_statistics, ["mean", "sd", "vol"]),
    ("blurriness", metrics.calculate_blurriness_statistics, ["mean", "median", "sd", "vol"]),
    ("edge_points", metrics.calculate_edge_points_statistics, ["mean", "sd", "vol"])
]

def summarize(keyframe_dir):
    values = {}
    errors = {}
    for name, function, suffixes in measures:
        try:
            result = function(keyframe_dir)
            values.update({f"{name}_{suffix}": v for suffix, v in zip(suffixes, result, strict=True)})
        except (OSError, ValueError, RuntimeError, metrics.cv2.error) as exc:
            errors[function.__name__] = str(exc)
    for function in GROUPS:
        try:
            values.update(function(keyframe_dir))
        except (OSError, ValueError, RuntimeError, metrics.cv2.error) as exc:
            errors[function.__name__] = str(exc)
    return values, errors


def calculate_color_percentages_statistics(keyframe_dir):
    video_measurements = {}
    results = metrics.calculate_color_percentages_statistics(keyframe_dir)
    colors = ['black', 'gray', 'white', 'red', 'green', 'blue', 'yellow', 'orange', 'brown', 'pink', 'purple']
    statistics = ['mean', 'sd', 'vol']
    for i, col in enumerate(colors):
        for j, stat in enumerate(statistics):
            video_measurements[f"{col}%_{stat}"] = results[j][i]
    return video_measurements


def calculate_hsv_statistics(keyframe_dir):
    video_measurements = {}
    results = metrics.calculate_hsv_statistics(keyframe_dir)
    output_keys = ["mean_avg_value", "mean_avg_saturation", "mean_sd_value", "mean_sd_saturation",
                   "sd_avg_value", "sd_avg_saturation", "sd_sd_value", "sd_sd_saturation", 
                    "vol_avg_value", "vol_avg_saturation", "vol_sd_value", "vol_sd_saturation"]
    for i, key in enumerate(output_keys):
        video_measurements[f"{key}"] = results[i]
    return video_measurements


def calculate_hue_circular_variance_statistics(keyframe_dir):
    video_measurements = {}
    results = metrics.calculate_hue_circular_variance_statistics(keyframe_dir)
    output_keys = ["mean_circular_var_hue", "sd_circular_var_hue", "vol_circular_var_hue"]
    for i, key in enumerate(output_keys):
        video_measurements[f"{key}"] = results[i]
    return video_measurements


def calculate_hsv_wavelet_mean_statistics(keyframe_dir):
    video_measurements = {}
    results = metrics.calculate_hsv_wavelet_mean_statistics(keyframe_dir)
    keys = ['hue', 'saturation', 'value']
    statistics = ['mean', 'sd', 'vol']
    for i, key in enumerate(keys):
        for j, stat in enumerate(statistics):
            video_measurements[f"{key}_avg_wavelet_{stat}"] = results[j][i]
    return video_measurements


def calculate_glcm_statistics(keyframe_dir):
    video_measurements = {}
    results = metrics.calculate_glcm_statistics(keyframe_dir)
    keys = ['hue', 'saturation', 'value']
    features = ["contrast", "correlation", "energy", "homogeneity"]
    statistics = ['mean', 'sd', 'vol']
    for j, key in enumerate(keys):
        for i, stat in enumerate(statistics):
            for k, feature in enumerate(features):
                video_measurements[f"{key}_{feature}_{stat}"] = results[i][j][k]
    return video_measurements


def calculate_low_dof_statistics(keyframe_dir):
    video_measurements = {}
    results = metrics.calculate_low_dof_statistics(keyframe_dir)
    keys = ['hue', 'saturation', 'value']
    statistics = ['mean', 'sd', 'vol']
    for i, key in enumerate(keys):
        for j, stat in enumerate(statistics):
            video_measurements[f"{key}_low_dof_{stat}"] = results[j][i]
    return video_measurements


def calculate_rule_of_thirds_statistics(keyframe_dir):
    video_measurements = {}
    results = metrics.calculate_rule_of_thirds_statistics(keyframe_dir)
    keys = ['saturation', 'value']
    statistics = ['mean', 'sd', 'vol']
    for j, key in enumerate(keys):
        for i, stat in enumerate(statistics):
            video_measurements[f"avg_inner_{key}_{stat}"] = results[i][j]
    return video_measurements


def calculate_nb_persons_statistics(keyframe_dir):
    video_measurements = {}
    results = metrics.calculate_nb_persons_statistics(keyframe_dir)
    keys = ['nb', 'largest_confidence', 'avg_confidence', 'largest_bb', 'sum_of_bb']
    statistics = ['mean', 'sd', 'vol']
    for j, key in enumerate(keys):
        for i, stat in enumerate(statistics):
            video_measurements[f"{key}_people_{stat}"] = results[i][j]
    return video_measurements


def calculate_nb_faces_statistics(keyframe_dir):
    video_measurements = {}
    results = metrics.calculate_nb_faces_statistics(keyframe_dir)
    keys = ['nb', 'largest_confidence', 'avg_confidence', 'largest_bb', 'sum_of_bb']
    statistics = ['mean', 'sd', 'vol']
    for j, key in enumerate(keys):
        for i, stat in enumerate(statistics):
            video_measurements[f"{key}_faces_{stat}"] = results[i][j]
    return video_measurements


def calculate_nb_hands_statistics(keyframe_dir):
    video_measurements = {}
    results = metrics.calculate_nb_hands_statistics(keyframe_dir)
    keys = ['nb', 'largest_bb', 'sum_of_bb']
    statistics = ['mean', 'sd', 'vol']
    for j, key in enumerate(keys):
        for i, stat in enumerate(statistics):
            video_measurements[f"{key}_hands_{stat}"] = results[i][j]
    return video_measurements


GROUPS = (
    calculate_color_percentages_statistics,
    calculate_hsv_statistics,
    calculate_hue_circular_variance_statistics,
    calculate_hsv_wavelet_mean_statistics,
    calculate_glcm_statistics,
    calculate_low_dof_statistics,
    calculate_rule_of_thirds_statistics,
    calculate_nb_persons_statistics,
    calculate_nb_faces_statistics,
    calculate_nb_hands_statistics,
)

def measure(path, duration, preview):
    """Sample complete interval midpoints; save only the first frame for inspection."""
    import tempfile
    import subprocess
    from pathlib import Path
    import cv2
    from shutil import copyfile

    count = int(duration // 10)
    times = [5 + 10*i for i in range(count)] or [duration / 2]
    decoded = []
    with tempfile.TemporaryDirectory() as tmp:
        capture = cv2.VideoCapture(str(path))
        codec = int(capture.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little")
        fps = capture.get(cv2.CAP_PROP_FPS)
        try:
            for index, t in enumerate(times):
                frame_path = Path(tmp) / f"frame_{index:09d}.png"
                if codec.lower() == b"av01" or not fps:
                    subprocess.run(["ffmpeg","-loglevel","error","-ss",str(t),"-i",str(path),"-frames:v","1",str(frame_path)], check=True)
                else:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
                    ok, frame = capture.read()
                    if ok:
                        cv2.imwrite(str(frame_path), frame)
                if frame_path.is_file():
                    decoded.append(t)
                    if len(decoded) == 1:
                        copyfile(frame_path, preview)
        finally:
            capture.release()
        if not decoded:
            raise ValueError("No keyframe decoded")
        values, errors = summarize(tmp)
    return values, {"observations": len(decoded), "timestamps": decoded, "requested": len(times), "errors": errors}
