"""Resolve user-supplied detector files and verify the study versions."""
import hashlib
import os
from pathlib import Path

HASHES = {
    "deploy.prototxt": "e745f552db65437206ce713bbd8d2e332a522676ae533dda391f029002ac415f",
    "mobilenet_iter_73000.caffemodel": "52eed8be80522c152a17fb56740de705b79881bde1a167e0e747310523685fc7",
    "face_detection_yunet_2023mar.onnx": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
}

def asset(name):
    directory = Path(os.environ.get("MAVIC_MODELS", Path.home() / ".cache/mavic/models"))
    path = directory / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != HASHES[name]:
        raise ValueError(f"Install the matching {name} in {directory}. See MODELS.md.")
    return path

SOURCES = {
    "deploy.prototxt": "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt",
    "mobilenet_iter_73000.caffemodel": "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/mobilenet_iter_73000.caffemodel",
    "face_detection_yunet_2023mar.onnx": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
}

def prepare_detectors():
    """Download missing study assets into the local model cache."""
    from urllib.request import urlopen
    from tempfile import NamedTemporaryFile
    from shutil import copyfileobj
    directory = Path(os.environ.get("MAVIC_MODELS", Path.home() / ".cache/mavic/models"))
    directory.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        if (directory / name).exists():
            asset(name)
            continue
        temp = None
        try:
            with urlopen(url, timeout=90) as response, NamedTemporaryFile(dir=directory,delete=False) as f:
                temp = Path(f.name)
                copyfileobj(response,f)
            if name == "deploy.prototxt":
                temp.write_bytes(temp.read_bytes().rstrip(b"\r\n"))
            if hashlib.sha256(temp.read_bytes()).hexdigest() != HASHES[name]:
                raise ValueError(f"Unexpected weights for {name}. See MODELS.md.")
            temp.replace(directory/name)
        finally:
            if temp:
                temp.unlink(missing_ok=True)
