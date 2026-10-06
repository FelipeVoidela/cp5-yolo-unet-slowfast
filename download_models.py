"""Baixa os pesos pré-treinados e os vídeos de teste (nenhum modelo é treinado no CP5)."""
import urllib.request
from pathlib import Path

FILES = {
    "models/u2net_human_seg.onnx": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx",
    "models/kinetics_classnames.json": "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json",
    "models/yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt",
    "videos/archery.mp4": "https://dl.fbaipublicfiles.com/pytorchvideo/projects/archery.mp4",
    "videos/arm_wrestling.mp4": "https://github.com/open-mmlab/mmaction2/raw/main/demo/demo.mp4",
    "videos/pedestrians.avi": "https://github.com/opencv/opencv/raw/4.x/samples/data/vtest.avi",
}

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    for dst, url in FILES.items():
        p = ROOT / dst
        if p.exists():
            print(f"ok      {dst}")
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        print(f"baixando {dst} ...")
        urllib.request.urlretrieve(url, p)

    # SlowFast R50 (Kinetics-400) via torch.hub -> models/torch_cache
    from cp5.models import ActionClassifier

    ActionClassifier()
    print("SlowFast ok")
