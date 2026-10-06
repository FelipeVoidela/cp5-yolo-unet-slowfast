"""Roda todos os experimentos do relatório e monta o vídeo de demonstração.

1. archery.mp4        - vídeo 1 (tiro com arco, 2 pessoas, câmera móvel)
2. arm_wrestling.mp4  - vídeo 2 (queda de braço, close-up, várias pessoas ao fundo)
3. transition.mp4     - fim do vídeo 1 + início do vídeo 2: testa se a classe/confiança muda na troca de ação
4. pedestrians.avi    - caso de falha: pessoas pequenas, 10 fps, ação fora das classes do Kinetics

Uso: python run_experiments.py [--skip-existing]
"""
import sys
from pathlib import Path

import cv2
import numpy as np

from cp5.pipeline import Models, process_video
from cp5.viz import VideoWriter, read_frames

VIDEOS = Path("videos")
OUT = Path("outputs")
EXPERIMENTS = [
    ("archery", VIDEOS / "archery.mp4", None),
    ("arm_wrestling", VIDEOS / "arm_wrestling.mp4", None),
    ("transition", VIDEOS / "transition.mp4", None),
    ("pedestrians", VIDEOS / "pedestrians.avi", 150),
]
TITLES = {
    "archery": "Video 1 - tiro com arco",
    "arm_wrestling": "Video 2 - queda de braco",
    "transition": "Transicao: arco -> queda de braco (corte em t=4s)",
    "pedestrians": "Caso de falha: pedestres pequenos, 10 fps",
}


def make_transition(path: Path, seconds: float = 4.0, size=(320, 240), fps=30.0):
    a, fa = read_frames(VIDEOS / "archery.mp4")
    b, fb = read_frames(VIDEOS / "arm_wrestling.mp4")
    clip = a[-int(seconds * fa):] + b[:int(seconds * fb)]
    w = VideoWriter(path, fps)
    for f in clip:
        w.write(cv2.resize(f, size, interpolation=cv2.INTER_AREA))
    w.close()


def letterbox(img, W=960, H=720):
    h, w = img.shape[:2]
    s = min(W / w, H / h)
    r = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((H, W, 3), np.uint8)
    y, x = (H - r.shape[0]) // 2, (W - r.shape[1]) // 2
    canvas[y:y + r.shape[0], x:x + r.shape[1]] = r
    return canvas


def make_demo(names, path=OUT / "demo.mp4", fps=30.0):
    w = VideoWriter(path, fps)
    for name in names:
        card = np.zeros((720, 960, 3), np.uint8)
        cv2.putText(card, TITLES[name], (40, 330), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(card, "caixa (YOLO) + mascara (U-Net) + acao (SlowFast)", (40, 390),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1, cv2.LINE_AA)
        for _ in range(int(fps * 2)):
            w.write(card)
        frames, src_fps = read_frames(OUT / name / "annotated.mp4")
        step = src_fps / fps  # reamostra para fps fixo mantendo a duração real
        for k in range(int(len(frames) / step)):
            w.write(letterbox(frames[min(len(frames) - 1, int(k * step))]))
    w.close()
    print(f"demo: {path}")


if __name__ == "__main__":
    skip = "--skip-existing" in sys.argv
    if not (VIDEOS / "transition.mp4").exists():
        make_transition(VIDEOS / "transition.mp4")
    models = None
    for name, video, max_frames in EXPERIMENTS:
        if skip and (OUT / name / "summary.json").exists():
            print(f"[{name}] já processado, pulando")
            continue
        models = models or Models()
        process_video(video, OUT / name, models, max_frames=max_frames)
    make_demo([n for n, _, _ in EXPERIMENTS])
