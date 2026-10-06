"""Leitura/escrita de vídeo e desenho das três saídas (caixa, máscara, ação)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

PALETTE = [(255, 99, 71), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
           (70, 240, 240), (240, 50, 230), (210, 245, 60), (0, 128, 128), (170, 110, 40)]
BANNER_H = 74
MIN_WIDTH = 640


def color_for(track_id: int) -> tuple[int, int, int]:
    r, g, b = PALETTE[track_id % len(PALETTE)] if track_id >= 0 else (255, 225, 25)
    return (b, g, r)


def read_frames(path: str | Path, max_frames: int | None = None) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while max_frames is None or len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise ValueError(f"O vídeo não tem quadros legíveis: {path}")
    return frames, fps


class VideoWriter:
    """Grava H.264 com PyAV (toca no navegador/Teams); cai para mp4v do OpenCV se faltar o codec."""

    def __init__(self, path: str | Path, fps: float):
        self.path, self.fps = Path(path), fps
        self.container = self.stream = self.cv = None

    def write(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        if self.container is None and self.cv is None:
            try:
                import av

                self.container = av.open(str(self.path), "w")
                self.stream = self.container.add_stream("libx264", rate=round(self.fps))
                self.stream.width, self.stream.height = w - w % 2, h - h % 2
                self.stream.pix_fmt = "yuv420p"
                self.stream.options = {"crf": "20"}
            except Exception:
                self.container = None
                self.cv = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        if self.cv is not None:
            self.cv.write(frame)
            return
        import av

        frame = frame[: self.stream.height, : self.stream.width]
        for pkt in self.stream.encode(av.VideoFrame.from_ndarray(frame, format="bgr24")):
            self.container.mux(pkt)

    def close(self):
        if self.cv is not None:
            self.cv.release()
        elif self.container is not None:
            for pkt in self.stream.encode():
                self.container.mux(pkt)
            self.container.close()


def _text(img, s, org, scale=0.55, color=(255, 255, 255), bg=None):
    if bg is not None:  # etiqueta com fundo sólido (legível sobre qualquer imagem)
        (tw, th), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        x, y = org
        cv2.rectangle(img, (x - 2, y - th - 3), (x + tw + 2, y + base), bg, -1)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_detections(frame: np.ndarray, dets, draw_box=True, draw_mask=True) -> np.ndarray:
    out = frame.copy()
    for d in dets:
        c = color_for(d.track_id)
        if draw_mask and d.mask is not None and d.mask.any():
            layer = out.copy()
            layer[d.mask] = c
            out = cv2.addWeighted(layer, 0.45, out, 0.55, 0)
            cnts, _ = cv2.findContours(d.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, c, 1, cv2.LINE_AA)
    for d in dets:
        if not draw_box:
            continue
        c = color_for(d.track_id)
        x1, y1, x2, y2 = d.box
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        label = f"ID {d.track_id} {d.conf:.2f}" if d.track_id >= 0 else f"pessoa {d.conf:.2f}"
        _text(out, label, (x1 + 2, max(14, y1 - 4)), 0.42, (0, 0, 0), bg=c)
    return out


def upscale(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w >= MIN_WIDTH:
        return frame
    s = MIN_WIDTH / w
    return cv2.resize(frame, (MIN_WIDTH, round(h * s)), interpolation=cv2.INTER_CUBIC)


def add_banner(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    h, w = frame.shape[:2]
    banner = np.full((BANNER_H, w, 3), 24, np.uint8)
    for i, (s, c) in enumerate(lines[:3]):
        _text(banner, s, (8, 20 + i * 22), 0.52, c)
    return np.vstack([banner, frame])


def scale_dets(dets, s: float):
    """Escala caixas/máscaras para o quadro ampliado (somente para desenho)."""
    if s == 1.0:
        return dets
    from copy import copy

    out = []
    for d in dets:
        d2 = copy(d)
        d2.box = tuple(int(round(v * s)) for v in d.box)
        if d.mask is not None:
            h, w = d.mask.shape
            d2.mask = cv2.resize(d.mask.astype(np.uint8), (round(w * s), round(h * s)),
                                 interpolation=cv2.INTER_NEAREST).astype(bool)
        out.append(d2)
    return out
