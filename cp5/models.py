"""Wrappers dos três modelos pré-treinados usados no CP5 (nenhum é re-treinado).

- PersonDetector  -> YOLO11n (Ultralytics, COCO): detecção + rastreamento da classe "person".
- PersonSegmenter -> U^2-Net "human_seg" (U-Net aninhada, pesos públicos do rembg): máscara da
                     pessoa dentro do recorte da caixa, projetada de volta no quadro original.
- ActionClassifier -> SlowFast R50 8x8 (PyTorchVideo, Kinetics-400): ação em janelas de quadros.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "torch_cache"))


# --------------------------------------------------------------------------- YOLO
@dataclass
class Detection:
    track_id: int
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 em pixels do quadro original
    conf: float
    mask: np.ndarray | None = field(default=None, repr=False)  # bool HxW (quadro inteiro)
    mask_prob: float = 0.0  # probabilidade média da U-Net dentro da máscara


class PersonDetector:
    def __init__(self, weights: str = "yolo11n.pt", conf: float = 0.35, track: bool = True):
        from ultralytics import YOLO

        path = MODELS_DIR / weights
        self.model = YOLO(str(path) if path.exists() else weights)
        self.conf = conf
        self.track = track

    def reset(self):
        # Limpa o estado do rastreador entre vídeos diferentes.
        if self.model.predictor is not None and hasattr(self.model.predictor, "trackers"):
            for t in self.model.predictor.trackers:
                t.reset()

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        kw = dict(classes=[0], conf=self.conf, verbose=False)
        res = (self.model.track(frame, persist=True, tracker="bytetrack.yaml", **kw)
               if self.track else self.model.predict(frame, **kw))[0]
        dets = []
        if res.boxes is None:
            return dets
        ids = res.boxes.id.int().tolist() if res.boxes.id is not None else [-1] * len(res.boxes)
        for tid, xyxy, c in zip(ids, res.boxes.xyxy.tolist(), res.boxes.conf.tolist()):
            x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
            dets.append(Detection(tid, (x1, y1, x2, y2), float(c)))
        return dets


# --------------------------------------------------------------------------- U-Net
class PersonSegmenter:
    """U^2-Net (Qin et al., 2020) treinada para segmentação de pessoas, via ONNX Runtime."""

    SIZE = 320
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, weights: str = "u2net_human_seg.onnx", threshold: float = 0.5,
                 margin: float = 0.25):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(str(MODELS_DIR / weights), providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.threshold = threshold
        self.margin = margin

    def predict_prob(self, bgr: np.ndarray) -> np.ndarray:
        """Mapa de probabilidade [0,1] do tamanho da imagem de entrada."""
        h, w = bgr.shape[:2]
        x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (self.SIZE, self.SIZE),
                       interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
        x = (x / max(x.max(), 1e-6) - self.MEAN) / self.STD
        x = x.transpose(2, 0, 1)[None]
        d1 = self.sess.run(None, {self.inp: x})[0][0, 0]
        d1 = (d1 - d1.min()) / max(d1.max() - d1.min(), 1e-6)
        return cv2.resize(d1, (w, h), interpolation=cv2.INTER_LINEAR)

    def square_crop(self, frame: np.ndarray, box):
        """Recorte QUADRADO centrado na caixa (com margem). A U-Net recebe 320x320; recortar a
        caixa alta e estreita e esticá-la deforma a pessoa e degrada muito a máscara."""
        x1, y1, x2, y2 = box
        side = int(max(x2 - x1, y2 - y1) * (1 + 2 * self.margin))
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        sx, sy = cx - side // 2, cy - side // 2
        H, W = frame.shape[:2]
        pad = [max(0, -sy), max(0, sy + side - H), max(0, -sx), max(0, sx + side - W)]
        crop = frame[max(0, sy):min(H, sy + side), max(0, sx):min(W, sx + side)]
        if any(pad):  # completa com borda ESPELHADA quando o quadrado sai do quadro (a replicada
            # cria listras que confundem a U-Net: preenchimento médio 0,41 -> 0,47 em V1, 0,37 -> 0,48 em V2)
            crop = cv2.copyMakeBorder(crop, *pad, cv2.BORDER_REFLECT)
        return crop, (sx, sy, side)

    def segment(self, frame: np.ndarray, det: Detection) -> None:
        """Recorta a região detectada, segmenta e projeta a máscara de volta no quadro."""
        H, W = frame.shape[:2]
        full = np.zeros((H, W), bool)
        if det.box[2] - det.box[0] < 4 or det.box[3] - det.box[1] < 4:
            det.mask = full
            return
        crop, (sx, sy, side) = self.square_crop(frame, det.box)
        prob = self.predict_prob(crop)
        crop_mask = prob > self.threshold
        # projeção de volta: coordenadas do recorte -> coordenadas do quadro original
        fx1, fy1, fx2, fy2 = max(0, sx), max(0, sy), min(W, sx + side), min(H, sy + side)
        full[fy1:fy2, fx1:fx2] = crop_mask[fy1 - sy:fy2 - sy, fx1 - sx:fx2 - sx]
        # A máscara só vale dentro da caixa do YOLO (evita "vazar" para pessoas vizinhas).
        x1, y1, x2, y2 = det.box
        box_only = np.zeros_like(full)
        box_only[max(0, y1):y2, max(0, x1):x2] = True
        det.mask = full & box_only
        inside = prob[fy1 - sy:fy2 - sy, fx1 - sx:fx2 - sx][det.mask[fy1:fy2, fx1:fx2]]
        det.mask_prob = float(inside.mean()) if inside.size else 0.0


# --------------------------------------------------------------------------- SlowFast
class ActionClassifier:
    NUM_FRAMES = 32  # via rápida; a via lenta usa 32 / ALPHA = 8 quadros
    ALPHA = 4
    SIDE = 256
    MEAN = np.array([0.45, 0.45, 0.45], np.float32)
    STD = np.array([0.225, 0.225, 0.225], np.float32)

    def __init__(self):
        import torch

        self.torch = torch
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        self.model = torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r50",
                                    pretrained=True, trust_repo=True, verbose=False).eval()
        raw = json.loads((MODELS_DIR / "kinetics_classnames.json").read_text(encoding="utf-8"))
        self.classes = [""] * len(raw)
        for k, v in raw.items():
            self.classes[v] = k.strip('"')

    def _prep(self, frames: list[np.ndarray]) -> list:
        torch = self.torch
        idx = np.linspace(0, len(frames) - 1, self.NUM_FRAMES).round().astype(int)
        clip = []
        for i in idx:
            f = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            h, w = f.shape[:2]
            s = self.SIDE / min(h, w)
            f = cv2.resize(f, (max(self.SIDE, round(w * s)), max(self.SIDE, round(h * s))))
            h, w = f.shape[:2]
            y0, x0 = (h - self.SIDE) // 2, (w - self.SIDE) // 2
            clip.append(f[y0:y0 + self.SIDE, x0:x0 + self.SIDE])
        x = (np.stack(clip).astype(np.float32) / 255.0 - self.MEAN) / self.STD  # T H W C
        fast = torch.from_numpy(x).permute(3, 0, 1, 2).contiguous()  # C T H W
        slow = torch.index_select(fast, 1, torch.linspace(0, fast.shape[1] - 1, fast.shape[1] // self.ALPHA).long())
        return [slow[None], fast[None]]

    def __call__(self, frames: list[np.ndarray], topk: int = 5) -> list[tuple[str, float]]:
        with self.torch.no_grad():
            probs = self.torch.softmax(self.model(self._prep(frames)), dim=1)[0]
        conf, idx = probs.topk(topk)
        return [(self.classes[i], float(c)) for i, c in zip(idx.tolist(), conf.tolist())]
