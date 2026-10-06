"""Passo 1 do roteiro: executa os três modelos SEPARADAMENTE (sem integração).

- YOLO sozinho      -> só caixas;
- U-Net sozinha     -> U^2-Net aplicada ao quadro INTEIRO (sem caixa do YOLO);
- SlowFast sozinho  -> só a ação por janela, no quadro inteiro.

Uso: python run_individual.py videos/archery.mp4 videos/arm_wrestling.mp4
"""
import json
import sys
from pathlib import Path

import numpy as np

from cp5.models import Detection
from cp5.pipeline import Models, windows
from cp5.viz import VideoWriter, add_banner, draw_detections, read_frames, scale_dets, upscale


def run(video: Path, models: Models, max_frames=None):
    out = Path("outputs") / video.stem / "individual"
    out.mkdir(parents=True, exist_ok=True)
    frames, fps = read_frames(video, max_frames)
    n = len(frames)
    stats = {"video": video.name, "frames": n}

    # YOLO sozinho
    models.detector.reset()
    w = VideoWriter(out / "yolo_only.mp4", fps)
    counts = []
    for i, f in enumerate(frames):
        dets = models.detector(f)
        counts.append(len(dets))
        big = upscale(f)
        img = draw_detections(big, scale_dets(dets, big.shape[1] / f.shape[1]), draw_mask=False)
        w.write(add_banner(img, [(f"YOLO sozinho  quadro {i}  pessoas={len(dets)}", (200, 200, 200))]))
    w.close()
    stats["yolo_mean_people"] = round(float(np.mean(counts)), 2)
    stats["yolo_frames_without_person"] = int(sum(c == 0 for c in counts))
    print(f"[{video.name}] YOLO ok")

    # U-Net sozinha (quadro inteiro)
    w = VideoWriter(out / "unet_only.mp4", fps)
    cover = []
    for i, f in enumerate(frames):
        mask = models.segmenter.predict_prob(f) > models.segmenter.threshold
        cover.append(float(mask.mean()))
        d = Detection(0, (0, 0, 0, 0), 0.0, mask)
        big = upscale(f)
        img = draw_detections(big, scale_dets([d], big.shape[1] / f.shape[1]), draw_box=False)
        w.write(add_banner(img, [(f"U-Net sozinha (quadro inteiro)  quadro {i}  area={mask.mean():.1%}",
                                  (200, 200, 200))]))
    w.close()
    stats["unet_mean_mask_area_fraction"] = round(float(np.mean(cover)), 4)
    print(f"[{video.name}] U-Net ok")

    # SlowFast sozinho
    wins = windows(n, 32, 16)
    preds = [models.action(frames[a:b]) for a, b in wins]
    centers = np.array([(a + b - 1) / 2 for a, b in wins])
    w = VideoWriter(out / "slowfast_only.mp4", fps)
    for i, f in enumerate(frames):
        k = int(np.abs(centers - i).argmin())
        name, c = preds[k][0]
        w.write(add_banner(upscale(f), [(f"SlowFast sozinho  quadro {i}  janela {k}", (200, 200, 200)),
                                        (f"{name}  {c:.0%}", (80, 255, 80))]))
    w.close()
    stats["slowfast_windows"] = [{"frames": list(wins[k]), "top3": preds[k][:3]} for k in range(len(wins))]
    print(f"[{video.name}] SlowFast ok")
    (out / "individual_summary.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    models = Models()
    for v in sys.argv[1:] or ["videos/archery.mp4", "videos/arm_wrestling.mp4"]:
        run(Path(v), models)
