"""Pipeline integrado YOLO -> U-Net -> SlowFast para um vídeo curto."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .models import ActionClassifier, PersonDetector, PersonSegmenter
from .viz import VideoWriter, add_banner, draw_detections, read_frames, scale_dets, upscale


class Models:
    """Carrega os três modelos uma única vez para reutilizar entre vídeos."""

    def __init__(self, action: bool = True):
        t = time.time()
        self.detector = PersonDetector()
        self.segmenter = PersonSegmenter()
        self.action = ActionClassifier() if action else None
        print(f"[modelos] carregados em {time.time() - t:.1f}s")


def windows(n_frames: int, size: int, stride: int) -> list[tuple[int, int]]:
    """Janelas [ini, fim) consecutivas; a última é alinhada ao fim do vídeo."""
    if n_frames <= size:
        return [(0, n_frames)]
    starts = list(range(0, n_frames - size + 1, stride))
    if starts[-1] + size < n_frames:
        starts.append(n_frames - size)
    return [(s, s + size) for s in starts]


def person_crop(frames, dets_per_frame, a, b, margin=0.15):
    """Recorte quadrado em torno da união das caixas de pessoa na janela (extensão opcional)."""
    boxes = [d.box for f in range(a, b) for d in dets_per_frame[f]]
    if not boxes:
        return None
    H, W = frames[a].shape[:2]
    x1, y1 = min(b_[0] for b_ in boxes), min(b_[1] for b_ in boxes)
    x2, y2 = max(b_[2] for b_ in boxes), max(b_[3] for b_ in boxes)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = min(max(x2 - x1, y2 - y1) * (1 + 2 * margin), W, H)
    x1 = int(np.clip(cx - side / 2, 0, W - side))
    y1 = int(np.clip(cy - side / 2, 0, H - side))
    side = int(side)
    return [f[y1:y1 + side, x1:x1 + side] for f in frames[a:b]], (x1, y1, x1 + side, y1 + side)


def process_video(video: str | Path, out_dir: str | Path, models: Models, *, max_frames: int | None = None,
                  window: int = 32, stride: int = 16, crop_action: bool = True,
                  moments: list[float] = (0.2, 0.5, 0.8)) -> dict:
    video, out_dir = Path(video), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, fps = read_frames(video, max_frames)
    n = len(frames)
    print(f"[{video.name}] {n} quadros, {fps:.1f} fps, {frames[0].shape[1]}x{frames[0].shape[0]}")

    # 1) YOLO + U-Net quadro a quadro --------------------------------------------------------
    models.detector.reset()
    dets_per_frame, t0 = [], time.time()
    with open(out_dir / "detections.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame", "time_s", "track_id", "x1", "y1", "x2", "y2", "det_conf",
                     "mask_pixels", "mask_fill_ratio", "mask_mean_prob"])
        for i, f in enumerate(frames):
            dets = models.detector(f)
            for d in dets:
                models.segmenter.segment(f, d)
                area = max(1, (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))
                px = int(d.mask.sum())
                wr.writerow([i, f"{i / fps:.3f}", d.track_id, *d.box, f"{d.conf:.4f}", px,
                             f"{px / area:.4f}", f"{d.mask_prob:.4f}"])
            dets_per_frame.append(dets)
    t_seg = time.time() - t0
    print(f"  YOLO+U-Net: {t_seg:.1f}s ({n / max(t_seg, 1e-6):.1f} qps)")

    # 2) SlowFast em janelas temporais consecutivas ---------------------------------------------
    wins = windows(n, window, stride)
    results = []  # por janela: {"full": [...top5], "crop": [...] | None}
    t0 = time.time()
    if models.action is not None:
        for a, b in wins:
            r = {"range": (a, b), "full": models.action(frames[a:b])}
            if crop_action:
                pc = person_crop(frames, dets_per_frame, a, b)
                r["crop"] = models.action(pc[0]) if pc else None
                r["crop_box"] = pc[1] if pc else None
            results.append(r)
    t_act = time.time() - t0
    print(f"  SlowFast: {len(wins)} janelas em {t_act:.1f}s")

    with open(out_dir / "actions.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["window", "start_frame", "end_frame", "t_start", "t_end", "mode",
                     "top1", "conf1", "top2", "conf2", "top3", "conf3"])
        for k, r in enumerate(results):
            a, b = r["range"]
            for mode in ("full", "crop"):
                top = r.get(mode)
                if not top:
                    continue
                wr.writerow([k, a, b - 1, f"{a / fps:.2f}", f"{(b - 1) / fps:.2f}", mode,
                             *[x for name, c in top[:3] for x in (name, f"{c:.4f}")]])

    # janela "responsável" por cada quadro = a de centro mais próximo
    centers = np.array([(a + b - 1) / 2 for a, b in wins])
    frame_win = [int(np.abs(centers - i).argmin()) for i in range(n)] if results else [None] * n

    # 3) Visualização -------------------------------------------------------------------------
    writer = VideoWriter(out_dir / "annotated.mp4", fps)
    moment_frames = sorted({min(n - 1, int(m * n)) for m in moments})
    snaps = {}
    for i, f in enumerate(frames):
        big = upscale(f)
        s = big.shape[1] / f.shape[1]
        img = draw_detections(big, scale_dets(dets_per_frame[i], s))
        lines = [(f"quadro {i}/{n - 1}  t={i / fps:5.2f}s  pessoas={len(dets_per_frame[i])}", (200, 200, 200))]
        if results:
            k = frame_win[i]
            r = results[k]
            name, c = r["full"][0]
            lines.append((f"Acao (quadro inteiro): {name}  {c:.0%}   [janela {k}: q{r['range'][0]}-{r['range'][1] - 1}]",
                          (80, 255, 80) if c >= 0.5 else (0, 200, 255)))
            if crop_action and r.get("crop"):
                n2, c2 = r["crop"][0]
                lines.append((f"Acao (recorte pessoa): {n2}  {c2:.0%}", (255, 200, 120)))
                cb = r["crop_box"]
                cv2.rectangle(img, tuple(int(v * s) for v in cb[:2]), tuple(int(v * s) for v in cb[2:]),
                              (255, 200, 120), 1, cv2.LINE_4)
        img = add_banner(img, lines)
        writer.write(img)
        if i in moment_frames:
            snaps[i] = img
    writer.close()

    snap_dir = out_dir / "moments"
    snap_dir.mkdir(exist_ok=True)
    moment_info = []
    for i, img in snaps.items():
        p = snap_dir / f"frame_{i:04d}.png"
        cv2.imwrite(str(p), img)
        info = {"frame": i, "time_s": round(i / fps, 2), "image": str(p.relative_to(out_dir)),
                "detections": [{"id": d.track_id, "box": d.box, "conf": round(d.conf, 3),
                                "mask_fill": round(float(d.mask.sum()) / max(1, (d.box[2] - d.box[0]) * (d.box[3] - d.box[1])), 3)}
                               for d in dets_per_frame[i]]}
        if results:
            r = results[frame_win[i]]
            info["window"] = frame_win[i]
            info["action_full"] = r["full"][:3]
            info["action_crop"] = r.get("crop", [])[:3] if r.get("crop") else None
        moment_info.append(info)

    _plot_actions(results, fps, n, out_dir / "actions_plot.png", video.stem)

    ids = {d.track_id for ds in dets_per_frame for d in ds}
    fills = [float(d.mask.sum()) / max(1, (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))
             for ds in dets_per_frame for d in ds]
    summary = {
        "video": video.name, "frames": n, "fps": fps, "size": [frames[0].shape[1], frames[0].shape[0]],
        "frames_with_person": sum(1 for ds in dets_per_frame if ds),
        "mean_people_per_frame": round(float(np.mean([len(ds) for ds in dets_per_frame])), 2),
        "unique_track_ids": len(ids),
        "mean_det_conf": round(float(np.mean([d.conf for ds in dets_per_frame for d in ds] or [0])), 3),
        "mean_mask_fill_ratio": round(float(np.mean(fills or [0])), 3),
        "window": window, "stride": stride, "n_windows": len(results),
        "windows": [{"k": k, "frames": r["range"], "full": r["full"][:3], "crop": (r.get("crop") or [])[:3]}
                    for k, r in enumerate(results)],
        "moments": moment_info,
        "time_yolo_unet_s": round(t_seg, 1), "time_slowfast_s": round(t_act, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  saída: {out_dir}")
    return summary


def _plot_actions(results, fps, n, path, title):
    if not results:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 2.8), dpi=130)
    for mode, color, ls in (("full", "#2a6fdb", "-"), ("crop", "#e07b00", "--")):
        pts = [((r["range"][0] + r["range"][1] - 1) / 2 / fps, r[mode][0]) for r in results if r.get(mode)]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ax.plot(xs, [p[1][1] for p in pts], ls, marker="o", color=color,
                label="quadro inteiro" if mode == "full" else "recorte da pessoa")
        prev = None
        for x, (name, c) in pts:
            if name != prev:
                ax.annotate(name, (x, c), textcoords="offset points", xytext=(0, 6 if mode == "full" else -12),
                            fontsize=7, color=color, ha="center")
                prev = name
    ax.set_xlim(0, n / fps)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("tempo (s) - centro da janela")
    ax.set_ylabel("confiança top-1")
    ax.set_title(f"SlowFast por janela - {title}", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
