"""Testes do CP5. Rodar a partir da pasta do projeto:  python -m pytest -v tests

Usa trechos curtos dos vídeos (≈40 quadros) para caber em poucos minutos na CPU.
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cp5.models import Detection, PersonSegmenter  # noqa: E402
from cp5.pipeline import Models, person_crop, process_video, windows  # noqa: E402
from cp5.viz import VideoWriter, read_frames  # noqa: E402

VIDEOS = ROOT / "videos"
OUT = ROOT / "outputs"
PY = sys.executable


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def models():
    return Models()


def write_clip(path, frames, fps=30.0):
    w = VideoWriter(path, fps)
    for f in frames:
        w.write(f)
    w.close()
    return path


@pytest.fixture(scope="session")
def short_clip(tmp_path_factory):
    frames, _ = read_frames(VIDEOS / "archery.mp4", 40)
    return write_clip(tmp_path_factory.mktemp("clip") / "archery40.mp4", frames)


@pytest.fixture(scope="session")
def empty_scene_clip(tmp_path_factory):
    """Vídeo sintético sem nenhuma pessoa (gradiente que se move)."""
    yy, xx = np.mgrid[0:240, 0:320]
    frames = [np.dstack([(xx + 4 * i) % 256, yy % 256, np.full_like(xx, 90)]).astype(np.uint8) for i in range(36)]
    return write_clip(tmp_path_factory.mktemp("clip") / "sem_pessoa.mp4", frames)


# --------------------------------------------------------------------------- unidades
def test_windows_cover_whole_video():
    ws = windows(300, 32, 16)
    assert ws[0] == (0, 32) and ws[-1][1] == 300
    assert all(b - a == 32 for a, b in ws)
    assert all(ws[i + 1][0] - ws[i][0] <= 16 for i in range(len(ws) - 1))
    assert windows(20, 32, 16) == [(0, 20)]  # vídeo mais curto que a janela
    assert windows(32, 32, 16) == [(0, 32)]


@pytest.mark.parametrize("box", [(100, 50, 160, 200), (0, 0, 50, 230), (280, 120, 320, 240), (5, 100, 315, 140)])
def test_mask_projection_back_to_frame(box, monkeypatch):
    """A máscara feita no recorte quadrado (com padding nas bordas) tem que voltar exatamente
    para o mesmo lugar no quadro. 'U-Net' falsa: marca os pixels vermelhos do recorte."""
    frame = np.zeros((240, 320, 3), np.uint8)
    x1, y1, x2, y2 = box
    frame[y1 + 10:y2 - 10, x1 + 5:x2 - 5] = (0, 0, 255)
    seg = PersonSegmenter.__new__(PersonSegmenter)
    seg.threshold, seg.margin = 0.5, 0.25
    monkeypatch.setattr(seg, "predict_prob", lambda crop: (crop[..., 2] > 128).astype(np.float32), raising=False)
    d = Detection(1, box, 0.9)
    seg.segment(frame, d)
    expected = frame[..., 2] > 128
    assert d.mask.shape == frame.shape[:2]
    assert np.array_equal(d.mask, expected)


def test_square_crop_is_square_and_padded():
    seg = PersonSegmenter.__new__(PersonSegmenter)
    seg.margin = 0.25
    frame = np.random.randint(0, 255, (240, 320, 3), np.uint8)
    crop, (sx, sy, side) = seg.square_crop(frame, (0, 0, 40, 200))  # caixa colada no canto
    assert crop.shape[0] == crop.shape[1] == side
    assert sx < 0 and sy < 0  # saiu do quadro -> padding


def test_person_crop_square_inside_frame():
    frames = [np.zeros((240, 320, 3), np.uint8)] * 4
    dets = [[Detection(1, (250, 10, 318, 238), 0.9)]] * 4
    crops, (x1, y1, x2, y2) = person_crop(frames, dets, 0, 4)
    assert x2 - x1 == y2 - y1
    assert 0 <= x1 and 0 <= y1 and x2 <= 320 and y2 <= 240
    assert all(c.shape[:2] == (y2 - y1, x2 - x1) for c in crops)
    assert person_crop(frames, [[]] * 4, 0, 4) is None


def test_read_frames_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_frames(tmp_path / "nao_existe.mp4")
    (tmp_path / "vazio.mp4").write_bytes(b"")
    with pytest.raises((FileNotFoundError, ValueError)):
        read_frames(tmp_path / "vazio.mp4")


# --------------------------------------------------------------------------- modelos reais
def test_yolo_finds_two_archers(models):
    frames, _ = read_frames(VIDEOS / "archery.mp4", 1)
    models.detector.reset()
    dets = models.detector(frames[0])
    assert len(dets) == 2
    assert all(d.conf > 0.5 for d in dets)


def test_unet_mask_inside_box_and_reasonable(models):
    """Máscara nunca sai da caixa; preenchimento médio plausível para pessoas em pé (40-60%).
    Obs.: em alguns quadros isolados a U-Net pega só o tronco (limitação documentada no relatório)."""
    frames, _ = read_frames(VIDEOS / "archery.mp4", 150)
    models.detector.reset()
    fills = []
    for f in frames[::15]:
        for d in models.detector(f):
            models.segmenter.segment(f, d)
            x1, y1, x2, y2 = d.box
            outside = d.mask.copy()
            outside[y1:y2, x1:x2] = False
            assert not outside.any(), "máscara vazou para fora da caixa"
            assert d.mask.any(), "máscara vazia para pessoa grande e visível"
            fills.append(d.mask.sum() / ((x2 - x1) * (y2 - y1)))
    assert len(fills) == 20
    assert 0.35 < np.mean(fills) < 0.7, f"preenchimento médio estranho: {np.mean(fills):.2f}"


@pytest.mark.parametrize("video,label", [("archery.mp4", "archery"), ("arm_wrestling.mp4", "arm wrestling")])
def test_slowfast_classifies(models, video, label):
    frames, _ = read_frames(VIDEOS / video, 32)
    top = models.action(frames)
    assert top[0][0] == label
    confs = [c for _, c in top]
    assert len(top) == 5 and confs == sorted(confs, reverse=True) and 0 <= sum(confs) <= 1 + 1e-5


# --------------------------------------------------------------------------- pipeline completo
def test_process_video_outputs(models, short_clip, tmp_path):
    s = process_video(short_clip, tmp_path, models, window=32, stride=8)
    n = s["frames"]
    assert n == 40 and s["frames_with_person"] == 40 and s["unique_track_ids"] == 2
    for f in ("annotated.mp4", "detections.csv", "actions.csv", "actions_plot.png", "summary.json"):
        assert (tmp_path / f).stat().st_size > 0, f
    assert len(list((tmp_path / "moments").glob("*.png"))) == 3

    out_frames, _ = read_frames(tmp_path / "annotated.mp4")
    assert len(out_frames) == n
    assert out_frames[0].shape[1] >= 640  # vídeo ampliado para legibilidade

    rows = list(csv.DictReader(open(tmp_path / "detections.csv", encoding="utf-8")))
    assert {int(r["frame"]) for r in rows} == set(range(n))
    assert all(0 <= float(r["mask_fill_ratio"]) <= 1 for r in rows)

    acts = list(csv.DictReader(open(tmp_path / "actions.csv", encoding="utf-8")))
    assert {r["mode"] for r in acts} == {"full", "crop"}
    assert len(acts) == 2 * len(windows(n, 32, 8))
    assert all(r["top1"] == "archery" for r in acts)


def test_process_video_without_people(models, empty_scene_clip, tmp_path):
    s = process_video(empty_scene_clip, tmp_path, models)
    assert s["frames_with_person"] == 0
    assert all(w["crop"] == [] for w in s["windows"])  # sem pessoa -> sem recorte, sem erro
    assert (tmp_path / "annotated.mp4").exists()


def test_process_video_shorter_than_window(models, short_clip, tmp_path):
    s = process_video(short_clip, tmp_path, models, max_frames=20, crop_action=False)
    assert s["frames"] == 20 and s["n_windows"] == 1


def test_main_cli(short_clip, tmp_path):
    r = subprocess.run([PY, "-W", "ignore", str(ROOT / "main.py"), str(short_clip), "--out", str(tmp_path / "cli"),
                        "--max-frames", "34", "--window", "16", "--stride", "8", "--no-crop-action"],
                       cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr[-2000:]
    s = json.loads((tmp_path / "cli" / "summary.json").read_text(encoding="utf-8"))
    assert s["frames"] == 34 and s["window"] == 16 and s["n_windows"] == len(windows(34, 16, 8))
    assert all(w["crop"] == [] for w in s["windows"])


def test_main_cli_bad_video(tmp_path):
    r = subprocess.run([PY, "-W", "ignore", str(ROOT / "main.py"), str(tmp_path / "x.mp4")],
                       cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode != 0 and "Não foi possível abrir o vídeo" in r.stderr


def test_run_individual(models, short_clip, tmp_path, monkeypatch):
    import run_individual

    monkeypatch.chdir(tmp_path)
    run_individual.run(Path(short_clip), models, max_frames=34)
    d = tmp_path / "outputs" / Path(short_clip).stem / "individual"
    for f in ("yolo_only.mp4", "unet_only.mp4", "slowfast_only.mp4"):
        assert len(read_frames(d / f)[0]) == 34, f
    s = json.loads((d / "individual_summary.json").read_text(encoding="utf-8"))
    assert s["yolo_mean_people"] == 2.0 and 0 < s["unet_mean_mask_area_fraction"] < 0.5
    assert s["slowfast_windows"][0]["top3"][0][0] == "archery"


# --------------------------------------------------------------------------- entregáveis gerados
def load(v):
    return json.loads((OUT / v / "summary.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("v", ["archery", "arm_wrestling", "transition", "pedestrians"])
def test_experiment_outputs_consistent(v):
    s = load(v)
    frames, _ = read_frames(OUT / v / "annotated.mp4")
    assert len(frames) == s["frames"]
    rows = list(csv.DictReader(open(OUT / v / "detections.csv", encoding="utf-8")))
    assert len({int(r["frame"]) for r in rows}) == s["frames_with_person"]
    assert len(s["moments"]) == 3


def test_report_numbers_match_outputs():
    """Confere os números citados no relatório contra os arquivos de saída."""
    a, w, t, p = (load(v) for v in ("archery", "arm_wrestling", "transition", "pedestrians"))
    assert all(x["full"][0][0] == "archery" for x in a["windows"]) and len(a["windows"]) == 18
    assert all(x["full"][0][0] == "arm wrestling" for x in w["windows"]) and len(w["windows"]) == 17
    assert a["unique_track_ids"] == 2 and a["frames_with_person"] == 300
    assert w["unique_track_ids"] == 31 and w["frames"] - w["frames_with_person"] == 35
    w6, w7, w8 = t["windows"][6], t["windows"][7], t["windows"][8]
    assert w6["full"][0][0] == "archery" and w6["full"][1][0] == "arm wrestling"
    assert w7["full"][0][0] == "arm wrestling" and round(w7["full"][0][1], 2) == 0.94
    assert round(w7["crop"][0][1], 2) == 0.85 and round(w8["full"][0][1], 2) == 1.0
    cf = np.mean([x["full"][0][1] for x in p["windows"]])
    cc = np.mean([x["crop"][0][1] for x in p["windows"]])
    agree = np.mean([x["full"][0][0] == x["crop"][0][0] for x in p["windows"]])
    assert (round(cf, 2), round(cc, 2), round(agree, 2)) == (0.29, 0.40, 0.67)
    # transição: corte no quadro 119 (4 s de V1 a 29,97 fps)
    rows = list(csv.DictReader(open(OUT / "transition" / "detections.csv", encoding="utf-8")))
    ids_after_cut = {r["track_id"] for r in rows if int(r["frame"]) == 119}
    assert "2" in ids_after_cut


def test_report_pdf(tmp_path):
    from pypdf import PdfReader

    r = subprocess.run([PY, str(ROOT / "make_report.py")], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    reader = PdfReader(ROOT / "relatorio_cp5.pdf")
    assert len(reader.pages) <= 3
    text = " ".join(pg.extract_text() for pg in reader.pages)
    for must in ("Tabela 1", "Tabela 2", "Erros explicados", "E1", "E2", "Parecer dos integrantes",
                 "A caixa acompanha a pessoa?", "A máscara acompanha o contorno?",
                 "A ação prevista corresponde", "A confiança cai ou a classe muda"):
        assert must in text, must


def test_demo_video():
    frames_total = sum(load(v)["frames"] / load(v)["fps"] for v in ("archery", "arm_wrestling", "transition", "pedestrians"))
    cap = cv2.VideoCapture(str(OUT / "demo.mp4"))
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    assert (cap.get(3), cap.get(4)) == (960, 720)
    assert abs(dur - (frames_total + 4 * 2)) < 1.0  # vídeos + 4 cartões de 2 s


@pytest.mark.parametrize("url", __import__("download_models").FILES.values())
def test_download_urls_alive(url):
    import urllib.request

    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        assert r.status == 200
