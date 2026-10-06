"""CP5 - Detecção de ações de pessoas com YOLO + U-Net + SlowFast.

Uso:
    python main.py videos/archery.mp4
    python main.py meu_video.mp4 --out outputs/meu_video --window 32 --stride 16
    python main.py videos/pedestrians.avi --max-frames 200 --no-crop-action
"""
import argparse
from pathlib import Path

from cp5.pipeline import Models, process_video


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="caminho do vídeo de entrada")
    ap.add_argument("--out", help="pasta de saída (padrão: outputs/<nome do vídeo>)")
    ap.add_argument("--max-frames", type=int, help="processa só os N primeiros quadros")
    ap.add_argument("--window", type=int, default=32, help="quadros por janela do SlowFast (padrão 32)")
    ap.add_argument("--stride", type=int, default=16, help="passo entre janelas (padrão 16 = 50%% de sobreposição)")
    ap.add_argument("--no-crop-action", action="store_true",
                    help="não roda o SlowFast no recorte da pessoa (extensão opcional)")
    args = ap.parse_args()

    out = Path(args.out) if args.out else Path("outputs") / Path(args.video).stem
    process_video(args.video, out, Models(), max_frames=args.max_frames, window=args.window,
                  stride=args.stride, crop_action=not args.no_crop_action)


if __name__ == "__main__":
    main()
