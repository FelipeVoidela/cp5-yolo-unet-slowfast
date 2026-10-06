# CP5 – Detecção de Ações de Pessoas com YOLO + U-Net + SlowFast

Lê um vídeo curto e mostra, em cada trecho, **onde está a pessoa** (caixa – YOLO), **quais pixels
pertencem a ela** (máscara – U-Net) e **qual ação está fazendo** (SlowFast). Só usa modelos já
treinados, sem nenhum treinamento adicional.

## Integrantes

| Nome | RM |
|---|---|
| Açussena Macedo Mautone | 552568 |
| Felipe Heilmann Marques | 551026 |
| Felipe Voidela Toledo | 98595 |
| Carlos Eduardo Caramante Ribeiro | 552159 |
| Ian Cancian Nachtergaele | 98387 |

## Entregáveis

| Item | Arquivo |
|---|---|
| Código executável | `main.py`, `cp5/`, `run_experiments.py`, `run_individual.py` |
| Vídeo de demonstração (caixa + máscara + ação sobrepostas) | [`outputs/demo.mp4`](outputs/demo.mp4) |
| Relatório (PDF, até 3 páginas) | [`relatorio_cp5.pdf`](relatorio_cp5.pdf) |
| Extensão opcional: SlowFast no quadro inteiro × no recorte | seção 5 do relatório e `outputs/*/actions.csv` (`mode`) |
| Testes automatizados | `tests/test_cp5.py` |

| Etapa | Modelo | Pesos |
|---|---|---|
| Detecção + rastreamento | YOLO11n (Ultralytics) + ByteTrack | COCO, classe `person` |
| Segmentação | U²-Net `human_seg` (U-Net aninhada) via ONNX Runtime | Supervisely Person (rembg) |
| Ação | SlowFast R50 8×8 (PyTorchVideo, `torch.hub`) | Kinetics-400 |

## Instalação

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python download_models.py
```

## Como rodar

```bash
# um vídeo qualquer -> outputs/<nome>/
python main.py videos/archery.mp4

# passo 1 do roteiro: cada modelo sozinho
python run_individual.py videos/archery.mp4 videos/arm_wrestling.mp4

# todos os experimentos do relatório + outputs/demo.mp4
python run_experiments.py

# gera o relatório em PDF (relatorio_cp5.pdf)
python make_report.py

# testes automatizados (~5 min na CPU)
pip install pytest pypdf
python -m pytest -v tests
```

> No Windows, crie o ambiente virtual num caminho curto (ex.: `C:\venvs\cp5`). Caminhos muito longos
> estouram o limite de 260 caracteres na instalação do `onnxruntime`.

Opções do `main.py`: `--window 32` (quadros por janela do SlowFast), `--stride 16` (passo entre
janelas), `--max-frames N`, `--no-crop-action` (desliga a comparação quadro inteiro × recorte).

## Como funciona

1. **YOLO** detecta e rastreia cada pessoa (ID fixo por pessoa).
2. **U-Net**: para cada caixa, recorta um **quadrado** centrado nela (margem de 25%, para não
   deformar a pessoa ao redimensionar para 320×320), gera o mapa de probabilidade, limiariza em 0,5
   e **projeta a máscara de volta** no quadro original, limitada à caixa.
3. **SlowFast** classifica janelas consecutivas de 32 quadros (passo 16): via rápida com 32
   quadros e via lenta com 8. Registra top-3 e confiança de cada janela. A janela é rodada duas
   vezes: no **quadro inteiro** e no **recorte da pessoa** (extensão opcional).
4. A visualização sobrepõe caixa, máscara e ação; cada quadro mostra a janela de centro mais próximo.

## Saídas (`outputs/<vídeo>/`)

- `annotated.mp4` – vídeo com as três saídas sobrepostas
- `detections.csv` – por quadro/pessoa: caixa, confiança, área e preenchimento da máscara
- `actions.csv` – por janela: top-3 e confiança (quadro inteiro e recorte)
- `actions_plot.png` – confiança top-1 ao longo do tempo
- `moments/` – três momentos de cada vídeo usados no relatório
- `summary.json` – resumo numérico
- `individual/` – os três modelos rodados separadamente
- `outputs/demo.mp4` – vídeo de demonstração com todos os experimentos
