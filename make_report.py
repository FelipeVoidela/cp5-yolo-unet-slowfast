"""Gera relatorio_cp5.pdf (máx. 3 páginas) a partir de outputs/*/summary.json.

As colunas numéricas vêm dos arquivos de saída; as avaliações qualitativas ("a caixa acompanha?",
"a máscara acompanha o contorno?") foram feitas olhando os quadros em outputs/*/moments/.
"""
import csv
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path("outputs")
S = {v: json.loads((OUT / v / "summary.json").read_text(encoding="utf-8"))
     for v in ("archery", "arm_wrestling", "transition", "pedestrians")}

# ---- integrantes e parecer individual de cada um -----------------------------------------------
INTEGRANTES = ["Açussena Macedo Mautone - RM 552568", "Felipe Heilmann Marques - RM 551026",
               "Felipe Voidela Toledo - RM 98595", "Carlos Eduardo Caramante Ribeiro - RM 552159",
               "Ian Cancian Nachtergaele - RM 98387"]
PARECERES = {
    "Açussena Macedo Mautone": (
        "U-Net e projeção da máscara",
        "O que mais me chamou atenção foi o peso do pré-processamento. Sem mexer no modelo, só trocar o "
        "recorte esticado por um quadrado levou o preenchimento de 0,26 para 0,43, e espelhar a borda levou "
        "para 0,47. A U²-Net ainda falha com pessoas pequenas ou cortadas pelo quadro (IDs 7 e 16 ficam sem "
        "máscara), o que faz sentido para um modelo treinado com pessoas inteiras e em primeiro plano."),
    "Felipe Heilmann Marques": (
        "YOLO e leitura do vídeo",
        "No vídeo do arco o YOLO foi impecável: 300/300 quadros e só 2 IDs. Na queda de braço ficou claro que o "
        "problema não é detectar, e sim rastrear. O ByteTrack olha só posição e movimento, por isso gerou 31 IDs "
        "e herdou o ID da arqueira no corte de cena. Como próximo passo, eu testaria um rastreador com "
        "re-identificação por aparência e um YOLO maior, para pegar o oponente de camisa preta."),
    "Felipe Voidela Toledo": (
        "integração do pipeline e visualização",
        "A parte mais difícil foi juntar escalas de tempo diferentes: YOLO e U-Net respondem por quadro, o "
        "SlowFast responde por janela de 32 quadros. Resolvemos mostrando em cada quadro a janela de centro "
        "mais próximo, e isso explica o atraso de ~0,5 s na troca de ação. O custo também pesa: na CPU são "
        "~1,6 s por quadro, quase tudo na U-Net. Sem GPU, não dá para usar em tempo real."),
    "Carlos Eduardo Caramante Ribeiro": (
        "SlowFast e classificação de ações",
        "O SlowFast acertou todas as janelas de arco e de queda de braço. Mas a confiança de 1,00 em quadros "
        "borrados e até corrompidos mostra que esse número não é uma probabilidade em que dá para confiar. Nos "
        "pedestres ele ficou perdido porque 'andar' não existe no Kinetics-400. Aprendi que a lista de classes "
        "do pré-treino importa tanto quanto a arquitetura."),
    "Ian Cancian Nachtergaele": (
        "experimentos, testes e apresentação",
        "O clipe de transição foi a forma mais direta de responder se a classe muda quando a ação termina: a "
        "troca acontece, com queda de confiança para 0,94 só na janela que mistura as duas cenas. Os testes "
        "automatizados se pagaram: foi um deles que revelou a máscara cortada na cintura pela borda "
        "replicada. Sem eles, esse erro teria passado despercebido."),
}

LABEL = {"archery": "V1 arco", "arm_wrestling": "V2 queda de braço", "transition": "Transição",
         "pedestrians": "Pedestres"}
GT = {"archery": "archery", "arm_wrestling": "arm wrestling", "pedestrians": "andar (não existe no K-400)"}

# avaliação visual de cada momento: (caixa acompanha?, máscara acompanha contorno?, observação)
QUAL = {
    ("archery", 60): ("Sim (2/2)", "Sim", "corpo e chapéu inteiros"),
    ("archery", 150): ("Sim (2/2)", "Sim", "corpo inteiro, com pernas"),
    ("archery", 240): ("Sim (2/2)", "Quase", "perde calça vermelha do ID 1"),
    ("arm_wrestling", 56): ("Parcial", "Não", "garoto: só o braço; oponente vaza p/ mesa"),
    ("arm_wrestling", 140): ("Parcial", "Parcial", "garoto inteiro; oponente não detectado"),
    ("arm_wrestling", 224): ("Não", "Não", "borrão: 1 caixa p/ 2 pessoas"),
    ("transition", 46): ("Sim (2/2)", "Sim", "trecho de arco"),
    ("transition", 115): ("Sim (2/2)", "Sim", "4 quadros antes do corte"),
    ("transition", 184): ("Parcial", "Parcial", "ID 2 herdado; IDs 7 e 16 sem máscara"),
    ("pedestrians", 30): ("Parcial", "Sim", "2 pessoas numa caixa; 1 oculta"),
    ("pedestrians", 75): ("Parcial", "Sim", "pessoa atrás da placa perdida"),
    ("pedestrians", 120): ("Parcial", "Sim", "pessoa na borda não detectada"),
}

ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica", fontSize=8.4, leading=10.4,
                      alignment=TA_JUSTIFY, spaceAfter=3)
h1 = ParagraphStyle("h1", parent=body, fontName="Helvetica-Bold", fontSize=10, leading=12, spaceBefore=5,
                    spaceAfter=3, textColor=colors.HexColor("#1f3a8a"))
title = ParagraphStyle("t", parent=body, fontName="Helvetica-Bold", fontSize=13, leading=16, alignment=1)
small = ParagraphStyle("s", parent=body, fontSize=7.2, leading=8.6, alignment=0, spaceAfter=0)
cap = ParagraphStyle("c", parent=small, fontName="Helvetica-Oblique", alignment=1, spaceAfter=4)


def P(t, st=body):
    return Paragraph(t, st)


def table(rows, widths, header_bg="#1f3a8a"):
    hdr = ParagraphStyle("hdr", parent=small, textColor=colors.white, fontName="Helvetica-Bold")
    t = Table([[P(str(c), hdr if i == 0 else small) for c in r] for i, r in enumerate(rows)],
              colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#9ca3af")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
    ]))
    return t


def br(x):
    return f"{x:.2f}".replace(".", ",")


def pct(x):
    return f"{x * 100:.0f}%"


def window_accuracy(v):
    s = S[v]
    if v == "transition":
        cut = 119  # 1º quadro do vídeo 2 dentro do clipe de transição
        gt = lambda w: "archery" if (w["frames"][0] + w["frames"][1] - 1) / 2 < cut else "arm wrestling"
    elif v in ("archery", "arm_wrestling"):
        gt = lambda w: GT[v]
    else:
        return "n/a", "n/a"
    ws = s["windows"]
    f = sum(w["full"][0][0] == gt(w) for w in ws) / len(ws)
    c = sum(w["crop"][0][0] == gt(w) for w in ws if w["crop"]) / len(ws)
    return pct(f), pct(c)


def build(path="relatorio_cp5.pdf"):
    W = A4[0] - 3 * cm
    I1, I2 = (json.loads((OUT / v / "individual" / "individual_summary.json").read_text(encoding="utf-8"))
              for v in ("archery", "arm_wrestling"))
    st = []
    st += [P("Applied Computer Vision 2026 - CheckPoint 5<br/>Detecção de Ações de Pessoas com YOLO + U-Net + SlowFast", title),
           P("Prof. Dr. Paulo Sergio Rodrigues - Outubro de 2026", ParagraphStyle("p", parent=small, alignment=1)),
           Spacer(1, 3),
           P("<br/>".join(INTEGRANTES), ParagraphStyle("i", parent=small, alignment=1)),
           Spacer(1, 4)]

    st += [P("1. Pipeline", h1), P(
        "Todos os modelos são pré-treinados, sem treino adicional. <b>YOLO11n</b> (COCO, classe <i>person</i>) "
        "detecta cada pessoa e o rastreador <b>ByteTrack</b> mantém um ID por pessoa. Para cada caixa, a "
        "<b>U-Net</b> (U²-Net <i>human_seg</i>, uma U-Net aninhada treinada para segmentar pessoas, escolhida por "
        "manter a arquitetura codificador-decodificador com <i>skip connections</i> da U-Net e ter pesos públicos "
        "para pessoas) recebe um "
        "<b>recorte quadrado</b> centrado na caixa (margem de 25%), redimensionado para 320×320. O mapa de "
        "probabilidade é limiarizado em 0,5 e <b>projetado de volta</b> nas coordenadas do quadro original, "
        "limitado à caixa. O <b>SlowFast R50 8×8</b> (Kinetics-400, 400 ações) classifica janelas de 32 quadros "
        "com passo 16 (via rápida com 32 quadros e via lenta com 8). Cada janela é registrada com top-3 e "
        "confiança em <i>actions.csv</i> e roda duas vezes: no quadro inteiro e no recorte quadrado da união "
        "das caixas da janela (extensão opcional). No vídeo final, cada quadro mostra caixa, ID, confiança, "
        "máscara colorida com contorno e a ação da janela de centro mais próximo. Os três modelos também foram "
        "rodados isoladamente (<i>run_individual.py</i>) antes da integração. Tudo rodou em CPU: cerca de "
        "1,6 s por quadro com 2 pessoas (a U-Net domina o custo) e cerca de 2 s por inferência do SlowFast.")]

    st += [P("2. Experimentos", h1), P(
        "<b>V1</b>: tiro com arco (10 s, 30 fps, 320×240, 2 arqueiras, câmera estável). <b>V2</b>: queda de "
        "braço (10 s, 28 fps, 340×256, close-up, câmera tremida, pessoas ao fundo). Também rodamos dois testes "
        "extras. O primeiro é um clipe de <b>transição</b>, com os últimos 4 s de V1 seguidos dos primeiros 4 s "
        "de V2 (corte no quadro 119, t = 3,97 s), para medir o que acontece quando a ação termina e outra começa. "
        "O segundo é um caso de falha, <b>pedestres</b> (OpenCV <i>vtest</i>, 10 fps, pessoas pequenas, ação "
        "<i>andar</i>, que não existe no Kinetics-400). Os momentos analisados são 20%, 50% e 80% de cada vídeo. "
        f"<b>Modelos isolados</b> (passo 1, <i>outputs/*/individual</i>): o YOLO sozinho encontra {br(I1['yolo_mean_people'])} "
        f"e {br(I2['yolo_mean_people'])} pessoas/quadro em V1 e V2. A U-Net aplicada ao quadro inteiro, sem caixa, marca "
        f"{I1['unet_mean_mask_area_fraction']:.0%} e {I2['unet_mean_mask_area_fraction']:.0%} dos pixels como pessoa, mas "
        "gera uma única máscara para todas as pessoas, por isso o recorte guiado pelo YOLO é necessário para ter uma "
        f"máscara por pessoa. O SlowFast sozinho prevê <i>{I1['slowfast_windows'][0]['top3'][0][0]}</i> e "
        f"<i>{I2['slowfast_windows'][0]['top3'][0][0]}</i>, as mesmas classes do pipeline integrado.")]

    rows = [["Vídeo", "Quadros com pessoa", "Pessoas/quadro", "IDs únicos", "Conf. YOLO", "Preench. máscara*",
             "Janelas", "Acerto SlowFast (inteiro / recorte)", "Conf. média top-1 (inteiro / recorte)"]]
    for v in S:
        s = S[v]
        ws = s["windows"]
        af, ac = window_accuracy(v)
        cf = sum(w["full"][0][1] for w in ws) / len(ws)
        cc = sum(w["crop"][0][1] for w in ws if w["crop"]) / len(ws)
        rows.append([LABEL[v], f"{s['frames_with_person']}/{s['frames']}", br(s["mean_people_per_frame"]),
                     s["unique_track_ids"], br(s["mean_det_conf"]), br(s["mean_mask_fill_ratio"]),
                     s["n_windows"], f"{af} / {ac}", f"{br(cf)} / {br(cc)}"])
    st += [table(rows, [W * x for x in (.13, .10, .09, .07, .08, .10, .08, .18, .17)]),
           P("Tabela 1 - Resumo por vídeo. *Preench. máscara = pixels da máscara ÷ área da caixa (uma pessoa "
             "em pé ocupa tipicamente 40-60% da caixa).", cap)]

    rows = [["Vídeo", "t (s)", "Caixa segue?", "Máscara segue?", "Preench.", "Ação inteiro (conf.)",
             "Ação recorte (conf.)", "Bate?", "Observação"]]
    for v in S:
        for m in S[v]["moments"]:
            box_ok, mask_ok, obs = QUAL[(v, m["frame"])]
            fills = [d["mask_fill"] for d in m["detections"]]
            af, ac = m["action_full"][0], m["action_crop"][0]
            if v == "transition":
                truth = "archery" if m["frame"] < 119 else "arm wrestling"
            else:
                truth = GT[v]
            ok = "Sim" if af[0] == truth else "Não"
            rows.append([LABEL[v], f"{m['time_s']:.1f}".replace(".", ","), box_ok, mask_ok,
                         br(sum(fills) / max(1, len(fills))), f"{af[0]} ({br(af[1])})",
                         f"{ac[0]} ({br(ac[1])})", ok, obs])
    st += [Spacer(1, 3), table(rows, [W * x for x in (.12, .06, .08, .08, .07, .17, .17, .06, .19)]),
           P("Tabela 2 - Três momentos por vídeo (20%, 50% e 80% da duração). Quadros em outputs/&lt;vídeo&gt;/moments.", cap)]

    shots = [OUT / "archery/moments/frame_0150.png", OUT / "arm_wrestling/moments/frame_0056.png",
             OUT / "transition/moments/frame_0184.png", OUT / "pedestrians/moments/frame_0030.png"]
    iw = W / 4 - 2
    imgs = [Image(str(p), width=iw, height=iw * 554 / 640) for p in shots]
    st += [KeepTogether([Table([imgs], colWidths=[W / 4] * 4, style=[("LEFTPADDING", (0, 0), (-1, -1), 1),
                                                                     ("RIGHTPADDING", (0, 0), (-1, -1), 1)]),
                         P("Figura 1 - (a) V1 t=5 s: caixas e máscaras corretas; (b) V2 t=2 s: máscara do garoto só no braço e a do oponente "
                           "de preto vaza para mesa e fundo; (c) transição t=6,1 s: ID 2 herdado da arqueira e "
                           "IDs 7 e 16 sem máscara; (d) pedestres t=3 s: duas pessoas numa única caixa, 'playing cricket' 13%.", cap)]),
           KeepTogether([Image(str(OUT / "transition/actions_plot.png"), width=W * 0.78, height=W * 0.78 * 2.8 / 8),
                         P("Figura 2 - Confiança top-1 por janela no clipe de transição (corte em t = 3,97 s).", cap)])]

    fill_v1, fill_v2, fill_ped = (S[v]["mean_mask_fill_ratio"] for v in ("archery", "arm_wrestling", "pedestrians"))
    det_v1 = list(csv.DictReader(open(OUT / "archery" / "detections.csv", encoding="utf-8")))
    n_v1, low_v1 = len(det_v1), sum(float(r["mask_fill_ratio"]) < 0.3 for r in det_v1)
    st += [P("3. Respostas às perguntas", h1),
           P("<b>A caixa acompanha a pessoa?</b> Sim, em V1: 300/300 quadros com as 2 pessoas e só 2 IDs no "
             "vídeo inteiro. Em V2, só parcialmente: com close-up, câmera tremida e oclusões, o ByteTrack gerou "
             "<b>31 IDs</b> para poucas pessoas reais, e 35 quadros (12,5%) ficaram sem nenhuma detecção. "
             "Nos pedestres a caixa segue bem quem está visível, mas falha com oclusão (pessoa atrás da placa) "
             "e junta pessoas lado a lado."),
           P("<b>A máscara acompanha o contorno?</b> Sim, para pessoas de corpo inteiro e visíveis (V1, "
             f"pedestres: preenchimento de {br(fill_v1)} a {br(fill_ped)}, contorno seguindo braços, chapéu e pernas). Em V2 a máscara "
             "falha quando a pessoa aparece só em parte (braço, ombro) ou se confunde com o fundo escuro. Quando "
             "a pessoa está pequena e ocluída ao fundo, a máscara sai praticamente vazia (preenchimento ~0)."),
           P("<b>A ação prevista corresponde ao trecho?</b> Sim em V1 (<i>archery</i>, 18/18 janelas) e em V2 "
             "(<i>arm wrestling</i>, 17/17), as duas classes que existem no Kinetics-400. Nos pedestres, não: "
             "<i>andar</i> não está entre as 400 classes, e o modelo oscila entre <i>playing cricket</i>, "
             "<i>throwing ball</i> e <i>jogging</i> com confiança de 0,13 a 0,65."),
           P("<b>A confiança cai ou a classe muda quando a ação começa ou termina?</b> Sim. No clipe de "
             "transição, a janela 6 (quadros 96-127, só 9 do vídeo novo) continua <i>archery</i> 1,00, mas já com "
             "<i>arm wrestling</i> como 2ª classe. A janela 7 (quadros 112-143, que contém o corte) muda para "
             "<i>arm wrestling</i> com confiança de <b>0,94</b> (0,85 no recorte), e a janela 8 volta a 1,00. A "
             "troca de classe atrasa cerca de meia janela (~0,5 s). Fora das transições a confiança fica saturada "
             "em ~1,00, e a queda só aparece na janela que mistura as duas ações.")]

    st += [P("4. Erros explicados", h1),
           P("<b>E1 - Recorte esticado degradava a máscara (corrigido).</b> Na 1ª versão, a caixa do YOLO "
             "(alta e estreita, ~80×200 px) era redimensionada direto para 320×320. A pessoa ficava ~2,5× mais "
             "larga e a U-Net, treinada com proporções naturais, segmentava só pedaços (calça, mochila). O "
             "preenchimento médio em V1 era de <b>0,26</b>. Trocando por um recorte <b>quadrado</b> com contexto "
             "(margem de 25%), o preenchimento subiu para <b>0,43</b> e a máscara passou a cobrir cabeça e tronco. "
             "Os testes automatizados acharam um 2º problema: quando o quadrado sai do vídeo (pessoa na borda), "
             "completar com a borda <i>replicada</i> cria listras que a U-Net confunde com fundo, e a máscara "
             f"parava na cintura. Com borda <i>espelhada</i>, V1 foi para <b>{br(fill_v1)}</b> e V2 de 0,38 para "
             f"<b>{br(fill_v2)}</b>. Ainda restam {low_v1} de {n_v1} máscaras de V1 com preenchimento &lt; 0,3 (ex.: "
             "ID 2 nos primeiros quadros, só o tronco)."),
           P("<b>E2 - Troca e herança de ID no corte de cena.</b> No quadro 119 da transição, o ByteTrack "
             "associou a caixa do garoto de V2 ao ID 2 da arqueira, porque as caixas se sobrepõem (IoU alto) e "
             "o rastreador só usa posição e movimento, não aparência. Em V2 o mesmo mecanismo, junto com "
             "câmera tremida e borrão, gera 31 IDs. Um rastreador com re-identificação por aparência (ex.: "
             "BoT-SORT com ReID) ou detectar cortes de cena e reiniciar o rastreador resolveria isso."),
           P("<b>E3 - Ação sem pessoa e excesso de confiança do SlowFast.</b> Em V2, nos quadros 187-208 o borrão "
             "deixa só antebraços visíveis e o YOLO não detecta ninguém. Nos quadros 252-267 o próprio arquivo "
             "tem blocos corrompidos de compressão. Mesmo assim o SlowFast responde <i>arm wrestling</i> com "
             "<b>1,00</b>: ele usa o contexto da janela inteira (32 quadros) e não verifica a pessoa. As saídas "
             "do modelo pré-treinado são extremamente confiantes (logit top-1 ~79 contra ~27 da 2ª classe), "
             "então 1,00 aqui não significa certeza calibrada."),
           P("<b>E4 - Classe inexistente e pessoas pequenas (pedestres).</b> Sem <i>walking</i> no Kinetics-400, o "
             "SlowFast é forçado a escolher a classe mais parecida e varia a cada janela. O vídeo em 10 fps "
             "também faz 32 quadros cobrirem 3,2 s em vez de ~1 s, um movimento diferente do que o modelo viu "
             "no treino. Além disso, com 5-6 pessoas espalhadas, a união das caixas cobre quase o quadro inteiro "
             "e o 'recorte da pessoa' quase não isola nada.")]

    st += [P("5. Extensão: SlowFast no quadro inteiro × no recorte da pessoa", h1),
           P("Em V1, V2 e na transição, as duas entradas concordam em 100% das janelas, porque a ação ocupa quase "
             "todo o quadro e o contexto (alvos, mesa) é redundante. A diferença aparece na janela de "
             "transição (0,94 no quadro inteiro e 0,85 no recorte): o recorte tem menos contexto para "
             "desempatar. Nos pedestres o recorte <b>aumenta</b> a confiança média (0,29 → 0,40) e concorda com o "
             "quadro inteiro em só 67% das janelas: remover a grama e o prédio reduz distrações, mas ainda não há "
             "classe correta. Conclusão: o recorte ajuda quando a pessoa é pequena no quadro e tende a prejudicar "
             "quando o contexto é parte da ação.")]

    st += [P("6. Parecer dos integrantes", h1)] + [
        P(f"<b>{nome}</b> ({papel}): {texto}") for nome, (papel, texto) in PARECERES.items()]

    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                            topMargin=1.2 * cm, bottomMargin=1.2 * cm, title="CP5 - YOLO + U-Net + SlowFast")
    doc.build(st)
    print(f"relatório: {path}")


if __name__ == "__main__":
    build()
