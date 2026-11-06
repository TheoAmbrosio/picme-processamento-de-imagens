"""Gera o diagrama da ResTheo Especial principal no padrão dos ResTheo 18/34."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

DESTINO = Path(__file__).resolve().parent
CORES = ["#e9a55c", "#75aed9", "#85bca6"]


def desenhar(ax, nome="res_theo_especial", canais_estagios=(32, 64, 128), rodape=None,
             blocos=None, dropout_interno=True):
    blocos = blocos or (2,) * len(canais_estagios)
    ax.set_xlim(0, 10)
    ax.set_ylim(21, 0)
    ax.axis("off")
    cx = 4.15

    def texto(x, y, conteudo, tamanho=10, **kwargs):
        ax.text(x, y, conteudo, fontsize=tamanho,
                ha=kwargs.pop("ha", "center"), va="center", **kwargs)

    def caixa(y, titulo, subtitulo="", cor="#e5eaf0", altura=.61):
        ax.add_patch(FancyBboxPatch(
            (2.3, y), 3.7, altura,
            boxstyle="round,pad=0.025,rounding_size=0.055",
            facecolor=cor, edgecolor="#344253", linewidth=.8))
        texto(cx, y + altura * (.32 if subtitulo else .5), titulo, 9.2,
              fontweight="bold")
        if subtitulo:
            texto(cx, y + altura * .74, subtitulo, 7.6)

    def seta(y0, y1, x=cx):
        ax.annotate("", xy=(x, y1), xytext=(x, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#344253", lw=.8,
                                    shrinkA=0, shrinkB=0))

    texto(4.9, .35, nome, 20, fontweight="bold", color="#24374b")
    texto(4.9, .76, f"{'ResNet-18 estreita' if dropout_interno else 'ResNet-34'} até layer{len(canais_estagios)} • blocos {list(blocos)}",
          10, color="#516176")
    caixa(1.1, "RGB 224 × 224", "Normalização ImageNet", "#f1f3f5")
    seta(1.71, 1.94)
    caixa(1.94, f"Conv 7×7, {canais_estagios[0]} · stride 2",
          "BatchNorm → D → ReLU → D → MaxPool 3×3 → D" if dropout_interno else "BatchNorm → ReLU → MaxPool 3×3", "#c9bfdf", .76)
    texto(cx, 2.87, f"{canais_estagios[0]} × 56 × 56", 8.5, color="#516176")
    y = 3.14
    seta(2.70, y)

    for estagio, (quantidade, canais, lado, cor) in enumerate(zip(
            blocos, canais_estagios, (56, 28, 14), CORES), start=1):
        topo = y
        ax.add_patch(FancyBboxPatch(
            (.35, topo - .08), 8.8, quantidade * 1.57 + .40,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor=cor, alpha=.12, edgecolor=cor, linewidth=1.1))
        texto(.56, topo + .12, f"layer{estagio}", 8.7, ha="left",
              fontweight="bold", color="#344253")
        for bloco in range(quantidade):
            entrada = y + .24
            projecao = estagio > 1 and bloco == 0
            stride = 2 if projecao else 1
            texto(1.68, y + .68, f"Bloco {bloco + 1}", 7.6, color="#516176")
            seta(y, entrada)
            caixa(entrada, f"Conv 3×3, {canais} · stride {stride}",
                  "D → BN → D → ReLU → D" if dropout_interno else "BN → ReLU", cor, .46)
            seta(entrada + .46, entrada + .59)
            caixa(entrada + .59, f"Conv 3×3, {canais} · stride 1",
                  "D → BN → D" if dropout_interno else "BN", cor, .46)
            soma = entrada + 1.21
            seta(entrada + 1.05, soma - .10)
            ax.add_patch(Circle((cx, soma), .10, facecolor="white",
                                edgecolor="#344253", lw=.85))
            texto(cx, soma, "+", 9)

            sx = 7.6
            cor_residuo = "#39709c" if projecao else "#627080"
            ax.plot([cx, sx, sx, cx + .12],
                    [entrada - .10, entrada - .10, soma, soma],
                    color=cor_residuo, lw=.9, ls="--" if projecao else "-")
            ax.annotate("", xy=(cx + .10, soma), xytext=(cx + .48, soma),
                        arrowprops=dict(arrowstyle="-|>", color=cor_residuo, lw=.9))
            texto(sx + .13, entrada + .48,
                  ("Conv 1×1\ns2 → D → BN → D" if dropout_interno else "Conv 1×1\nstride 2 → BN") if projecao else "Identidade",
                  7, ha="left", color=cor_residuo)
            texto(cx, soma + .21, "ReLU → D" if dropout_interno else "ReLU", 7.8)
            y += 1.57
            if bloco < quantidade - 1:
                seta(soma + .29, y + .24)
        texto(.58, y + .11, f"{canais} × {lado} × {lado}", 8.4,
              ha="left", fontweight="bold", color="#344253")
        y += .47
        seta(soma + .29, y)

    caixa(y, "Média global → Flatten", f"{canais_estagios[-1]} características · " + ("D após ambas" if dropout_interno else "Dropout 10% após Flatten"),
          "#d8dde4")
    y += .61
    seta(y, y + .21)
    y += .21
    caixa(y, f"fc: Linear {canais_estagios[-1]} → 1", "He normal fan_out · sem dropout" if dropout_interno else "Pesos carregados do melhor checkpoint", "#e8b6c4", .72)
    y += .72
    seta(y, y + .20)
    y += .20
    caixa(y, "Logit [B, 1]", "Treino: BCEWithLogitsLoss", "#f1f3f5")
    y += .61
    seta(y, y + .20)
    y += .20
    caixa(y, "Avaliação: sigmoid", "P(cachorro) · limiar 0,5", "#d9e6ed")
    y += .88
    texto(cx, y, ("layer3 e layer4 removidas" if len(canais_estagios) == 2 else "layer4 removida")
          + (" • modelo inicializado do zero" if dropout_interno else " • origem ImageNet; fine-tuning da época 33"), 8.4,
          fontweight="bold")
    y += .34
    for linha in rodape or (
            "AdamW · LR com warmup 1e-6 → 1e-3 · weight decay 0,01",
            "110 épocas · dados originais 1–20 · ampliação 21–110",
            "Dropout: 0→20% (20–50), 20% até 80, 20→5% (80–100)"):
        texto(cx, y, linha, 8.1)
        y += .31
    return y


def gerar():
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(8, 18))
    fim = desenhar(ax)
    ax.set_ylim(fim + .8, 0)
    fig.set_size_inches(8, (fim + 1) * .81)
    fig.subplots_adjust(left=.02, right=.98, top=.99, bottom=.045)
    fig.text(.5, .026,
             "D: dropout após operações selecionadas; desativado na validação. Conv1 e fc ficam sem dropout.",
             ha="center", fontsize=8.6)
    fig.text(.5, .014,
             "Setas laterais mostram os resíduos. As somas (+) não recebem dropout.",
             ha="center", fontsize=8.6)
    for nome in ("arquitetura_res_theo_especial", "arquitetura_res_theo_especial18_antigo"):
        base = DESTINO / nome
        fig.savefig(base.with_suffix(".png"), dpi=180, facecolor="white")
        fig.savefig(base.with_suffix(".svg"), facecolor="white")
        fig.savefig(base.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    gerar()
