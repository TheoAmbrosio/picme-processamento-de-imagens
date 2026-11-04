"""Gera uma prancha das cinco variantes efetivas da ResTheo Especial."""
from pathlib import Path
import sys

import torch
from PIL import Image, ImageDraw

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "residuals networks"))

from res_theo_especial.config_res_theo_especial import CONFIG
from res_theo_especial.dados_res_theo_especial import DatasetManifesto, carregar_divisao


def para_pil(tensor: torch.Tensor) -> Image.Image:
    media = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    desvio = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    pixels = ((tensor.cpu() * desvio + media).clamp(0, 1) * 255).byte()
    return Image.fromarray(pixels.permute(1, 2, 0).numpy(), mode="RGB")


def main() -> None:
    conjuntos = carregar_divisao(
        Path(CONFIG.manifesto), ("treino", "validacao"),
        aceitar_rotulos_duplicados=CONFIG.aceitar_rotulos_duplicados,
        usar_dados_ampliados=True, esquema_ampliacao=CONFIG.esquema_ampliacao)
    grupo = conjuntos["treino"][:5]
    dataset = DatasetManifesto(
        grupo, esquema_ampliacao=CONFIG.esquema_ampliacao, semente=CONFIG.semente,
        ruido_gaussiano_variancia=CONFIG.ruido_gaussiano_variancia,
        pixels_brancos_fracao=CONFIG.pixels_brancos_fracao,
        cutout_retangulos=CONFIG.cutout_retangulos,
        cutout_lado_minimo=CONFIG.cutout_lado_minimo,
        cutout_lado_maximo=CONFIG.cutout_lado_maximo)
    titulos = (
        "1. Original esticada",
        "2. Rotacao fixa",
        "3. Preto e branco",
        "4. Gaussiano + 2% branco",
        "5. Cutout: 8 quadrados",
    )
    margem, topo, intervalo = 20, 54, 16
    largura = margem * 2 + 3 * 224 + 2 * intervalo
    altura = margem * 2 + 2 * (topo + 224) + intervalo
    prancha = Image.new("RGB", (largura, altura), "white")
    desenho = ImageDraw.Draw(prancha)
    for indice, titulo in enumerate(titulos):
        coluna, linha = indice % 3, indice // 3
        x = margem + coluna * (224 + intervalo)
        y = margem + linha * (topo + 224 + intervalo)
        desenho.text((x, y), titulo, fill="black")
        prancha.paste(para_pil(dataset[indice][0]), (x, y + topo))
    destino = RAIZ / "docs/exemplo_ampliacao_res_theo_especial.png"
    prancha.save(destino)
    print(destino)


if __name__ == "__main__":
    main()
