"""Prancha de amostras reais do pipeline Theo Ajuste, sem executar treino."""
from pathlib import Path
import sys

from PIL import Image, ImageDraw
import torch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / 'residuals networks'))
from theo_ajuste.config_res_theo_especial import CONFIG
from theo_ajuste.dados_res_theo_especial import DatasetManifesto


def gerar():
    arquivos = [RAIZ / 'dataset/PetImages' / classe / '0.jpg' for classe in ('Cat', 'Dog')]
    variantes = ('original', 'cinza', 'ruido_sal', 'quadrados_coloridos', 'cutout')
    titulos = ('Original', 'Preto e branco', 'Gaussiano 0,08 + 2% branco',
               '8 quadrados coloridos', '8 quadrados pretos')
    painel = Image.new('RGB', (1240, 575), 'white')
    desenho = ImageDraw.Draw(painel)
    desenho.text((16, 10), 'Theo Ajuste — transformacoes reais; novos sorteios de quadrados a cada leitura', fill='black')
    media = torch.tensor([.485,.456,.406])[:,None,None]
    desvio = torch.tensor([.229,.224,.225])[:,None,None]
    for linha, arquivo in enumerate(arquivos):
        dataset = DatasetManifesto([(arquivo, linha, v) for v in variantes],
            esquema_ampliacao=CONFIG.esquema_ampliacao, semente=CONFIG.semente,
            ruido_gaussiano_variancia=CONFIG.ruido_gaussiano_variancia,
            pixels_brancos_fracao=CONFIG.pixels_brancos_fracao,
            cutout_retangulos=CONFIG.cutout_retangulos,
            cutout_lado_minimo=CONFIG.cutout_lado_minimo,
            cutout_lado_maximo=CONFIG.cutout_lado_maximo)
        for coluna, titulo in enumerate(titulos):
            x, y = 16 + coluna * 245, 40 + linha * 265
            desenho.text((x, y), titulo, fill='black')
            tensor, _ = dataset[coluna]
            pixels = ((tensor * desvio + media).clamp(0,1) * 255).round().byte()
            painel.paste(Image.fromarray(pixels.permute(1,2,0).numpy()), (x,y+24))
    destino = RAIZ / 'docs/amostras_theo_ajuste.png'
    painel.save(destino)
    print(destino)


if __name__ == '__main__':
    gerar()
