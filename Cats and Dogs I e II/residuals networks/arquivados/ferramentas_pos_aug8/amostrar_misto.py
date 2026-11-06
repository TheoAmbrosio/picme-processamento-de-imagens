"""Prancha das transformações reais do treino misto, sem tocar no processo."""
from pathlib import Path
import sys
from PIL import Image, ImageDraw, ImageFont
import torch

RAIZ = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RAIZ), str(RAIZ / 'residuals networks')]
from dataset_misto import DatasetMisto, VARIANTES


def gerar():
    torch.set_num_threads(1)
    torch.manual_seed(20261105)
    nomes = ('Original', 'Esticada H/V aleatória', 'Preto e branco',
             'Gaussiano: variância 0,10', '2% pixels brancos', 'Gaussiano + branco',
             '4 retângulos pretos', '10 quadrados livres',
             '10 quadrados centrais', '10 quadrados coloridos',
             'Ruído + quadrados', 'Barras H/V/diagonais', 'Faixa horizontal',
             'Blur original 12×12', 'Blur ruído 12×12', 'Blur quadrados 12×12')
    fonte = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    painel = Image.new('RGB', (1016, 2150), 'white')
    desenho = ImageDraw.Draw(painel)
    media = torch.tensor([.485,.456,.406])[:,None,None]
    desvio = torch.tensor([.229,.224,.225])[:,None,None]
    for animal, classe in enumerate(('Cat','Dog')):
        desenho.text((15, animal*1070+5), classe+' — mistura atual (sorteio ilustrativo)', fill='black', font=fonte)
        ds = DatasetMisto([(RAIZ/'dataset/PetImages'/classe/'0.jpg', animal, v) for v in VARIANTES])
        for i, nome in enumerate(nomes):
            x = 15 + (i%4)*252
            y = animal*1070 + 35 + (i//4)*258
            desenho.text((x,y), nome, fill='black', font=fonte)
            tensor,_ = ds[i]
            pixels = ((tensor*desvio+media).clamp(0,1)*255).round().byte()
            painel.paste(Image.fromarray(pixels.permute(1,2,0).numpy()),(x,y+24))
    destino = RAIZ/'resultados_residuals_networks/res_theo34_misto/amostras_misto.png'
    painel.save(destino)
    print(destino)


if __name__ == '__main__':
    gerar()
