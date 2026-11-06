"""Exibe as oito transformações reais do fine-tuning, sem executar treinamento."""
from pathlib import Path
import sys
from PIL import Image, ImageDraw, ImageFont
import torch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ / 'residuals networks'))
from res_theo34_aug8.ampliacao8 import DatasetManifesto, VARIANTES
from res_theo34_aug8.config_res_theo34 import CONFIG


def gerar():
    fonte = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    nomes = ('Original 224×224','Ruído + 2% branco','8 quadrados centrais',
             'Horizontal 2× + crop','Vertical 2× + crop','Ruído + quadrados',
             'Horizontal + ruído + quadrados','Vertical + ruído + quadrados')
    media = torch.tensor([.485,.456,.406])[:,None,None]
    desvio = torch.tensor([.229,.224,.225])[:,None,None]
    painel=Image.new('RGB',(1160,1100),'white')
    desenho=ImageDraw.Draw(painel)
    for animal,classe in enumerate(('Cat','Dog')):
        p=RAIZ/'dataset/PetImages'/classe/'0.jpg'
        ds=DatasetManifesto([(p,animal,v) for v in VARIANTES],config=CONFIG)
        for i,nome in enumerate(nomes):
            x=20+(i%4)*285
            y=18+(animal*2+i//4)*270
            desenho.text((x,y),nome,fill='black',font=fonte)
            tensor,_=ds[i]
            pixels=((tensor*desvio+media).clamp(0,1)*255).round().byte()
            painel.paste(Image.fromarray(pixels.permute(1,2,0).numpy()),(x,y+28))
    destino=RAIZ/'resultados_residuals_networks/res_theo34_aug8/amostras_aug8.png'
    painel.save(destino)
    print(destino)


if __name__=='__main__':
    gerar()
