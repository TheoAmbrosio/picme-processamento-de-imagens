"""Amostra das seis corrupções reais da fase com barras."""
from pathlib import Path
import sys
from PIL import Image,ImageOps,ImageDraw,ImageFont
import torch

RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ/'residuals networks'))
from res_theo34_barras.ampliacao_barras import DatasetManifesto,VARIANTES
from res_theo34_barras.config_res_theo34 import CONFIG


def gerar():
    fonte=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    painel=Image.new('RGB',(1744,600),'white')
    desenho=ImageDraw.Draw(painel)
    media=torch.tensor([.485,.456,.406])[:,None,None]
    desvio=torch.tensor([.229,.224,.225])[:,None,None]
    for linha,classe in enumerate(('Cat','Dog')):
        p=RAIZ/'dataset/PetImages'/classe/'0.jpg'
        ds=DatasetManifesto([(p,linha,v) for v in VARIANTES],config=CONFIG)
        with Image.open(p) as im:
            original=ImageOps.exif_transpose(im).convert('RGB').resize((224,224))
        imagens=[original]
        for i in range(6):
            x,_=ds[i]
            pixels=((x*desvio+media).clamp(0,1)*255).round().byte()
            imagens.append(Image.fromarray(pixels.permute(1,2,0).numpy()))
        titulos=('Original (referência)','Barras 1','Barras 2','Barras 3','Barras 4',
                 'Sorteio anterior 1','Sorteio anterior 2')
        for col,(titulo,im) in enumerate(zip(titulos,imagens)):
            x,y=16+col*246,18+linha*285
            desenho.text((x,y),titulo,font=fonte,fill='black')
            painel.paste(im,(x,y+30))
    destino=RAIZ/'resultados_residuals_networks/res_theo34_barras/amostras_barras.png'
    painel.save(destino)
    print(destino)


if __name__=='__main__':gerar()
