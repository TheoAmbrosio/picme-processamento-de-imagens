"""Quatro corrupções com barras e duas ampliações antigas por original."""
import math
import torch
from PIL import Image,ImageOps
from torchvision.transforms import functional as F, InterpolationMode
from .ampliacao8 import DatasetManifesto as Anterior, ler_manifesto, NORMALIZAR, ROTULOS

VARIANTES=('barras0','barras1','barras2','barras3','anterior0','anterior1')
ELEGIVEIS=('ruido','quadrados','ruido_quadrados')


def carregar_divisao(manifesto,splits_obrigatorios,**kwargs):
    kwargs.pop('usar_dados_ampliados',None)
    originais=ler_manifesto(manifesto,splits_obrigatorios,usar_dados_ampliados=False,**kwargs)
    avaliacao=ler_manifesto(manifesto,splits_obrigatorios,usar_dados_ampliados=True,**kwargs)
    return dict(treino=[(p,r,v) for p,r in originais['treino'] for v in VARIANTES],
                validacao=[(p,r,'avaliacao') for p,r in avaliacao['validacao']],
                teste=[(p,r,'avaliacao') for p,r in avaliacao['teste']])


def aplicar_barras(x,config):
    yy,xx=torch.meshgrid(torch.arange(224),torch.arange(224),indexing='ij')
    for _ in range(int(torch.randint(config.barras_min,config.barras_max+1,()).item())):
        largura=int(torch.randint(config.barra_largura_min,config.barra_largura_max+1,()).item())
        cy,cx=torch.randint(37,187,(2,)).tolist()
        direcao=int(torch.randint(0,3,()).item())
        if direcao==0: distancia=(yy-cy).float()
        elif direcao==1: distancia=(xx-cx).float()
        else:
            sinal=1 if torch.rand(())<.5 else -1
            distancia=((yy-cy)+sinal*(xx-cx))/math.sqrt(2)
        x[:,distancia.abs()<=largura/2]=0
    return x


class DatasetManifesto(Anterior):
    def __getitem__(self,indice):
        p,r,v=self.amostras[indice]
        if v.startswith('anterior'):
            variante=ELEGIVEIS[int(torch.randint(0,len(ELEGIVEIS),()).item())]
            return Anterior([(p,r,variante)],config=self.config)[0]
        if v=='avaliacao': return super().__getitem__(indice)
        with Image.open(p) as im:
            im=F.resize(ImageOps.exif_transpose(im).convert('RGB'),[224,224],
                        InterpolationMode.BILINEAR,antialias=True)
            x=aplicar_barras(F.to_tensor(im),self.config)
        return F.normalize(x,[.485,.456,.406],[.229,.224,.225]),torch.tensor([r],dtype=torch.float32)
