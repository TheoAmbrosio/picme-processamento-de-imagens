"""Oito variantes virtuais por original; avaliação histórica preservada."""
import math
from pathlib import Path
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
from .dados_res_theo34 import carregar_divisao as ler_manifesto, NORMALIZAR, ROTULOS

VARIANTES = ('original','ruido','quadrados','horizontal','vertical',
             'ruido_quadrados','horizontal_ruido_quadrados','vertical_ruido_quadrados')


def carregar_divisao(manifesto, splits_obrigatorios, **kwargs):
    kwargs.pop('usar_dados_ampliados',None)
    originais = ler_manifesto(manifesto,splits_obrigatorios,
                             usar_dados_ampliados=False,**kwargs)
    avaliacao = ler_manifesto(manifesto,splits_obrigatorios,
                             usar_dados_ampliados=True,**kwargs)
    return {'treino': [(p,r,v) for p,r in originais['treino'] for v in VARIANTES],
            'validacao':[(p,r,'avaliacao') for p,r in avaliacao['validacao']],
            'teste':[(p,r,'avaliacao') for p,r in avaliacao['teste']]}


class DatasetManifesto(Dataset):
    def __init__(self, amostras, usar_dados_ampliados=True, config=None):
        self.amostras=amostras
        self.config=config
        self.class_to_idx=ROTULOS.copy()

    def __len__(self):
        return len(self.amostras)

    def __getitem__(self, indice):
        p,r,v=self.amostras[indice]
        with Image.open(p) as im:
            im=ImageOps.exif_transpose(im).convert('RGB')
            if v=='avaliacao':
                return NORMALIZAR(im),torch.tensor([r],dtype=torch.float32)
            im=F.resize(im,[224,224],InterpolationMode.BILINEAR,antialias=True)
            if 'horizontal' in v:
                im=F.center_crop(F.resize(im,[224,448],InterpolationMode.BILINEAR,antialias=True),[224,224])
            elif 'vertical' in v:
                im=F.center_crop(F.resize(im,[448,224],InterpolationMode.BILINEAR,antialias=True),[224,224])
            x=F.to_tensor(im)
        if 'ruido' in v:
            x=(x+torch.randn_like(x)*math.sqrt(self.config.ruido_variancia)).clamp_(0,1)
            n=round(self.config.pixels_brancos_fracao*224*224)
            indices=torch.randperm(224*224)[:n]
            x[:,indices//224,indices%224]=1
        if 'quadrados' in v:
            for _ in range(self.config.quadrados_quantidade):
                lado=int(torch.randint(self.config.quadrado_min,self.config.quadrado_max+1,()).item())
                # Centros no intervalo [37,187), região central 150×150.
                centros=torch.randint(37,187,(2,))
                y,z=[int(c)-lado//2 for c in centros]
                x[:,y:y+lado,z:z+lado]=0
        return F.normalize(x,[.485,.456,.406],[.229,.224,.225]),torch.tensor([r],dtype=torch.float32)
