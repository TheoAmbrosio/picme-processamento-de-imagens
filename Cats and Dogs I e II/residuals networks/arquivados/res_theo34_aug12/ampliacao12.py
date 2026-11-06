"""Aug8 + faixa horizontal aleatória + três versões com blur médio 10x10."""
import torch
from torch.nn import functional as F
from .ampliacao8 import DatasetManifesto as Aug8, ler_manifesto, ROTULOS, VARIANTES as BASE

VARIANTES=(*BASE,'faixa','blur_original','blur_ruido','blur_quadrados')


def carregar_divisao(manifesto,splits_obrigatorios,**kwargs):
    kwargs.pop('usar_dados_ampliados',None)
    originais=ler_manifesto(manifesto,splits_obrigatorios,usar_dados_ampliados=False,**kwargs)
    avaliacao=ler_manifesto(manifesto,splits_obrigatorios,usar_dados_ampliados=True,**kwargs)
    return dict(treino=[(p,r,v) for p,r in originais['treino'] for v in VARIANTES],
                validacao=[(p,r,'avaliacao') for p,r in avaliacao['validacao']],
                teste=[(p,r,'avaliacao') for p,r in avaliacao['teste']])


def blur_media(x,kernel=10):
    antes=(kernel-1)//2
    depois=kernel-1-antes
    return F.avg_pool2d(F.pad(x.unsqueeze(0),(antes,depois,antes,depois),mode='reflect'),
                        kernel,stride=1).squeeze(0)


class DatasetManifesto(Aug8):
    def __getitem__(self,indice):
        p,r,v=self.amostras[indice]
        base=v
        if v=='faixa':base=BASE[int(torch.randint(0,len(BASE),()).item())]
        elif v.startswith('blur_'):
            base={'blur_original':'original','blur_ruido':'ruido','blur_quadrados':'quadrados'}[v]
        x,_=Aug8([(p,r,base)],config=self.config)[0]
        media=torch.tensor([.485,.456,.406])[:,None,None]
        desvio=torch.tensor([.229,.224,.225])[:,None,None]
        if v=='faixa':
            largura=int(torch.randint(self.config.faixa_min,self.config.faixa_max+1,()).item())
            if largura:
                y=int(torch.randint(0,225-largura,()).item())
                x[:,y:y+largura,:]=-media/desvio
        elif v.startswith('blur_'):
            pixels=(x*desvio+media).clamp(0,1)
            x=(blur_media(pixels,self.config.blur_kernel)-media)/desvio
        return x,torch.tensor(r,dtype=torch.long)
