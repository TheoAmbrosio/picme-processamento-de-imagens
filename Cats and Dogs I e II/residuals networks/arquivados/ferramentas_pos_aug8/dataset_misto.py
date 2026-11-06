"""Ampliações mistas do participante, sem modificar datasets ou modelos anteriores."""
from dataclasses import dataclass, asdict
import math
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import functional as F, InterpolationMode


VARIANTES = (
    'original', 'esticada_aleatoria', 'cinza', 'ruido', 'pixels_brancos',
    'ruido_sal', 'retangulos_pretos', 'quadrados_pretos_livres',
    'quadrados_pretos_centrais', 'quadrados_coloridos', 'ruido_quadrados',
    'barras', 'faixa_horizontal', 'blur_original', 'blur_ruido', 'blur_quadrados',
)
BASES_FAIXA = ('original', 'ruido_sal', 'quadrados_pretos_centrais', 'ruido_quadrados')
MEDIA = (.485, .456, .406)
DESVIO = (.229, .224, .225)


@dataclass
class ConfigMisto:
    variantes: tuple[str, ...] = VARIANTES
    quadrados_antes: int = 8
    quadrados_aumento: float = .25
    variancia_antes: float = .08
    variancia_aumento: float = .25
    blur_kernel: int = 12
    pixels_brancos_fracao: float = .02
    lado_min: int = 20
    lado_max: int = 40
    retangulos_quantidade: int = 4
    regiao_central: int = 150
    barras_min: int = 1
    barras_max: int = 4
    barra_largura_min: int = 20
    barra_largura_max: int = 50
    faixa_min: int = 0
    faixa_max: int = 50
    esticamento_fator: int = 2

    @property
    def quadrados_quantidade(self):
        return math.ceil(self.quadrados_antes * (1 + self.quadrados_aumento))

    @property
    def ruido_variancia(self):
        return self.variancia_antes * (1 + self.variancia_aumento)

    def validar(self):
        if (not self.variantes or len(set(self.variantes)) != len(self.variantes)
                or not set(self.variantes).issubset(VARIANTES)):
            raise ValueError('Informe variantes conhecidas e distintas.')
        if self.ruido_variancia < 0 or not 1 <= self.lado_min <= self.lado_max <= 224:
            raise ValueError('Variância ou tamanho das oclusões inválido.')
        if not 0 <= self.pixels_brancos_fracao <= 1 or not 1 <= self.blur_kernel <= 224:
            raise ValueError('Pixels brancos ou kernel inválido.')
        if not 1 <= self.regiao_central <= 224:
            raise ValueError('Região central inválida.')

    def parametros_efetivos(self):
        return dict(asdict(self), quadrados_quantidade=self.quadrados_quantidade,
                    ruido_variancia=self.ruido_variancia)


CONFIG = ConfigMisto()


def carregar_divisao(manifesto, config=CONFIG, raiz_originais=None):
    """Reutiliza os splits existentes e os JPEGs de avaliação sem redividir dados."""
    from res_theo34_aug8.dados_res_theo34 import carregar_divisao as ler
    config.validar()
    comuns = dict(aceitar_rotulos_duplicados=True, raiz_originais=raiz_originais)
    originais = ler(Path(manifesto), ('treino', 'validacao'),
                    usar_dados_ampliados=False, **comuns)
    avaliacao = ler(Path(manifesto), ('treino', 'validacao'),
                    usar_dados_ampliados=True, **comuns)
    return dict(treino=[(p,r,v) for p,r in originais['treino'] for v in config.variantes],
                validacao=[(p,r,'avaliacao') for p,r in avaliacao['validacao']],
                teste=[(p,r,'avaliacao') for p,r in avaliacao['teste']])


def _inteiro(a,b):
    return int(torch.randint(a,b+1,()).item())


def aplicar_ruido(x, config):
    return (x + torch.randn_like(x) * math.sqrt(config.ruido_variancia)).clamp(0,1)


def aplicar_brancos(x, config):
    indices = torch.randperm(224*224)[:round(config.pixels_brancos_fracao*224*224)]
    x[:,indices//224,indices%224] = 1
    return x


def aplicar_quadrados(x, config, centrais=False, coloridos=False, retangulos=False):
    quantidade = config.retangulos_quantidade if retangulos else config.quadrados_quantidade
    for _ in range(quantidade):
        altura = _inteiro(config.lado_min,config.lado_max)
        largura = _inteiro(config.lado_min,config.lado_max) if retangulos else altura
        if centrais:
            margem = (224-config.regiao_central)//2
            y = _inteiro(margem,margem+config.regiao_central-1)-altura//2
            z = _inteiro(margem,margem+config.regiao_central-1)-largura//2
            y,z = max(0,min(y,224-altura)),max(0,min(z,224-largura))
        else:
            y,z = _inteiro(0,224-altura),_inteiro(0,224-largura)
        x[:,y:y+altura,z:z+largura] = torch.rand(3,1,1) if coloridos else 0
    return x


def aplicar_blur(x, config):
    k=config.blur_kernel
    a=(k-1)//2
    b=k-1-a
    return torch.nn.functional.avg_pool2d(
        torch.nn.functional.pad(x[None],(a,b,a,b),mode='reflect'),k,stride=1)[0]


def transformar(imagem, variante, config=CONFIG):
    im=F.resize(imagem.convert('RGB'),[224,224],InterpolationMode.BILINEAR,antialias=True)
    if variante=='esticada_aleatoria':
        tamanho=([224,224*config.esticamento_fator] if _inteiro(0,1)==0
                 else [224*config.esticamento_fator,224])
        im=F.center_crop(F.resize(im,tamanho,InterpolationMode.BILINEAR,antialias=True),[224,224])
    if variante=='cinza':
        im=ImageOps.grayscale(im).convert('RGB')
    if variante=='faixa_horizontal':
        base=BASES_FAIXA[_inteiro(0,len(BASES_FAIXA)-1)]
        x=transformar(im,base,config)
        largura=_inteiro(config.faixa_min,config.faixa_max)
        if largura:
            y=_inteiro(0,224-largura)
            x[:,y:y+largura,:]=0
        return x
    x=F.to_tensor(im)
    if variante in ('ruido','ruido_sal','ruido_quadrados','blur_ruido'):
        x=aplicar_ruido(x,config)
    if variante in ('pixels_brancos','ruido_sal','ruido_quadrados','blur_ruido'):
        x=aplicar_brancos(x,config)
    if variante=='retangulos_pretos':
        x=aplicar_quadrados(x,config,retangulos=True)
    elif variante=='quadrados_pretos_livres':
        x=aplicar_quadrados(x,config)
    elif variante=='quadrados_coloridos':
        x=aplicar_quadrados(x,config,coloridos=True)
    elif variante in ('quadrados_pretos_centrais','ruido_quadrados','blur_quadrados'):
        x=aplicar_quadrados(x,config,centrais=True)
    elif variante=='barras':
        from res_theo34_barras.ampliacao_barras import aplicar_barras
        x=aplicar_barras(x,config)
    if variante in ('blur_original','blur_ruido','blur_quadrados'):
        x=aplicar_blur(x,config)
    return x


class DatasetMisto(Dataset):
    """Retorna tensor RGB normalizado e rótulo binário [1] para o modelo Aug8."""
    class_to_idx={'Cat':0,'Dog':1}

    def __init__(self,amostras,config=CONFIG):
        config.validar()
        self.amostras=amostras
        self.config=config

    def __len__(self):
        return len(self.amostras)

    def __getitem__(self,indice):
        p,r,v=self.amostras[indice]
        with Image.open(p) as im:
            im=ImageOps.exif_transpose(im).convert('RGB')
            x=F.to_tensor(im) if v=='avaliacao' else transformar(im,v,self.config)
        return F.normalize(x,MEDIA,DESVIO),torch.tensor([r],dtype=torch.float32)
