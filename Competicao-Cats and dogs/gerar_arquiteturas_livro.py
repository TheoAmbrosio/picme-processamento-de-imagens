"""Diagramas de volumes no padrão do SVG do Aug8; sem executar modelos."""
from pathlib import Path
from math import sqrt
import shutil
import xml.etree.ElementTree as ET

from PIL import Image, ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D

RAIZ = Path(__file__).resolve().parent
PROJETO_THEO = RAIZ.parent / 'Cats and Dogs I e II'

# nome, canais, altura, largura, cabeça em escala esquemática
MODELOS = {
    'Dulce': dict(titulo='Dulce — CNN224 v2', fonte='cats_and_dogs_224_v2.ipynb, célula27',
        saida='Softmax → P(Cat), P(Dog)',
        volumes=[('RGB',3,224,224,False),
          ('Conv5×5 + ReLU + Pool2 + Dropout20%',32,111,111,False),
          ('Conv5×5 + ReLU + Pool2 + Dropout20%',64,54,54,False),
          ('Conv5×5 + ReLU + Pool2 + Dropout20%',64,26,26,False),
          ('Conv5×5 + ReLU + Pool2 + Dropout20%',128,12,12,False),
          ('Conv5×5 + ReLU + Pool2 + Dropout20%',128,5,5,False),
          ('Flatten3200',3200,1,1,True),('FC1024 + ReLU',1024,1,1,True),
          ('FC32 + ReLU',32,1,1,True),('FC2 logits',2,1,1,True)]),
    'Lisandra': dict(titulo='Lisandra — AlexNetBinary com BatchNorm',fonte='Notebook GitHub, célula5, publicação538a5e9',
        saida='1 logit; sigmoid externo → P(Dog)',
        volumes=[('RGB + Normalize ImageNet',3,224,224,False),
          ('Conv11×11 s4 p2 + BN + ReLU + Pool3 s2',96,27,27,False),
          ('Conv5×5 p2 + BN + ReLU + Pool3 s2',256,13,13,False),
          ('Conv3×3 p1 + BN + ReLU',384,13,13,False),
          ('Conv3×3 p1 + BN + ReLU',384,13,13,False),
          ('Conv3×3 p1 + BN + ReLU + Pool3 s2',256,6,6,False),
          ('Flatten9216',9216,1,1,True),('FC4096 + ReLU + Dropout50%',4096,1,1,True),
          ('FC4096 + ReLU + Dropout50%',4096,1,1,True),
          ('FC1 logit',1,1,1,True)]),
    'Bandeira': dict(titulo='Bandeira — CNN V4',fonte='src/model.py, classe CNN (V4)',
        saida='1 logit; sigmoid na avaliação',
        volumes=[('RGB',3,384,384,False),
          ('Conv3×3 p1 + BN + ReLU + Pool2',32,192,192,False),
          ('Conv3×3 p1 + BN + ReLU + Pool2',64,96,96,False),
          ('Conv3×3 p1 + BN + ReLU + Pool2',128,48,48,False),
          ('Conv3×3 p1 + BN + ReLU + Pool2',256,24,24,False),
          ('Conv3×3 p1 + BN + ReLU + Pool2',512,12,12,False),
          ('Global AvgPool + Flatten',512,1,1,True),
          ('Dropout50%',512,1,1,True),('FC1 logit',1,1,1,True)])}


def desenhar(participante, config):
    plt.rcParams.update({'font.family':'DejaVu Serif','svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(24,8))
    ax.axis('off');ax.set_aspect('equal',adjustable='box')
    x=0;maxh=max(v[2] for v in config['volumes']);topo=maxh*(1+1/sqrt(2))
    for i,(nome,c,h,p,densa) in enumerate(config['volumes']):
        # Mesmo eixo por canal/pixel para os mapas de características.
        # Apenas vetores densos são esquemáticos, explicitado na legenda.
        largura=(max(85,min(c,140)) if densa else c)
        altura=4 if densa else h
        profundidade=4 if densa else p
        dx=dy=profundidade/sqrt(2)
        cor='#a7d6d0' if densa else '#a77548' if i==0 else '#cd8c70'
        ax.add_patch(Polygon([(x,0),(x+largura,0),(x+largura,altura),(x,altura)],facecolor=cor,edgecolor='black',lw=.65,zorder=3))
        ax.add_patch(Polygon([(x+largura,0),(x+largura+dx,dy),(x+largura+dx,altura+dy),(x+largura,altura)],facecolor=cor,edgecolor='black',lw=.65,zorder=2))
        ax.add_patch(Polygon([(x,altura),(x+dx,altura+dy),(x+largura+dx,altura+dy),(x+largura,altura)],facecolor=cor,edgecolor='black',lw=.65,zorder=4))
        if i==0:
            foto=PROJETO_THEO/'dataset/PetImages/Dog/0.jpg'
            if foto.exists():
                with Image.open(foto) as im:
                    imagem=ImageOps.fit(ImageOps.exif_transpose(im).convert('RGB'),(224,224))
                    ax.imshow(imagem,extent=(0,1,0,1),origin='upper',
                        transform=Affine2D.from_values(dx,dy,0,altura,x+largura,0)+ax.transData,zorder=4)
        centro=x+largura/2
        dimensao=f'{c} × {h} × {p}' if not densa else f'{c} componentes'
        if densa:
            ax.annotate(dimensao,(centro,altura+dy),xytext=(centro,65+(i%2)*60),
                fontsize=10,rotation=30,ha='left',arrowprops=dict(arrowstyle='-',lw=.5))
        else:
            ax.text(centro,altura+dy+22,dimensao,ha='left',rotation=30,fontsize=11)
        rotulo=nome.replace(' + ReLU','').replace('Dropout','DO')
        ax.text(centro,-24,rotulo,rotation=35,ha='right',va='top',fontsize=10)
        fim=x+largura+dx
        if i<len(config['volumes'])-1:
            ax.annotate('',xy=(fim+6,0),xytext=(fim+1,0),arrowprops=dict(arrowstyle='->',lw=.6))
        x=fim+(18 if densa else 7)
    ax.annotate(config['saida'],(x-22,0),xytext=(x+100,topo*.6),
        ha='right',fontsize=11,arrowprops=dict(arrowstyle='-',lw=.65))
    ax.text(0,topo+180,config['titulo']+' — arquitetura em volumes',fontsize=18,fontweight='bold')
    ax.text(0,topo+135,'Dimensões C × H × W nas saídas de cada bloco; blocos convolucionais em escala comum.',fontsize=11)
    ax.text(x/2,-340,'Comprimento = canais; altura e profundidade = pixels. Cabeça densa em escala esquemática, com dimensões indicadas.',ha='center',fontsize=10)
    ax.text(x/2,-385,'ReLU após convoluções e FCs intermediárias; DO = dropout. Sem resíduos. Foto RGB ilustrativa.',ha='center',fontsize=10)
    ax.text(x/2,-425,'Fonte: '+config['fonte']+'. Nenhum treinamento executado para gerar o desenho.',ha='center',fontsize=10)
    ax.set(xlim=(-70,x+160),ylim=(-460,topo+230))
    fig.subplots_adjust(left=.025,right=.985,top=.97,bottom=.04)
    destino=RAIZ/participante/'modelo principal'
    for ext in ('svg','png','pdf'):
        fig.savefig(destino/f'arquitetura_livro.{ext}',dpi=160,facecolor='white',bbox_inches='tight',pad_inches=.2)
    plt.close(fig)
    ET.parse(destino/'arquitetura_livro.svg')


def main():
    for nome,config in MODELOS.items():desenhar(nome,config)
    # Preserva exatamente o desenho já aprovado do Theo, inclusive resíduos.
    for ext in ('svg','png','pdf'):
        shutil.copy2(PROJETO_THEO/'docs/diagramas'/f'arquitetura_res_theo34_aug8_livro.{ext}',
            RAIZ/'Theo/modelo principal'/f'arquitetura_livro.{ext}')
    print('SVG, PNG e PDF no padrão livro: Theo, Lisandra, Dulce e Bandeira.')


if __name__=='__main__':main()
