"""Volumes em escala única: canais × altura × profundidade espacial."""
from pathlib import Path
from math import sqrt
import random
from PIL import Image, ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D

DESTINO=Path(__file__).resolve().parent
VOLUMES=(
    ('RGB',1,3,224,224,'#a77548'),
    ('Conv 7×7, s2',1,64,112,112,'#ae7357'),
    ('MaxPool 3×3, s2',1,64,56,56,'#c48768'),
    ('layer1: 3 BasicBlocks',3,64,56,56,'#cd8c70'),
    ('layer2: 4 BasicBlocks',4,128,28,28,'#ce8d72'),
    ('layer3: 6 BasicBlocks',6,256,14,14,'#ce8d72'),
    ('AvgPool + Flatten',1,256,1,1,'#a7d6d0'),
    ('Dropout 10%',1,256,1,1,'#a7d6d0'),
    ('FC: 1 logit',1,1,1,1,'#a7d6d0'),
)


def gerar():
    plt.rcParams.update({'font.family':'DejaVu Serif','svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(23,7))
    ax.axis('off')
    ax.set_aspect('equal',adjustable='box')
    x=0
    for indice,(nome,n,c,h,p,cor) in enumerate(VOLUMES):
        largura=n*c
        altura_real=h
        if h==p==1:
            h=p=3
            if c==1: largura=3
        abreviado=indice==5
        gap=60
        if abreviado:
            largura=3*c+gap
        dx=dy=p/sqrt(2)
        ax.add_patch(Polygon([(x,0),(x+largura,0),(x+largura,h),(x,h)],
                             facecolor=cor,edgecolor='black',lw=.65,zorder=3))
        ax.add_patch(Polygon([(x+largura,0),(x+largura+dx,dy),
                              (x+largura+dx,h+dy),(x+largura,h)],
                             facecolor=cor,edgecolor='black',lw=.65,zorder=2))
        ax.add_patch(Polygon([(x,h),(x+dx,h+dy),
                              (x+largura+dx,h+dy),(x+largura,h)],
                             facecolor=cor,edgecolor='black',lw=.65,zorder=4))
        if indice==0:
            candidatos=list((DESTINO.parents[1]/'dataset/PetImages/Dog').glob('*.jpg'))
            random.Random(42).shuffle(candidatos)
            for caminho in candidatos:
                try:
                    with Image.open(caminho) as foto:
                        foto=ImageOps.fit(ImageOps.exif_transpose(foto).convert('RGB'),(224,224))
                        ax.imshow(foto,extent=(0,1,0,1),origin='upper',
                                  transform=Affine2D.from_values(dx,dy,0,h,x+largura,0)+ax.transData,
                                  zorder=4)
                    break
                except OSError:
                    continue
        for j in range(1,n if not abreviado else 3):
            px=x+j*c
            if abreviado and j==2: px+=gap
            ax.plot([px,px],[0,h],color='black',lw=.65,zorder=5)
            ax.plot([px,px+dx],[h,h+dy],color='black',lw=.55,zorder=5)
        if abreviado:
            inicio_gap=x+2*c
            ax.add_patch(Polygon([(inicio_gap,-2),(inicio_gap+gap,-2),
                                  (inicio_gap+gap+dx,h+dy+2),(inicio_gap,h+dy+2)],
                                 facecolor='white',edgecolor='none',zorder=6))
            ax.text(inicio_gap+gap/2,h/2,'...',ha='center',va='center',fontsize=21,zorder=7)
            ax.text(x+largura/2,h+dy+140,'3 blocos centrais omitidos; total: 6',ha='center',fontsize=10)
        if n>1:
            for j in range(3 if abreviado else n):
                inicio=x+j*c+(gap if abreviado and j==2 else 0)
                fim=inicio+c
                ybase=h+dy+7
                yatalho=ybase+35
                projecao=indice in (4,5) and j==0
                cor_atalho='#356b8b' if projecao else '#444444'
                ax.plot([inicio+4,inicio+4,fim-9],
                        [ybase,yatalho,yatalho],lw=.85,color=cor_atalho,
                        ls='--' if projecao else '-',zorder=8)
                ax.annotate('',xy=(fim-9,ybase+7),xytext=(fim-9,yatalho),
                            arrowprops=dict(arrowstyle='->',lw=.85,color=cor_atalho),zorder=8)
                ax.text(fim-9,ybase,'+',fontsize=10,ha='center',va='center',
                        color=cor_atalho,zorder=9,
                        bbox=dict(boxstyle='circle,pad=.08',facecolor='white',edgecolor=cor_atalho,lw=.6))
                ax.text((inicio+fim)/2,yatalho+7,'P' if projecao else 'I',
                        fontsize=8,ha='center',color=cor_atalho)
        h_real=altura_real
        if n>1:
            dimensao=f'{c} × {h_real} × {h_real} por bloco'
        else:
            dimensao=f'{c} × {h_real} × {h_real}'
        # Rótulos não alteram a escala geométrica; uma linha guia identifica
        # os volumes estreitos cujos textos precisam ficar acima da figura.
        centro=x+largura/2
        if indice in (1,2):
            ytexto=280 if indice==1 else 205
            ax.annotate(dimensao,(centro,h+dy),xytext=(centro+20,ytexto),
                        fontsize=10,rotation=30,ha='left',
                        arrowprops=dict(arrowstyle='-',lw=.5))
        else:
            ax.text(centro,h+dy+(95 if n>1 else 22),dimensao,ha='center' if n>1 or h_real==1 else 'left',
                    rotation=0 if n>1 else 30,fontsize=11)
        ax.text(centro,-20,nome,rotation=35,ha='right',va='top',fontsize=11)
        if indice<len(VOLUMES)-1:
            ax.annotate('',xy=(x+largura+dx+4,0),xytext=(x+largura+dx+1,0),
                        arrowprops=dict(arrowstyle='->',lw=.6))
        x+=largura+dx+5
    ax.annotate('Sigmoid → P(Dog)',(x-5,0),xytext=(x-55,240),
                ha='right',fontsize=11,arrowprops=dict(arrowstyle='-',lw=.65))
    ax.set(xlim=(-60,x+70),ylim=(-290,610))
    ax.text(0,570,'ResTheo34 Aug8 — blocos e conexões residuais',fontsize=17,fontweight='bold')
    ax.text(x/2,-215,'I = identidade; P = projeção Conv 1×1 + BN, stride 2. Cada atalho termina na soma (+) do respectivo bloco, seguida de ReLU.',ha='center',fontsize=10)
    ax.text(x/2,-260,'Comprimento = canais; altura/profundidade = pixels. Volumes 1×1 ampliados somente ×3 para visualização. Layer3 abreviada.',ha='center',fontsize=10)
    fig.subplots_adjust(left=.025,right=.985,top=.97,bottom=.04)
    for nome in ('arquitetura_res_theo34_aug8_livro','arquitetura_res_theo34_aug8_volumes'):
        for ext in ('svg','png','pdf'):
            fig.savefig(DESTINO/f'{nome}.{ext}',dpi=160,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    gerar()
