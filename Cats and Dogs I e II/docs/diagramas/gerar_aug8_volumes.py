"""Arquitetura Aug8 com volumes em perspectiva, em SVG vetorial."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch

DESTINO=Path(__file__).resolve().parent


def gerar():
    plt.rcParams['svg.fonttype']='none'
    plt.rcParams['font.family']='DejaVu Sans'
    fig,ax=plt.subplots(figsize=(24,9))
    ax.set(xlim=(-.8,25.3),ylim=(-3,8.5))
    ax.axis('off')
    ax.text(11.8,7.8,'ResTheo34 Aug8',ha='center',fontsize=24,fontweight='bold',color='#24374b')
    ax.text(11.8,7.15,'ResNet-34 até layer3 · blocos residuais [3, 4, 6] · layer4 removida',ha='center',fontsize=13,color='#516176')
    volumes=[
        (0,.18,5.3,'#c8d4df','Entrada RGB','3 × 224 × 224'),
        (2.1,.65,3.8,'#ce8e65','Conv 7×7, s2','64 × 112 × 112'),
        (4.5,.8,2.65,'#ddad79','MaxPool 3×3, s2','64 × 56 × 56'),
        (7.3,1.65,2.65,'#e9a55c','layer1 · 3 blocos','64 × 56 × 56'),
        (11.1,2.05,1.85,'#75aed9','layer2 · 4 blocos','128 × 28 × 28'),
        (15.2,2.55,1.2,'#85bca6','layer3 · 6 blocos','256 × 14 × 14'),
        (19.6,.3,.9,'#bac7d7','Média global','256 × 1 × 1'),
        (21.2,.16,1.5,'#c9bfdf','Flatten → Dropout','256 · p = 0,10'),
        (23.1,.18,.45,'#e8b6c4','Linear 256 → 1','1 logit'),
    ]
    baseline=.6
    for i,(x,w,h,cor,titulo,dim) in enumerate(volumes):
        dx,dy=.32,.28
        ax.add_patch(Rectangle((x,baseline),w,h,facecolor=cor,edgecolor='#2e3a49',lw=1.1,zorder=3))
        ax.add_patch(Polygon([(x+w,baseline),(x+w+dx,baseline+dy),
                              (x+w+dx,baseline+h+dy),(x+w,baseline+h)],
                            facecolor=cor,edgecolor='#2e3a49',alpha=.8,zorder=2))
        ax.add_patch(Polygon([(x,baseline+h),(x+dx,baseline+h+dy),
                              (x+w+dx,baseline+h+dy),(x+w,baseline+h)],
                            facecolor=cor,edgecolor='#2e3a49',alpha=.7,zorder=4))
        if i in (3,4,5):
            for j in range(1,(3,4,6)[i-3]):
                px=x+w*j/(3,4,6)[i-3]
                ax.plot([px,px],[baseline,baseline+h],color='#455468',lw=.8,zorder=5)
        ax.text(x+w/2,baseline+h+.65,dim,ha='center',fontsize=11,color='#344253')
        ax.text(x+w/2,-.1,titulo,ha='center',va='top',rotation=24,fontsize=11)
        if i<len(volumes)-1:
            nx=volumes[i+1][0]
            ax.annotate('',xy=(nx-.08,baseline+.07),xytext=(x+w+dx+.05,baseline+.07),
                        arrowprops=dict(arrowstyle='-|>',lw=1.2,color='#344253'))
    ax.text(2.5,5.35,'BN → ReLU',ha='center',fontsize=10,color='#516176')
    for x,w,altura,rotulo in ((7.3,1.65,4.7,'Atalhos internos: identidades por bloco'),
                              (11.1,2.05,4.2,'Atalhos internos: projeção s2 no 1º bloco'),
                              (15.2,2.55,3.6,'Atalhos internos: projeção s2 no 1º bloco')):
        ax.plot([x-.25,x-.25,x+w+.4],[baseline+.1,altura,altura],color='#53748e',lw=1.1)
        ax.annotate('',xy=(x+w+.4,baseline+.7),xytext=(x+w+.4,altura),
                    arrowprops=dict(arrowstyle='-|>',lw=1.1,color='#53748e'))
        ax.text(x+w/2,altura+.15,rotulo,ha='center',fontsize=9,color='#53748e')
        ax.text(x+w+.4,baseline+.55,'+',ha='center',fontsize=15,color='#344253')
    ax.text(23.2,-1.3,'Avaliação: sigmoid\nP(cachorro)',ha='center',fontsize=11,color='#344253')
    ax.text(11.7,-1.75,'Cada bloco residual: Conv 3×3 → BN → ReLU → Conv 3×3 → BN → soma do atalho → ReLU.',ha='center',fontsize=12)
    ax.text(11.7,-2.25,'Primeiro bloco de layer2/layer3 reduz a resolução (stride 2). Dropout somente antes de fc, desligado na avaliação.',ha='center',fontsize=11,color='#516176')
    ax.text(11.7,-2.7,'Volumes e atalhos esquemáticos por estágio, sem escala proporcional · Cat = 0, Dog = 1 · BCEWithLogitsLoss',ha='center',fontsize=10,color='#516176')
    fig.tight_layout()
    for ext in ('svg','png','pdf'):
        fig.savefig(DESTINO/f'arquitetura_res_theo34_aug8_volumes.{ext}',dpi=180,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    gerar()
