"""Diagramas vetoriais dos modelos; não importa nem executa treinamento."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

DESTINO=Path(__file__).resolve().parent
CORES=['#e9a55c','#75aed9','#85bca6']


def desenhar(ax,profundidade,blocos,batch):
    ax.set_xlim(0,10);ax.set_ylim(29.4,0);ax.axis('off')
    cx=4.15
    def texto(x,y,s,tamanho=10,**kw):
        ax.text(x,y,s,fontsize=tamanho,ha=kw.pop('ha','center'),va='center',**kw)
    def caixa(y,titulo,sub='',cor='#e5eaf0',altura=.61):
        ax.add_patch(FancyBboxPatch((2.3,y),3.7,altura,boxstyle='round,pad=0.025,rounding_size=0.055',facecolor=cor,edgecolor='#344253',linewidth=.8))
        texto(cx,y+altura*(.32 if sub else .5),titulo,9.2,fontweight='bold')
        if sub:texto(cx,y+altura*.74,sub,7.6)
    def seta(y0,y1,x=cx):
        ax.annotate('',xy=(x,y1),xytext=(x,y0),arrowprops=dict(arrowstyle='-|>',color='#344253',lw=.8,shrinkA=0,shrinkB=0))
    texto(4.9,.35,f'res_theo{profundidade}',20,fontweight='bold',color='#24374b')
    texto(4.9,.76,f'ResNet-{profundidade} até layer3 • grupos {blocos}',10,color='#516176')
    caixa(1.1,'RGB 224 × 224','Normalização ImageNet', '#f1f3f5');seta(1.71,1.94)
    caixa(1.94,'Conv 7×7, 64 · stride 2','BatchNorm → ReLU → MaxPool 3×3, stride 2 · D','#c9bfdf',.76)
    texto(4.15,2.87,'64 × 56 × 56',8.5,color='#516176')
    y=3.14
    seta(2.70,y)
    for stage,n in enumerate(blocos):
        canais=[64,128,256][stage];lado=[56,28,14][stage]
        top=y
        ax.add_patch(FancyBboxPatch((.35,top-.08),8.8,n*1.57+.40,boxstyle='round,pad=0.03,rounding_size=0.08',facecolor=CORES[stage],alpha=.12,edgecolor=CORES[stage],linewidth=1.1))
        texto(.56,top+.12,f'layer{stage+1}',8.7,ha='left',fontweight='bold',color='#344253')
        for b in range(n):
            entrada=y+.24;stride=2 if stage>0 and b==0 else 1
            pro=stage>0 and b==0
            texto(1.68,y+.68,f'Bloco {b+1}',7.6,color='#516176')
            seta(y,entrada)
            caixa(entrada,f'Conv 3×3, {canais} · stride {stride}','BN → ReLU · D',CORES[stage],.46)
            seta(entrada+.46,entrada+.59)
            caixa(entrada+.59,f'Conv 3×3, {canais} · stride 1','BN · D',CORES[stage],.46)
            soma=entrada+1.21
            seta(entrada+1.05,soma-.10)
            ax.add_patch(Circle((cx,soma),.10,facecolor='white',edgecolor='#344253',lw=.85))
            texto(cx,soma,'+',9)
            # Identidade ou projeção externa: a soma não recebe dropout.
            sx=7.6
            ax.plot([cx,sx,sx,cx+.12],[entrada-.10,entrada-.10,soma,soma],color='#39709c' if pro else '#627080',lw=.9,ls='--' if pro else '-')
            ax.annotate('',xy=(cx+.10,soma),xytext=(cx+.48,soma),arrowprops=dict(arrowstyle='-|>',color='#39709c' if pro else '#627080',lw=.9))
            texto(sx+.13,entrada+.48,'Conv 1×1\ns2 → BN → D' if pro else 'Identidade',7,ha='left',color='#39709c' if pro else '#627080')
            texto(cx,soma+.21,'ReLU → D',7.8)
            y+=1.57
            if b<n-1:seta(soma+.29,y+.24)
        texto(.58,y+.11,f'{canais} × {lado} × {lado}',8.4,ha='left',fontweight='bold',color='#344253')
        y+=.47
        seta(soma+.29,y)
    caixa(y,'Média global → Flatten','256 características · D após ambas','#d8dde4');y+=.61
    seta(y,y+.21);y+=.21
    caixa(y,'fc: Linear 256 → 1','Pesos e bias novos: Normal(0, 1) → D','#e8b6c4',.72);y+=.72
    seta(y,y+.2);y+=.2
    caixa(y,'Logit [B, 1]','Treino: BCEWithLogitsLoss','#f1f3f5');y+=.61
    seta(y,y+.2);y+=.2
    caixa(y,'Avaliação: sigmoid','P(cachorro) · classe com limiar 0,5','#d9e6ed');y+=.88
    texto(cx,y,'layer4 removida • backbone pré-treinado ImageNet',8.4,fontweight='bold');y+=.34
    texto(cx,y,f'Configuração atual: batch {batch} · Adam · LR 0,001',8.3);y+=.31
    texto(cx,y,'25 épocas · só fc nas épocas 1–10 · libera na 11',8.3)
    return y


def gerar():
    plt.rcParams['font.family']='DejaVu Sans'
    fig,axes=plt.subplots(1,2,figsize=(15,24))
    for ax,prof,blocos,batch in zip(axes,[18,34],[[2,2,2],[3,4,6]],[128,64]):
        desenhar(ax,prof,blocos,batch)
    fig.subplots_adjust(left=.015,right=.985,top=.975,bottom=.065,wspace=.04)
    fig.text(.5,.045,'D = nn.Dropout(p=0,3) após cada operação primitiva selecionada: Conv, BN, ReLU, pooling, Flatten e fc.',ha='center',fontsize=10)
    fig.text(.5,.032,'Máscaras aleatórias durante treino; dropout desativado na avaliação. Projeções downsample também recebem D.',ha='center',fontsize=9)
    fig.text(.5,.019,'As operações de soma (+) não recebem dropout. Dimensões: canais × altura × largura. Blocos oficiais preservados.',ha='center',fontsize=9)
    fig.savefig(DESTINO/'arquiteturas_res_theo18_34.png',dpi=160,facecolor='white')
    fig.savefig(DESTINO/'arquiteturas_res_theo18_34.svg',facecolor='white')
    fig.savefig(DESTINO/'arquiteturas_res_theo18_34.pdf',facecolor='white')
    plt.close(fig)
    for prof,blocos,batch in [(18,[2,2,2],128),(34,[3,4,6],64)]:
        fig,ax=plt.subplots(figsize=(8,24))
        fim=desenhar(ax,prof,blocos,batch)
        ax.set_ylim(fim+.7,0)
        fig.set_size_inches(8,(fim+1)*.81)
        fig.subplots_adjust(left=.02,right=.98,top=.99,bottom=.03)
        fig.text(.5,.012,'D: dropout 30% após cada operação primitiva; desligado em avaliação.',ha='center',fontsize=9)
        for ext in ('svg','png'):
            fig.savefig(DESTINO/f'arquitetura_res_theo{prof}.{ext}',dpi=160,facecolor='white')
        plt.close(fig)


if __name__=='__main__':
    gerar()
