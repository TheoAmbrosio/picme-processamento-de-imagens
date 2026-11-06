"""Arquitetura principal e trajetória factual das execuções que a originaram."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gerar_res_theo_especial18_antigo import desenhar

RAIZ=Path(__file__).resolve().parents[2]
DESTINO=Path(__file__).resolve().parent
ETAPAS=(
 ('Referência histórica: Adam, LR0,001; 5 épocas congeladas;16×; dropout0',
  'res_theo34/complementados,dropout,com_back',0,21),
 ('Repetição: ImageNet; Adam LR0,001;16×; dropout0',
  'res_theo34_repeticao/20261105_105951_490569_bs64_ep50_cong5ep_do0_dadosamp_pre',0,36),
 ('Ramo da33: LRs1e-5/5e-5/1e-4; cosseno; dropout na cabeça7%;16×',
  'res_theo34_finetuning/20261105_152655_704173_bs64_ep30_cong0ep_do7_dadosamp_pre',0,30),
 ('Principal — novo ramo da33: mesmos LRs/cosseno; dropout10%;8×',
  'res_theo34_aug8/20261105_163938_589179_bs64_ep30_cong0ep_do10_dadosamp_pre',0,30))


def arquitetura():
    # A função compartilhada preserva o mesmo padrão visual dos diagramas antigos.
    fig,ax=plt.subplots(figsize=(9,24))
    fim=desenhar(ax,nome='res_theo34_aug8',canais_estagios=(64,128,256),
        blocos=(3,4,6),dropout_interno=False,rodape=(
        'Origem: ResNet-34 ImageNet, checkpoint da época33 (97,34% validação)',
        'AdamW decay0,01; batch64;30 épocas; dropout10% antes de fc',
        'LR base1e-5; layer3 5e-5; fc1e-4; cosseno T_max30, mínimo1e-7',
        'Treino8×: ruído/quadrados centrais/esticamentos2×+crop; validação preservada'))
    ax.set_ylim(fim+.6,0)
    fig.set_size_inches(9,(fim+1)*.75)
    fig.subplots_adjust(left=.02,right=.98,top=.99,bottom=.03)
    fig.text(.5,.012,'Setas laterais: resíduos; nós (+): somas. Dropout somente após Flatten, desligado em avaliação.',ha='center',fontsize=9)
    for ext in ('svg','png','pdf'):
        fig.savefig(DESTINO/f'arquitetura_res_theo34_aug8.{ext}',dpi=160,facecolor='white')
    plt.close(fig)


def trajetoria():
    fig,eixos=plt.subplots(len(ETAPAS),2,figsize=(23,20),layout='constrained')
    resumo=[]
    for linha,(titulo,relativo,inicio,fim) in enumerate(ETAPAS):
        pasta=RAIZ/'resultados_residuals_networks'/relativo
        with (pasta/'historico.csv').open() as f:
            rows=[r for r in csv.DictReader(f) if inicio<int(r['epoca'])<=fim]
        if not rows: continue
        xs=[int(r['epoca']) for r in rows]
        melhor=max(rows,key=lambda r:float(r['acuracia_validacao']))
        resumo.append(dict(execucao=relativo,epocas_observadas=[min(xs),max(xs)],
                           melhor_epoca=int(melhor['epoca']),acuracia_validacao=float(melhor['acuracia_validacao'])))
        for col,(chave,rotulo,fator) in enumerate((('acuracia','Acurácia (%)',100),('bce','Perda BCE',1))):
            ax=eixos[linha,col]
            for split,nome in (('treino','Treino'),('validacao','Validação')):
                ax.plot(xs,[float(r[f'{chave}_{split}'])*fator for r in rows],label=nome,lw=2)
            ax.set(title=titulo,xlabel='Época da respectiva execução',ylabel=rotulo)
            ax.grid(alpha=.25);ax.legend()
            if col==0:
                y=float(melhor['acuracia_validacao'])*100
                ax.scatter([int(melhor['epoca'])],[y],color='black',zorder=5)
                ax.annotate(f'{y:.2f}% · época {melhor["epoca"]}',(int(melhor['epoca']),y),xytext=(8,-20),textcoords='offset points')
    fig.suptitle('Trajetória do ResTheo34 Aug8 — referência, repetição e ramos de fine-tuning\nCada linha é uma etapa independente; não implica continuidade entre todos os pesos.',fontsize=21)
    destino=RAIZ/'resultados_residuals_networks/res_theo34_aug8'
    for ext in ('png','pdf'):
        fig.savefig(destino/f'trajetoria_treinamentos.{ext}',dpi=180,facecolor='white')
    (destino/'resumo_trajetoria.json').write_text(json.dumps(resumo,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    plt.close(fig)


def trajetoria_unificada():
    plt.rcParams['svg.fonttype']='none'
    fig,(ax,perda)=plt.subplots(2,1,figsize=(24,13),sharex=True,
                               gridspec_kw={'height_ratios':[2,1]})
    fig.subplots_adjust(left=.055,right=.98,top=.88,bottom=.29,hspace=.12)
    cores=('#eff3f8','#eef6ee','#fff4e6','#f4effa','#e7f5f4')
    descricoes=(
        '1 · Referência histórica: ImageNet; Adam 0,001; batch 64; 16×; dropout 0%; backbone liberado na época 6.',
        '2 · REINÍCIO ImageNet: repetição nas mesmas condições; melhor checkpoint na época 33 (97,34%).',
        '3 · RETORNO ao checkpoint 33: AdamW novo; LRs 1e-5 / 5e-5 / 1e-4; cosseno; dropout na cabeça 7%; 16×.',
        '4 · NOVO RETORNO ao checkpoint 33: mesmos grupos de LR e cosseno; dropout na cabeça 10%; ampliação 8× com ruído, quadrados e esticamentos.')
    deslocamento=0
    for indice,((titulo,relativo,inicio,fim),cor) in enumerate(zip(ETAPAS,cores),1):
        with (RAIZ/'resultados_residuals_networks'/relativo/'historico.csv').open() as f:
            rows=[r for r in csv.DictReader(f) if inicio<int(r['epoca'])<=fim]
        if not rows:continue
        xs=list(range(deslocamento+1,deslocamento+len(rows)+1))
        for eixo,chave,fator in ((ax,'acuracia',100),(perda,'bce',1)):
            eixo.axvspan(xs[0]-.5,xs[-1]+.5,color=cor,zorder=0)
            for split,nome,linha_cor in (('treino','Treino','#2563a6'),('validacao','Validação','#db6323')):
                eixo.plot(xs,[float(r[f'{chave}_{split}'])*fator for r in rows],
                          color=linha_cor,lw=2,label=nome if indice==1 else None)
            if indice>1:
                eixo.axvline(xs[0]-.5,color='#444',ls='--',lw=1.5)
            eixo.grid(alpha=.22)
        meio=(xs[0]+xs[-1])/2
        ax.text(meio,1.015,f'{indice} · épocas {rows[0]["epoca"]}–{rows[-1]["epoca"]}',
                transform=ax.get_xaxis_transform(),ha='center',fontsize=11,fontweight='bold')
        melhor=max(range(len(rows)),key=lambda i:float(rows[i]['acuracia_validacao']))
        valor=float(rows[melhor]['acuracia_validacao'])*100
        ax.scatter([xs[melhor]],[valor],color='#222',s=35,zorder=6)
        ax.annotate(f'{valor:.2f}%',(xs[melhor],valor),xytext=(0,-24),
                    textcoords='offset points',ha='center',fontsize=11,
                    bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=2))
        deslocamento+=len(rows)
    ax.set(ylabel='Acurácia (%)',ylim=(75,101.5))
    perda.set(ylabel='Perda BCE',xlabel='Épocas exibidas em sequência cronológica (não representam continuidade entre todos os checkpoints)')
    ax.legend(loc='lower right',fontsize=12)
    fig.suptitle('ResTheo34 → Aug8 · histórico unificado de treinamento e validação',fontsize=24,y=.97)
    fig.text(.5,.93,'Linhas verticais = mudanças de etapa · curvas não são ligadas nos reinícios · pontos pretos = melhor acurácia de validação de cada etapa',ha='center',fontsize=12)
    for i,texto in enumerate(descricoes):
        fig.text(.06,.205-i*.033,texto,fontsize=11,ha='left')
    destino=RAIZ/'resultados_residuals_networks/res_theo34_aug8'
    for ext in ('svg','png','pdf'):
        fig.savefig(destino/f'trajetoria_unificada.{ext}',dpi=170,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    arquitetura()
    trajetoria()
    trajetoria_unificada()
