"""Históricos de épocas já salvos: extração documental, sem treinamento."""
from pathlib import Path
import csv
import json
import re
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RAIZ=Path(__file__).resolve().parent
CAMPOS=('epoca','perda_treino','perda_validacao','acuracia_treino','acuracia_validacao')


def salvar(participante,nome,linhas,fonte,nota=''):
    pasta=RAIZ/participante/'modelo principal/evolucao do treinamento'/nome
    pasta.mkdir(parents=True,exist_ok=True)
    with (pasta/'historico.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=CAMPOS);w.writeheader();w.writerows(linhas)
    meta=dict(fonte=fonte,observacao=nota,epocas_disponiveis=len(linhas),
              unidade_acuracia='fração0–1',campos_ausentes='células vazias, sem interpolação',
              tipo='extração de dados existentes; nenhuma execução de treino')
    (pasta/'fonte.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    fig,axs=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for ax,medida,titulo in zip(axs,('acuracia','perda'),('Acurácia','Perda')):
        for fase,cor in (('treino','#2672b8'),('validacao','#df7530')):
            chave=f'{medida}_{fase}'
            pts=[(l['epoca'],l[chave]) for l in linhas if l.get(chave) is not None]
            if pts:ax.plot(*zip(*pts),color=cor,label=fase,marker='.' if len(pts)<6 else None)
        ax.set(xlabel='Época',ylabel=titulo);ax.grid(alpha=.25)
        if ax.lines:ax.legend()
        else:ax.text(.5,.5,'Não registrado na fonte',ha='center',transform=ax.transAxes)
    fig.suptitle(f'{participante} — {nome}'+ ('\n'+nota if nota else ''),fontsize=12)
    for ext in ('svg','png','pdf'):fig.savefig(pasta/f'graficos.{ext}',dpi=140)
    plt.close(fig)
    return dict(nome=nome,epocas=len(linhas),fonte=fonte,nota=nota)


def converter(h):
    mapa={'perda_treino':'perda_treino','perda_val':'perda_validacao',
          'acc_treino':'acuracia_treino','acc_val':'acuracia_validacao'}
    total=max(len(v) for v in h.values() if isinstance(v,list))
    return [dict(epoca=i+1,**{dest:h[k][i] for k,dest in mapa.items() if k in h and i<len(h[k])}) for i in range(total)]


def indice(nome,entradas,nota=''):
    p=RAIZ/nome/'modelo principal/evolucao do treinamento';p.mkdir(exist_ok=True)
    s=f'# Evolução do treinamento — {nome}\n\nPerda e acurácia de treino/validação por época, conforme fontes salvas.\n{nota}\n\n'
    for e in entradas:
        n=e['nome'];s+=f'## {n}\n\nÉpocas disponíveis:{e["epocas"]}. {e["nota"]}\n\n- [Gráfico SVG](<{n}/graficos.svg>) · [PNG](<{n}/graficos.png>) · [PDF](<{n}/graficos.pdf>).\n- [Tabela CSV](<{n}/historico.csv>) · [Origem dos dados](<{n}/fonte.json>).\n\n'
    (p/'README.md').write_text(s)
    (p/'inventario.json').write_text(json.dumps(entradas,ensure_ascii=False,indent=2)+'\n')
    readme=p.parent/'README.md';texto=readme.read_text()
    if 'evolucao%20do%20treinamento/README.md' not in texto:
        readme.write_text(texto+'\n## Evolução de treino e validação por época\n\n[Gráficos, tabelas CSV e fontes dos históricos](evolucao%20do%20treinamento/README.md).\n')


def theo():
    principal=RAIZ/'Theo/modelo principal'
    resumo=json.loads((principal/'graficos/resumo_trajetoria.json').read_text())
    etapas=(
        ('01_referencia_historica','Referência metodológica; interrompida na21. Não forneceu pesos à repetição.'),
        ('02_repeticao','Nova partida ImageNet;36 épocas observadas. Checkpoint33 origina os dois ramos seguintes.'),
        ('03_primeiro_finetuning','Ramo independente do checkpoint33 da repetição;16 épocas. Não é origem dos pesos do Aug8.'),
        ('Aug8','Novo ramo direto do checkpoint33 da repetição;30 épocas. Modelo principal.'))
    assert len(resumo)==len(etapas), 'Confira as etapas selecionadas na linhagem.'
    registros=[];tabela=[]
    for etapa,(nome,nota) in zip(resumo,etapas):
        p=RAIZ.parent/'Cats and Dogs I e II/resultados_residuals_networks'/etapa['execucao']/'historico.csv'
        with p.open() as f:
            linhas=[dict(epoca=int(r['epoca']),perda_treino=float(r['bce_treino']),
                perda_validacao=float(r['bce_validacao']),acuracia_treino=float(r['acuracia_treino']),
                acuracia_validacao=float(r['acuracia_validacao'])) for r in csv.DictReader(f)
                if etapa['epocas_observadas'][0]<=int(r['epoca'])<=etapa['epocas_observadas'][1]]
        registros.append(salvar('Theo',nome,linhas,str(p.relative_to(RAIZ.parent)),nota))
        tabela.extend(dict(etapa=nome,**r) for r in linhas)
    indice('Theo',registros,'''## Visão geral até o Aug8

- [Plot geral SVG com divisões entre etapas](visao%20geral/trajetoria_unificada.svg).
- [Plot geral PNG](visao%20geral/trajetoria_unificada.png).
- [Plot geral PDF](visao%20geral/trajetoria_unificada.pdf).
- [Painéis por etapa](visao%20geral/trajetoria_treinamentos.png).
- [Tabela completa, identificando a etapa e a época](visao%20geral/historico_etapas.csv).
- [Linhagem e origem dos pesos](../documentacao/LINHAGEM_MODELO_PRINCIPAL.md).

```text
1. Referência histórica (referência de método, não origem dos pesos)
2. Repetição: nova partida ImageNet → checkpoint da época33
   ├── 3. Primeiro fine-tuning:16 épocas, ramo independente
   └── 4. Aug8:30 épocas, modelo principal
```

As quatro partes estão na apresentação; isso não significa que todos os
pesos passaram por elas em sequência. O eixo do plot geral organiza as etapas
na apresentação, não representa uma única contagem de épocas de pesos herdados.
A continuação37–50 retirada anteriormente continua fora desta seleção.
''')
    destino=principal/'evolucao do treinamento/visao geral';destino.mkdir(exist_ok=True)
    for nome in ('trajetoria_unificada.svg','trajetoria_unificada.png','trajetoria_unificada.pdf',
                 'trajetoria_treinamentos.png','trajetoria_treinamentos.pdf','resumo_trajetoria.json'):
        shutil.copy2(principal/'graficos'/nome,destino/nome)
    with (destino/'historico_etapas.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=('etapa',*CAMPOS));w.writeheader();w.writerows(tabela)


def dulce():
    base=RAIZ/'Dulce/projeto/Cats vs Dogs';entradas=[]
    p=base/'historico/historico_treino_224_v2.json'
    entradas.append(salvar('Dulce','CNN224_v2_120ep',converter(json.loads(p.read_text())),str(p.relative_to(RAIZ))))
    p=base/'historico/historico_treino.json';h=json.loads(p.read_text())
    entradas.append(salvar('Dulce','historico_repeticao_100ep',converter(h['repeticao']),str(p.relative_to(RAIZ))+' chave repeticao',
        'Primeiro modelo, SGD LR0.01: repetição com mesma configuração e seed, confirmada pela autora.'))
    linhas=[dict(epoca=i+1,acuracia_validacao=a) for i,a in enumerate(h['acc_val_modelo_fechado'])]
    entradas.append(salvar('Dulce','historico_modelo_fechado_validacao',linhas,str(p.relative_to(RAIZ))+' chave acc_val_modelo_fechado',
        'Primeiro modelo, SGD LR0.01: execução original, só validação; confirmada pela autora.'))
    padrao=re.compile(r'Época\s+(\d+)\s+perda no treino:\s*([\d.]+)\s+acurácia no treino:\s*([\d.]+)\s+(?:perda na validação:\s*([\d.]+)\s+)?acurácia na validação:\s*([\d.]+)')
    for nome in ('cats_and_dogs_224.ipynb','cats_and_dogs_resnet.ipynb'):
        p=base/nome;n=json.loads(p.read_text())
        for i,c in enumerate(n['cells']):
            texto='\n'.join(''.join(o.get('text',[])) for o in c.get('outputs',[]))
            linhas=[]
            for ep,pt,at,pv,av in padrao.findall(texto):
                r=dict(epoca=int(ep),perda_treino=float(pt),acuracia_treino=float(at),acuracia_validacao=float(av))
                if pv:r['perda_validacao']=float(pv)
                linhas.append(r)
            if linhas:
                entradas.append(salvar('Dulce',p.stem+f'_celula{i}',linhas,str(p.relative_to(RAIZ))+f', célula{i}',
                    'Valores arredondados impressos no notebook; execução separada.'))
    indice('Dulce',entradas,'As duas séries de100 épocas são do primeiro modelo cats_and_dogs.ipynb, SGD LR0.01, conforme esclarecimento da autora. acc_val_modelo_fechado é a execução original (só validação); repeticao é outra execução com mesma configuração e seed, feita depois para registrar as quatro métricas. Não são continuação nem reconstrução exata das métricas de treino da execução original. JSON preserva mais precisão que logs impressos; execuções não foram concatenadas.')


def lisandra():
    from lisandra_atualizada import evolucao
    evolucao()


def bandeira():
    base=RAIZ/'Bandeira/modelo principal';repo=RAIZ/'Bandeira/projeto';entradas=[]
    fig,axs=plt.subplots(4,2,figsize=(16,17),layout='constrained')
    for i,suf in enumerate(('', '_2.0','_3.0','_4.0')):
        p=repo/f'data/training_history{suf}.csv'
        with p.open() as f:
            rows=[dict(epoca=int(float(r['epoch'])),perda_treino=float(r['train_loss']),
                perda_validacao=float(r['val_loss']),acuracia_treino=float(r['train_acc']),
                acuracia_validacao=float(r['val_acc'])) for r in csv.DictReader(f)]
        best=min(rows,key=lambda r:r['perda_validacao'])
        nota=f'Checkpoint por menor perda de validação: época{best["epoca"]}. Execução independente.'
        entradas.append(salvar('Bandeira',f'V{i+1}',rows,str(p.relative_to(RAIZ)),nota))
        for ax,chave in zip(axs[i],('perda','acuracia')):
            for fase in ('treino','validacao'):ax.plot([r['epoca'] for r in rows],[r[f'{chave}_{fase}'] for r in rows],label=fase)
            ax.axvline(best['epoca'],ls=':',color='black',label='checkpoint por val_loss')
            ax.set(title=f'V{i+1}: {chave}',xlabel='Época',ylabel=chave);ax.legend(fontsize=8);ax.grid(alpha=.2)
    destino=base/'evolucao do treinamento/visao geral';destino.mkdir(exist_ok=True)
    for ext in ('svg','png','pdf'):fig.savefig(destino/f'treinamentos_geral.{ext}',dpi=140)
    plt.close(fig)
    indice('Bandeira',entradas,'''## Visão geral

[SVG com as quatro versões](visao%20geral/treinamentos_geral.svg),
[PNG](visao%20geral/treinamentos_geral.png), [PDF](visao%20geral/treinamentos_geral.pdf).

Históricos numéricos publicados no repositório atualizado, não estimados das
imagens. V1–V4 são execuções separadas. A linha vertical marca a época
selecionada por menor perda de validação, conforme critério do autor.
''')


def main():
    theo();dulce();lisandra();bandeira()
    p=RAIZ/'EVOLUCAO_TREINAMENTOS.md'
    p.write_text('''# Treino e validação ao longo das épocas

Esta seção é sobre resultados de treinamento, distinta da documentação dos bancos de imagens.

| Participante | Material encontrado | Abrir |
|---|---|---|
| Theo |Referência21 épocas, repetição36, primeiro fine-tuning16 e Aug8 com30; CSVs e plot geral|[Evolução](Theo/modelo%20principal/evolucao%20do%20treinamento/README.md)|
| Dulce |CNN final120 épocas; séries históricas100 épocas; logs224 e ResNet18|[Evolução](Dulce/modelo%20principal/evolucao%20do%20treinamento/README.md)|
| Bandeira |Históricos completos V1–V4, CSVs e gráficos numéricos|[Evolução](Bandeira/modelo%20principal/evolucao%20do%20treinamento/README.md)|
| Lisandra |Pacote com10 épocas;checkpoint declarado da5 por menor loss;novo peso compatível recebido|[Evolução](Lisandra/modelo%20principal/evolucao%20do%20treinamento/README.md)|

Gráficos dos dados numéricos em SVG/PNG/PDF, tabelas em CSV, fonte identificada
por execução. Ausências não foram interpoladas. Séries independentes não foram
fundidas como continuação. Não foi executado nenhum treinamento.
''')
    print('Evolução por época organizada para os quatro participantes.')


if __name__=='__main__':main()
