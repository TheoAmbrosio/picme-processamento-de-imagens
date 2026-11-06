"""Consolida fontes e saídas existentes; não executa notebooks nem treinamentos."""
from pathlib import Path
import base64
import csv
import hashlib
import html
import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parent


def gravar(p, texto):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(texto, encoding='utf-8')


def json_salvar(p, obj):
    gravar(p, json.dumps(obj, ensure_ascii=False, indent=2)+'\n')


def diagrama(p, titulo, blocos, fonte):
    altura=100+len(blocos)*76
    partes=[f'<svg xmlns="http://www.w3.org/2000/svg" width="820" height="{altura}" viewBox="0 0 820 {altura}">',
        '<defs><marker id="a" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#334155"/></marker></defs>',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="410" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">{html.escape(titulo)}</text>']
    for i,linha in enumerate(blocos):
        y=48+i*76
        partes += [f'<rect x="45" y="{y}" width="730" height="50" rx="8" fill="#e5effb" stroke="#334155"/>',
            f'<text x="410" y="{y+30}" text-anchor="middle" font-family="sans-serif" font-size="15">{html.escape(linha)}</text>']
        if i<len(blocos)-1:partes.append(f'<path d="M410,{y+50} V{y+75}" stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')
    partes.append(f'<text x="20" y="{altura-12}" font-family="sans-serif" font-size="11">{html.escape(fonte)}</text></svg>')
    gravar(p,'\n'.join(partes));ET.parse(p)


def grafico(pasta, series, titulo):
    pasta.mkdir(parents=True,exist_ok=True)
    fig,axs=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for nome,rows in series:
        for chave,ax in (('acc',axs[0]),('loss',axs[1])):
            for fase,estilo in (('train','-'),('val','--')):
                pontos=[(r['epoch'],r[f'{fase}_{chave}']) for r in rows if f'{fase}_{chave}' in r]
                if pontos:ax.plot(*zip(*pontos),estilo,marker='.' if len(pontos)<3 else None,label=f'{nome} {fase}')
    for ax,rotulo in zip(axs,('Acurácia (fração)','Perda')):
        ax.set(xlabel='Época',ylabel=rotulo);ax.grid(alpha=.2)
        if ax.lines:ax.legend(fontsize=8)
    fig.suptitle(titulo)
    for ext in ('svg','png','pdf'):fig.savefig(pasta/f'treinamento.{ext}',dpi=140)
    plt.close(fig)


def notebook(p):
    return json.loads(p.read_text())


def principal(nome, corpo):
    pasta=RAIZ/nome/'modelo principal'
    pasta.mkdir(parents=True,exist_ok=True)
    gravar(pasta/'README.md',corpo)
    return pasta


def theo():
    origem=RAIZ.parent/'Cats and Dogs I e II'
    destino=RAIZ/'Theo/modelo principal'
    shutil.copytree(origem/'modelo principal',destino,dirs_exist_ok=True)
    p=destino/'README.md'
    s=p.read_text()
    for diretorio in ('resultados_residuals_networks','residuals%20networks','dataset_ampliado','dataset'):
        s=s.replace(f'../{diretorio}/',f'../../../Cats%20and%20Dogs%20I%20e%20II/{diretorio}/')
    p.write_text(s)
    gravar(RAIZ/'Theo/README.md','# Theo\n\n[Modelo principal: ResTheo34 Aug8](modelo%20principal/README.md).\n\nCópia documental consolidada; código, datasets e checkpoints permanecem no projeto original `../../Cats and Dogs I e II/`. Nenhum peso foi duplicado.\n')


def dulce():
    repo=RAIZ/'Dulce/projeto';origem=repo/'Cats vs Dogs'
    d=principal('Dulce','''# Dulce — CNN224 v2

Modelo final identificado explicitamente pelo README da autora:
`cats_and_dogs_224_v2.ipynb`. O experimento ResNet18 está preservado no
repositório, mas não substitui automaticamente esse modelo final.

- [Método e parâmetros](METODO.md).
- [Registro original da autora](../projeto/Cats%20vs%20Dogs/historico_decisoes.ipynb).
- [Notebook final](../projeto/Cats%20vs%20Dogs/cats_and_dogs_224_v2.ipynb).
- [Arquitetura SVG](arquitetura.svg).
- [Gráfico por épocas SVG](graficos/treinamento.svg).
- [Gráfico PNG](graficos/treinamento.png).
- [Histórico original](historico.json) e [parâmetros documentados](parametros.json).
- [Checkpoint no clone](../projeto/Cats%20vs%20Dogs/modelo_cats_and_dogs_224_v2.pt).

Fontes originais preservadas. Nenhum treinamento ou teste foi executado aqui.
As divisões diferem das dos outros participantes; estes gráficos não formam um ranking comum.
''')
    params=dict(arquitetura='CNN224 v2',filtros=[32,64,64,128,128],kernel=5,padding=1,
        dropout=.2,cabeca=[3200,1024,32,2],otimizador='Adam',learning_rate=3e-4,
        batch_size=32,epocas=120,semente=18,perda='nll_loss(log_softmax(logits)): entropia cruzada',
        entrada=[3,224,224],escala_pixels='[0,1]',divisao='15% teste;15% do restante validação; grupos inteiros;17934/3165/3723 originais',
        fonte='projeto/Cats vs Dogs/cats_and_dogs_224_v2.ipynb, células27,29,31,33')
    json_salvar(d/'parametros.json',params)
    h=json.loads((origem/'historico/historico_treino_224_v2.json').read_text())
    json_salvar(d/'historico.json',h)
    rows=[dict(epoch=i+1,train_acc=a,val_acc=h['acc_val'][i],train_loss=h['perda_treino'][i],val_loss=h['perda_val'][i]) for i,a in enumerate(h['acc_treino'])]
    grafico(d/'graficos',[('CNN224 v2',rows)],'Dulce — histórico salvo, sem reexecução')
    melhor=max(rows,key=lambda x:x['val_acc'])
    json_salvar(d/'resumo_resultado.json',dict(epocas_registradas=len(rows),melhor_validacao=melhor,fonte='historico_treino_224_v2.json'))
    gravar(d/'METODO.md',f'''# Método documentado

Cinco blocos Conv5×5(padding1) → ReLU → MaxPool2 → Dropout0.2,
canais32/64/64/128/128; mapas após pooling111/54/26/12/5.
Flatten3200 →1024 →32 →2, ReLU nas duas camadas intermediárias.
Treino com logits e entropia cruzada; forward expõe softmax.
Adam LR0.0003, batch32,120 épocas, seed18. Sem pesos pré-treinados neste modelo.

Filtragem de duplicatas/quase duplicatas antes de dividir por grupos:
15% para teste,15% do restante para validação;17934/3165/3723 originais;
descartadas imagens com lado menor que94px. Resize bilinear mantendo proporção
e corte central224. Pixels convertidos para[0,1]. Treino ganha versões fixas:
10% com rotação15° e10% com ruído gaussiano (desvio2 na escala0–255).
Em cada lote: espelhamento, zoom0.7–1, deslocamento, rotação±15°,
brilho e contraste0.8–1.2. Validação somente originais.

Histórico encontrado:{len(rows)} épocas. Melhor acurácia de validação
{melhor['val_acc']:.4%}, época{melhor['epoch']}. Critério de checkpoint: maior
acurácia de validação. Métricas de teste já presentes no repositório não
foram usadas para escolher ou alterar configurações nesta organização.
''')
    diagrama(d/'arquitetura.svg','Dulce — CNN224 v2',[
        'RGB 224×224',*[f'Conv5×5 → {c} canais → ReLU → Pool2 → Dropout20%' for c in [32,64,64,128,128]],
        'Flatten3200 → Linear1024 → ReLU','Linear32 → ReLU → Linear2 (logits no treino)'],
        'Fonte: notebook224_v2, célula27; sem execução do modelo.')
    gravar(RAIZ/'Dulce/README.md','# Dulce\n\n[Documentação consolidada](modelo%20principal/README.md). O clone original está em `projeto/`.\n')


def bandeira():
    from bandeira_atualizado import atualizar
    atualizar()


def lisandra():
    from lisandra_atualizada import consolidar
    consolidar()


def main():
    theo();dulce();bandeira();lisandra()
    for nome in ('Dulce','Bandeira'):
        pasta=RAIZ/nome/'modelo principal'
        gravar(pasta/'REGISTRO_MODELO.md',f'# Registro documental — {nome}\n\nConsolidado em06/11/2026 a partir das fontes recebidas, identificadas em ../../FONTES.json. Método em METODO.md; parâmetros em parametros.json; arquitetura SVG e gráficos preservam as diferenças de protocolo. Consulte as pendências no README. Nenhum treinamento ou teste executado; código dos autores não alterado.\n')
        with (pasta/'README.md').open('a') as f:
            f.write('\n[Registro da consolidação](REGISTRO_MODELO.md).\n')
            if nome=='Bandeira':f.write('\n[Plot geral V1–V4](graficos/treinamentos_geral.png).\n')
    fontes=[]
    for nome in ('Dulce','Bandeira','Lisandra'):
        repo=RAIZ/nome/'projeto'
        fontes.append(dict(participante=nome,url=subprocess.check_output(['git','remote','get-url','origin'],cwd=repo,text=True).strip(),
            branch=subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip(),
            commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()))
    fontes.append(dict(participante='Theo',origem='../Cats and Dogs I e II/modelo principal',modelo='ResTheo34 Aug8',melhor_epoca=29,validacao=.9832))
    json_salvar(RAIZ/'FONTES.json',fontes)
    from organizar_bancos_dados import main as bancos
    bancos()
    from organizar_evolucao import main as evolucao
    evolucao()
    print('Consolidação concluída, sem executar código dos participantes.')


if __name__=='__main__':main()
