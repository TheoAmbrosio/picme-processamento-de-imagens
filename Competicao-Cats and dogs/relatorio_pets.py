"""Relatório descritivo de avaliação já executada; não carrega nem ajusta modelos."""
from pathlib import Path
import argparse,csv,json,math,html,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NOMES=['Theo','Lisandra','Dulce','Bandeira']

def gerar(p):
    manifesto=json.loads((p/'manifesto.json').read_text());rotulos={}
    for r in manifesto:
        pasta=Path(r['arquivo_zip']).parent.name
        if pasta not in ('Gatos','Cachorros'):raise ValueError('Pasta sem rótulo explícito: '+pasta)
        rotulos[r['filename']]=int(pasta=='Cachorros')
    resultados=[];erros={};matrizes={}
    for nome in NOMES:
        rows=list(csv.DictReader((p/f'{nome}.csv').open()));assert set(rotulos)=={r['filename'] for r in rows}
        tn=fp=fn=tp=0;linhas=[]
        for r in rows:
            y=rotulos[r['filename']];pred=int(r['pred_label']);prob=float(r['probability'])
            assert pred==int(prob>=.5)
            tn+=int(y==pred==0);tp+=int(y==pred==1);fp+=int(y==0 and pred==1);fn+=int(y==1 and pred==0)
            if y!=pred:linhas.append(dict(filename=r['filename'],real=y,pred_label=pred,probability=prob))
        acc=(tp+tn)/len(rows);prec=tp/(tp+fp) if tp+fp else 0;rec=tp/(tp+fn) if tp+fn else 0;f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
        erros[nome]=linhas;matrizes[nome]=[[tn,fp],[fn,tp]]
        resultados.append(dict(modelo=nome,imagens=len(rows),acertos=tp+tn,erros=fp+fn,acuracia=acc,precisao_cao=prec,recall_cao=rec,f1_cao=f1,recall_gato=tn/(tn+fp),acuracia_balanceada=(rec+tn/(tn+fp))/2,gatos_corretos=tn,gatos_como_caes=fp,caes_como_gatos=fn,caes_corretos=tp))
    # Ranking por acurácia; empates recebem a mesma posição.
    resultados.sort(key=lambda r:-r['acertos']);ultimo=None;pos=0
    for i,r in enumerate(resultados,1):
        if r['acertos']!=ultimo:pos=i;ultimo=r['acertos']
        r['posicao']=pos
    campos=['posicao',*list(resultados[0])[:-1]]
    with (p/'ranking.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=campos);w.writeheader();w.writerows(resultados)
    (p/'metricas.json').write_text(json.dumps(dict(rotulos='pastas do ZIP original: Gatos=0,Cachorros=1',ranking_por='acuracia; empates preservados',resultados=resultados,matrizes=matrizes),ensure_ascii=False,indent=2)+'\n')
    with (p/'erros_por_modelo.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['modelo','filename','real','pred_label','probability']);w.writeheader()
        for nome,rs in erros.items():w.writerows(dict(modelo=nome,**r) for r in rs)
    plt.rcParams.update({'svg.fonttype':'none','font.size':12})
    fig,axs=plt.subplots(1,2,figsize=(13,5.8),layout='constrained')
    ns=[r['modelo'] for r in resultados];colors=['#286b78' if n=='Theo' else '#8b5290' if n=='Lisandra' else '#b96238' if n=='Dulce' else '#4367a6' for n in ns]
    for ax,chave,titulo in zip(axs,['acuracia','f1_cao'],['Acurácia — ranking','F1 da classe Cão']):
        vs=[r[chave]*100 for r in resultados];bars=ax.barh(ns,vs,color=colors);ax.invert_yaxis();ax.set(xlim=(0,110),xlabel='%',title=titulo);ax.set_xticks(range(0,101,20));ax.grid(axis='x',alpha=.2)
        for b,r,v in zip(bars,resultados,vs):ax.text(v+1,b.get_y()+b.get_height()/2,f'{v:.2f}%'+(f" · {r['acertos']}/77" if chave=='acuracia' else ''),va='center',fontsize=10)
    fig.suptitle('Pets — 77 imagens · 45 gatos e 32 cães',fontsize=18)
    for ext in ['svg','png','pdf']:fig.savefig(p/f'ranking.{ext}',dpi=180)
    plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(11,10),layout='constrained')
    for ax,nome in zip(axs.flat,NOMES):
        arr=np.array(matrizes[nome]);ax.imshow(arr,cmap='Blues',vmin=0,vmax=45)
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Gato','Cão'],yticklabels=['Gato','Cão'],xlabel='Previsto',ylabel='Real',title=nome)
        for i in range(2):
            for j in range(2):ax.text(j,i,str(arr[i,j]),ha='center',va='center',fontsize=25,color='white' if arr[i,j]>25 else '#202c39')
    fig.suptitle('Matrizes de confusão — referência: pastas do ZIP',fontsize=17)
    for ext in ['svg','png']:fig.savefig(p/f'matrizes_confusao.{ext}',dpi=160)
    plt.close(fig)
    pct=lambda v:f'{v*100:.2f}%'.replace('.',',')
    texto='''# Performance dos quatro modelos no banco Pets

## Como foi feita a comparação

Os quatro modelos receberam as mesmas 77 imagens:45 gatos e32 cachorros.
Os rótulos de referência foram retirados das pastas Gatos e Cachorros do ZIP enviado;
não houve revisão manual independente desses rótulos. A classe positiva é Cão=1.
O limiar foi0,5:probabilidade maior ou igual a0,5 vira Cão;abaixo disso vira Gato.

Foram usados os checkpoints fechados e o pré-processamento de cada participante.
Nenhum modelo foi retreinado, calibrado ou selecionado com esses resultados.
Theo:Aug8;Lisandra:AlexNet com BatchNorm;Dulce:CNN224v2;
Bandeira:V4 Final. Todos em eval(), com dropout desativado e inferência CUDA.
O Bandeira segue o fluxo publicado para imagem nova, com resize direto384×384;
a diferença previamente documentada em relação à preparação offline do treino permanece.

## Ranking

Critério:maior acurácia no conjunto completo. Empates mantêm a mesma posição.
Precisão, recall e F1 abaixo são da classe Cão. Também mostramos acurácia
balanceada, que dá o mesmo peso ao acerto em gatos e em cães.

|Posição|Modelo|Acertos|Acurácia|Precisão Cão|Recall Cão|F1 Cão|Acurácia balanceada|
|---|---|---|---|---|---|---|---|
'''
    for r in resultados:texto+=f"|{r['posicao']}|{r['modelo']}|{r['acertos']}/77|{pct(r['acuracia'])}|{pct(r['precisao_cao'])}|{pct(r['recall_cao'])}|{pct(r['f1_cao'])}|{pct(r['acuracia_balanceada'])}|\n"
    texto+='\n## Resultado de cada participante\n'
    for r in resultados:
        texto+=f"\n### {r['modelo']}\n\nAcertou{r['acertos']} das77 imagens ({pct(r['acuracia'])}) e errou{r['erros']}. "
        texto+=f"Entre os45 gatos, acertou{r['gatos_corretos']} e classificou{r['gatos_como_caes']} como cães. "
        texto+=f"Entre os32 cães, acertou{r['caes_corretos']} e classificou{r['caes_como_gatos']} como gatos. "
        texto+=f"O F1 da classe Cão foi{pct(r['f1_cao'])}.\n"
    melhor=resultados[0];texto+=f"\n## Leitura dos resultados\n\nNeste conjunto, {melhor['modelo']} ficou na primeira posição por acurácia, com{melhor['acertos']} acertos. "
    texto+='Cada imagem equivale a aproximadamente1,30 ponto percentual de acurácia. '
    texto+='As diferenças descrevem este banco pequeno e não demonstram, sozinhas, superioridade geral. '
    texto+='O conjunto contém mais gatos que cães; por isso o relatório separa os erros por classe e inclui acurácia balanceada.\n\n'
    texto+='A unanimidade foi44/77 e houve divergência em33 imagens. Concordância entre modelos não é uma medida de acerto. As probabilidades são as saídas contínuas dos modelos, sem calibração adicional.\n\n'
    texto+='## Arquivos\n\n- [Ranking visual](ranking.svg).\n- [Matrizes de confusão](matrizes_confusao.svg).\n- [Fotos e predições](amostra_classificacoes.png).\n- [CSV do ranking](ranking.csv).\n- [Erros por modelo](erros_por_modelo.csv).\n- [Quatro CSVs de predição](predicoes_csv.zip).\n- [Proveniência e pipelines](execucao.json).\n'
    # Espaçamento de números para leitura, sem alterar dados.
    import re
    texto=re.sub(r'(?<=[A-Za-zÀ-ÿ])(?=\d)', ' ',texto);texto=re.sub(r'(?<=\d)(?=[A-Za-zÀ-ÿ])',' ',texto)
    (p/'RELATORIO_PERFORMANCE.md').write_text(texto)
    linhas=''.join(f"<tr><td>{r['posicao']}</td><td>{r['modelo']}</td><td>{r['acertos']}/77</td><td>{pct(r['acuracia'])}</td><td>{pct(r['f1_cao'])}</td></tr>" for r in resultados)
    doc=f'''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Pets — resultados</title><style>body{{max-width:1200px;margin:40px auto;padding:20px;font:18px sans-serif;color:#202c39;background:#fcfbf8}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{padding:12px;border-bottom:1px solid #ccc}}a{{color:#286b78}}</style><h1>Pets — performance dos quatro modelos</h1><p>77 imagens ·45 gatos ·32 cães. Referência:pastas do ZIP. Ranking por acurácia.</p><table><tr><th>Posição</th><th>Modelo</th><th>Acertos</th><th>Acurácia</th><th>F1 Cão</th></tr>{linhas}</table><p><a href="RELATORIO_PERFORMANCE.md">Relatório completo</a> · <a href="predicoes_csv.zip">Quatro CSVs</a> · <a href="ranking.csv">Ranking CSV</a></p><img src="ranking.svg"><img src="matrizes_confusao.svg"><img src="concordancia.svg"><h2>Amostra de divergências</h2><img src="amostra_classificacoes.png"><p>Comparação descritiva deste banco;nenhum ajuste foi feito com esses resultados.</p></html>'''
    (p/'index.html').write_text(doc)
    (p/'README.md').write_text('# Resultados Pets\n\n[Abrir painel](index.html) · [Relatório escrito](RELATORIO_PERFORMANCE.md).\n\nCSV por participante:Theo.csv, Lisandra.csv, Dulce.csv, Bandeira.csv.\n\nColunas:filename, probability=P(Cão), pred_label (>=0,5→1; caso contrário0).\n\nRótulos de referência:pastas do ZIP;45 gatos e32 cães.\n')
    (p.parent/'README.md').write_text('# Resultados Pets para apresentação\n\n[Abrir avaliação de06/11/2026](<'+p.name+'/index.html>) · [Relatório](<'+p.name+'/RELATORIO_PERFORMANCE.md>).\n')
    from metodos_treinamento_pets import adicionar
    adicionar(p)
    comparar_validacao(p,resultados)
    report=p/'RELATORIO_PERFORMANCE.md'
    report.write_text(report.read_text().replace('p,p,','p.p.'))
    print(json.dumps(resultados,ensure_ascii=False,indent=2))

def comparar_validacao(p,resultados):
    k=Path(__file__).resolve().parent
    def linha(path,col,ep):return next(r for r in csv.DictReader(path.open()) if int(r[col])==ep)
    fontes={
      'Theo':(k/'Theo/modelo principal/treinamento_aug8/historico.csv',29,'maior acurácia de validação'),
      'Lisandra':(k/'Lisandra/modelo principal/evolucao do treinamento/historico_epocas.csv',5,'menor perda de validação; associação ao peso declarada pela autora'),
      'Dulce':(k/'Dulce/projeto/Cats vs Dogs/historico/historico_treino_224_v2.json',112,'maior acurácia de validação'),
      'Bandeira':(k/'Bandeira/projeto/data/training_history_4.0_Final.csv',10,'menor perda de validação na fase Final')}
    dados=[]
    for r in resultados:
        nome=r['modelo'];arquivo,ep,criterio=fontes[nome]
        if nome=='Dulce':
            h=json.loads(arquivo.read_text());val=h['acc_val'][ep-1];tr=h['acc_treino'][ep-1];loss=h['perda_val'][ep-1]
        else:
            row=linha(arquivo,'epoch' if nome=='Bandeira' else 'epoca',ep)
            val=float(row['val_acc' if nome=='Bandeira' else 'acuracia_validacao']);tr=float(row['train_acc' if nome=='Bandeira' else 'acuracia_treino']);loss=float(row['val_loss' if nome=='Bandeira' else 'bce_validacao' if nome=='Theo' else 'perda_validacao'])
        dados.append(dict(modelo=nome,epoca_checkpoint=ep,acuracia_treino_checkpoint=tr,acuracia_validacao_checkpoint=val,perda_validacao_checkpoint=loss,acuracia_pets=r['acuracia'],diferenca_pets_menos_validacao_pp=(r['acuracia']-val)*100,criterio_checkpoint=criterio,fonte=str(arquivo.relative_to(k))))
    with (p/'validacao_vs_pets.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dados[0]));w.writeheader();w.writerows(dados)
    fig,ax=plt.subplots(figsize=(12,6),layout='constrained');x=np.arange(4);largura=.34
    a=ax.bar(x-largura/2,[d['acuracia_validacao_checkpoint']*100 for d in dados],largura,label='Validação do treinamento',color='#426f9e')
    b=ax.bar(x+largura/2,[d['acuracia_pets']*100 for d in dados],largura,label='Nosso banco Pets',color='#df9352')
    for bars in (a,b):
        for bar in bars:ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f'{bar.get_height():.2f}%',ha='center',fontsize=10)
    ax.set(xticks=x,xticklabels=[d['modelo'] for d in dados],ylim=(0,113),ylabel='Acurácia (%)',title='Validação do treinamento × banco Pets (77 imagens)');ax.set_yticks(range(0,101,20));ax.legend(loc='upper right');ax.grid(axis='y',alpha=.2)
    fig.text(.5,.01,'Validações próprias e distintas; comparação descritiva, sem ajuste dos modelos.',ha='center',fontsize=10)
    for ext in ('svg','png','pdf'):fig.savefig(p/f'validacao_vs_pets.{ext}',dpi=180)
    plt.close(fig)
    pct=lambda v:f'{v*100:.2f}%'.replace('.',',')
    trecho='\n## Validação do treinamento × nosso banco Pets\n\nOs números de validação correspondem aos modelos usados nesta avaliação. Cada participante possui seu próprio conjunto de validação.\n\n|Modelo|Validação|Pets|Pets − validação|\n|---|---|---|---|\n'
    for d in dados:trecho+=f"|{d['modelo']}|{pct(d['acuracia_validacao_checkpoint'])}|{pct(d['acuracia_pets'])}|{d['diferenca_pets_menos_validacao_pp']:+.2f} p.p.|\n".replace('.',',')
    trecho+='\nTodos tiveram acurácia menor no banco Pets que na sua validação. A diferença é observada entre conjuntos distintos e não identifica, sozinha, sua causa. As 77 fotos não têm o mesmo tamanho, composição e distribuição dos conjuntos originais. Não foi feita alteração nos modelos a partir deste teste.\n\n[Gráfico de comparação](validacao_vs_pets.svg) · [Tabela completa e fontes](validacao_vs_pets.csv). A tabela também preserva acurácia de treino e perda de validação do modelo avaliado; acurácia de treino é a registrada durante o treino, com as ampliações e regularizações de cada modelo.\n'
    report=p/'RELATORIO_PERFORMANCE.md';s=report.read_text();s=s.replace('\n## Leitura dos resultados',trecho+'\n## Leitura dos resultados');report.write_text(s)
    linhas=''.join(f"<tr><td>{d['modelo']}</td><td>{pct(d['acuracia_validacao_checkpoint'])}</td><td>{pct(d['acuracia_pets'])}</td><td>{d['diferenca_pets_menos_validacao_pp']:+.2f} p.p.</td></tr>" for d in dados)
    section='<h2>Validação do treinamento × Pets</h2><p>Cada validação pertence ao respectivo participante.</p><table><tr><th>Modelo</th><th>Validação</th><th>Pets</th><th>Diferença</th></tr>'+linhas+'</table><img src="validacao_vs_pets.svg"><p><a href="validacao_vs_pets.csv">CSV completo e fontes</a></p>'
    index=p/'index.html';s=index.read_text().replace('<img src="ranking.svg">',section+'<img src="ranking.svg">');index.write_text(s)
    print('Comparação validação × Pets adicionada.')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('pasta',type=Path);gerar(ap.parse_args().pasta)
