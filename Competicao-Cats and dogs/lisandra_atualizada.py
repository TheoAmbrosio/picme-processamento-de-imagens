"""Consolidação da versão vigente da Lisandra publicada no GitHub."""
from pathlib import Path
import base64
import csv
import hashlib
import json
import re
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RAIZ=Path(__file__).resolve().parent
REPO=RAIZ/'Lisandra/projeto'
DESTINO=RAIZ/'Lisandra/modelo principal'


def escrever(p,s):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(s,encoding='utf-8')


def js(p,o):escrever(p,json.dumps(o,ensure_ascii=False,indent=2)+'\n')


def ler():
    p=REPO/'CatsandDogs_classification.ipynb'
    return p,json.loads(p.read_text())


def banco():
    if (REPO/'resultados_para_competicao/configuracao.json').exists():
        from lisandra_publicada import banco_publicado
        return banco_publicado()
    d=DESTINO/'banco de dados'
    escrever(d/'AlexNetBinary.md','''# Lisandra — dados da versão vigente no GitHub

Fonte: `projeto/CatsandDogs_classification.ipynb`, células2–4.
Kaggle `bhavikjikadara/dog-and-cat-classification-dataset`, via kagglehub.
Saída publicada:24998 imagens;19998 treino,2500 validação,2500 teste.
Split80/10/10 com shuffle=True, sem random_state ou stratify explícitos.
Nenhum manifesto nominal dos splits ou contagem por classe foi publicado.

Rótulo usa a pasta pai imediata:1 se parent_dir.lower()=='dog',0 caso
contrário. Para pastas Cat/Dog, Cat=0 e Dog=1. A regra atual foi conferida
no código; não houve nova leitura de todas as imagens nesta organização.

PIL RGB → Resize((224,224)), esticando a proporção → ToTensor[0,1].
Sem Normalize por canal nem augmentation explícito. Mesma transformação
nos três splits. Sem tratamento explícito de duplicatas ou imagens ilegíveis.

Batch efetivo50, apesar de comentário ainda citar500. Saída:400 lotes de
treino,50 validação e50 teste. Último lote do treino tem48 imagens pelas
contagens publicadas. Treino shuffle=True, demais shuffle=False.

A célula4 publica uma prancha de10 imagens com rótulos, extraída para
../amostra_rotulos.png. Ela é uma amostra, não verificação exaustiva do banco.
Não foram baixados dados nem executadas células do notebook aqui.
''')
    escrever(DESTINO/'BANCO_DE_DADOS.md','''# Banco de dados — Lisandra

[Dados de treino, validação e teste da versão atual](banco%20de%20dados/AlexNetBinary.md).

Referência única: notebook CatsandDogs_classification.ipynb do GitHub.
''')


def evolucao():
    if (REPO/'resultados_para_competicao/historico_epocas.csv').exists():
        from lisandra_publicada import evolucao_publicada
        return evolucao_publicada()
    recebido=RAIZ/'Lisandra/resultados_recebidos/training_history_Lisandra.csv'
    if recebido.exists():
        evolucao_csv(recebido)
        return
    p,n=ler();d=DESTINO/'evolucao do treinamento';d.mkdir(parents=True,exist_ok=True)
    texto='\n'.join(''.join(o.get('text',[])) for c in n['cells'] for o in c.get('outputs',[]))
    padrao=r'\[Treino\] Época (\d+) \| Lote (\d+)/(\d+) -> Perda: ([\d.]+) \| Acurácia: ([\d.]+)%'
    lotes=[dict(epoca=int(e),lote=int(l),lotes_previstos=int(t),perda=float(v),acuracia=float(a)/100) for e,l,t,v,a in re.findall(padrao,texto)]
    with (d/'progresso_lotes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=('epoca','lote','lotes_previstos','perda','acuracia'));w.writeheader();w.writerows(lotes)
    resumos=re.findall(r'RESUMO ÉPOCA (\d+).*?Treino - Perda Média: ([\d.]+) \| Acurácia Média: ([\d.]+)%\s*Valid\s*- Perda Média: ([\d.]+) \| Acurácia Média: ([\d.]+)%',texto,re.S)
    with (d/'historico_epocas.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(('epoca','perda_treino','acuracia_treino','perda_validacao','acuracia_validacao'))
        for e,pt,at,pv,av in resumos:w.writerow((e,pt,float(at)/100,pv,float(av)/100))
    js(d/'disponibilidade.json',dict(fonte=str(p.relative_to(RAIZ)),epocas_configuradas=5,
        epocas_com_resumo=len(resumos),lotes_treino_salvos=len(lotes),ultimo_lote=lotes[-1] if lotes else None,
        validacao_publicada=bool(resumos),dispositivo_registrado='cpu' if 'dispositivo: cpu' in texto else 'não identificado',
        observacao='Valores arredondados impressos; lotes não equivalem a médias por época.'))
    escrever(DESTINO/'saidas_notebook.txt',texto)
    fig,axs=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for ax,chave in zip(axs,('perda','acuracia')):
        if lotes:ax.plot([l['lote'] for l in lotes],[l[chave] for l in lotes],label='Treino: lote individual')
        ax.set(xlabel='Lote da época1',ylabel=chave.capitalize());ax.grid(alpha=.2)
    fig.suptitle('Lisandra — progresso parcial por lote; sem validação publicada')
    for ext in ('svg','png','pdf'):fig.savefig(d/f'progresso_lotes.{ext}',dpi=140)
    plt.close(fig)
    escrever(d/'README.md',f'''# Evolução do treino — Lisandra, versão vigente

Notebook configurado para5 épocas, com execução registrada em CPU.
Saídas contêm{len(lotes)} lotes de treino e{len(resumos)} resumos completos de época.
Último registro:{'época'+str(lotes[-1]['epoca'])+', lote'+str(lotes[-1]['lote'])+'/'+str(lotes[-1]['lotes_previstos']) if lotes else 'ausente'}.

- [Progresso por lote SVG](progresso_lotes.svg), [PNG](progresso_lotes.png), [PDF](progresso_lotes.pdf).
- [Tabela por lote CSV](progresso_lotes.csv).
- [Tabela de épocas CSV](historico_epocas.csv) — apenas cabeçalho se nenhuma época completa foi publicada.
- [Disponibilidade e fonte](disponibilidade.json).

Não há médias de validação publicadas nesta revisão. Os lotes individuais
não foram usados para inventar uma curva por época ou medir desempenho final.
Nenhum treinamento foi executado nesta organização.
''')


def evolucao_csv(recebido):
    with recebido.open(encoding='utf-8-sig') as f:originais=list(csv.DictReader(f))
    assert originais, 'CSV vazio.'
    linhas=[]
    for row in originais:
        r=dict(epoca=int(row['epoch']),perda_treino=float(row['train_loss']),
            perda_validacao=float(row['val_loss']),acuracia_treino=float(row['train_acc'])/100,
            acuracia_validacao=float(row['val_acc'])/100)
        assert all(0<=r[k]<=1 for k in ('acuracia_treino','acuracia_validacao'))
        assert r['perda_treino']>=0 and r['perda_validacao']>=0
        for fase in ('train','val'):
            assert abs(float(row[f'{fase}_acc'])+float(row[f'{fase}_err'])-100)<1e-8
        linhas.append(r)
    assert [r['epoca'] for r in linhas]==list(range(1,len(linhas)+1))
    d=DESTINO/'evolucao do treinamento';d.mkdir(parents=True,exist_ok=True)
    with (d/'historico_epocas.csv').open('w',newline='') as f:
        campos=('epoca','perda_treino','perda_validacao','acuracia_treino','acuracia_validacao')
        w=csv.DictWriter(f,fieldnames=campos);w.writeheader();w.writerows(linhas)
    melhor_acc=max(linhas,key=lambda r:r['acuracia_validacao'])
    melhor_loss=min(linhas,key=lambda r:r['perda_validacao'])
    meta=dict(fonte=str(recebido.relative_to(RAIZ)),sha256=hashlib.sha256(recebido.read_bytes()).hexdigest(),
        epocas_com_resumo=len(linhas),unidade_original='acurácia e erro em porcentagem0–100',
        unidade_csv_normalizado='acurácia em fração0–1',melhor_acuracia_validacao=melhor_acc,
        menor_perda_validacao=melhor_loss,ultima_epoca=linhas[-1],
        correspondencia_checkpoint='não confirmada pela autora',
        correspondencia_configuracao='CSV tem10 épocas; notebook publicado configura5; não atribuída configuração automaticamente',
        observacao='Dados enviados separadamente pela autora via Theo; nenhum treinamento reexecutado.')
    js(d/'disponibilidade.json',meta)
    fig,axs=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for ax,medida,ylabel in zip(axs,('perda','acuracia'),('Perda (valor registrado no CSV)','Acurácia (fração)')):
        for fase in ('treino','validacao'):
            ax.plot([r['epoca'] for r in linhas],[r[f'{medida}_{fase}'] for r in linhas],marker='.',label=fase)
        ax.set(xlabel='Época',ylabel=ylabel);ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Lisandra — histórico recebido: treino e validação por época')
    for ext in ('svg','png','pdf'):fig.savefig(d/f'evolucao_epocas.{ext}',dpi=140)
    plt.close(fig)
    escrever(d/'README.md',f'''# Evolução de treino e validação — Lisandra

CSV recebido da autora via Theo: **{len(linhas)} épocas completas**, com perda,
acurácia e erro de treino/validação. Arquivo original preservado; acurácias
convertidas de porcentagem para fração0–1 somente na cópia normalizada.

- [Gráfico SVG](evolucao_epocas.svg), [PNG](evolucao_epocas.png), [PDF](evolucao_epocas.pdf).
- [Histórico normalizado CSV](historico_epocas.csv).
- [CSV original recebido](../../resultados_recebidos/training_history_Lisandra.csv).
- [Fonte, integridade e resumo](disponibilidade.json).

Maior acurácia de validação:{melhor_acc['acuracia_validacao']:.2%}, época{melhor_acc['epoca']}.
Menor perda de validação:{melhor_loss['perda_validacao']:.6f}, época{melhor_loss['epoca']}.
Última época: treino{linhas[-1]['acuracia_treino']:.2%}, validação{linhas[-1]['acuracia_validacao']:.2%}.

## Associação aos pesos e configuração

O CSV tem10 épocas; o notebook publicado configura5 e só mostra parte da
primeira. Não foi presumido que os pesos enviados sejam da melhor época
por acurácia ou por perda deste CSV, nem que toda a configuração seja idêntica.
Essa associação depende de confirmação da autora. Nenhuma métrica de teste final
foi incluída. Não foram executados treino ou inferência em imagens reais.
''')
    p=DESTINO/'README.md'
    if p.exists():
        texto=p.read_text()
        antigo='''O notebook publicado ainda registra somente parte da primeira época,
em CPU. Não há resultado de validação de época completa ou histórico final
nesse notebook. Os pesos recebidos separadamente foram conferidos e são
compatíveis com esta arquitetura; época e critério de seleção não constam
do state_dict. Nenhum treinamento foi iniciado nesta organização.'''
        texto=texto.replace(antigo,'''Recebido separadamente um histórico de10 épocas completas, agora disponível
na seção de evolução. O notebook publicado ainda configura5 épocas;
a associação entre esse histórico, sua configuração e o checkpoint enviado
precisa ser confirmada pela autora. Nenhum treino foi iniciado aqui.''')
        escrever(p,texto)


def consolidar():
    if (REPO/'resultados_para_competicao/configuracao.json').exists():
        from lisandra_publicada import consolidar_publicado
        return consolidar_publicado()
    from organizar import diagrama
    p,n=ler();DESTINO.mkdir(exist_ok=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    js(DESTINO/'parametros.json',dict(modelo='AlexNetBinary vigente',commit=commit,
        fonte='projeto/CatsandDogs_classification.ipynb',sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        canais=[96,256,384,384,256],cabeca=[9216,4096,4096,1],dropout=[.5,.5],
        pre_treino='nenhum carregamento no código',entrada=[3,224,224],
        otimizador='Adam',learning_rate=.0001,batch_size=50,epocas=5,perda='BCELoss',
        ativacao_final='Sigmoid',rotulos='pasta pai Dog→1;outras→0',
        split='80/10/10; shuffle; sem random_state/stratify explícitos',
        normalizacao='ToTensor[0,1]; sem Normalize',augmentation='não definido'))
    escrever(DESTINO/'README.md','''# Lisandra — AlexNetBinary vigente

Referência atual: notebook do [repositório da Lisandra](https://github.com/lisandra21/cats_vs_dogs_Lisandra), branch main.

- [Notebook original do GitHub](../projeto/CatsandDogs_classification.ipynb).
- [Método e parâmetros](METODO.md), [JSON](parametros.json).
- [Registro do modelo](REGISTRO_MODELO.md).
- [Arquitetura no formato do livro SVG](arquitetura_livro.svg), [PNG](arquitetura_livro.png), [PDF](arquitetura_livro.pdf).
- [Diagrama sequencial SVG](arquitetura.svg).
- [Treino e validação: disponibilidade e progresso](evolucao%20do%20treinamento/README.md).
- [Banco de imagens](BANCO_DE_DADOS.md).
- [Amostra de rótulos publicada pela autora](amostra_rotulos.png).
- [Saídas originais](saidas_notebook.txt).
- [Pesos enviados pela autora](checkpoints/alexnet_cats_dogs_best.pth).
- [Conferência e identificação do checkpoint](checkpoints/checkpoint_info.json).

Rótulo atualmente definido pela pasta pai imediata; a busca no caminho
completo não está presente nesta versão. Arquitetura e hiperparâmetros
abaixo referem-se apenas ao código vigente.

O notebook publicado ainda registra somente parte da primeira época,
em CPU. Não há resultado de validação de época completa ou histórico final
nesse notebook. Os pesos recebidos separadamente foram conferidos e são
compatíveis com esta arquitetura; época e critério de seleção não constam
do state_dict. Nenhum treinamento foi iniciado nesta organização.
''')
    escrever(DESTINO/'METODO.md','''# Método da versão vigente

CNN AlexNetBinary do zero: Conv11×11(3→96,s4,p2)+ReLU+Pool3/s2;
Conv5×5(96→256,p2)+ReLU+Pool3/s2; três Conv3×3,p1 com384/384/256
canais e ReLU; último Pool3/s2. Flatten9216 → Linear4096 → ReLU →
Dropout50% → Linear4096 → ReLU → Dropout50% → Linear1 → Sigmoid.
A camada Linear1000 não existe nesta arquitetura.

Adam LR0.0001, BCELoss,5 épocas previstas,batch50. Predição pelo limiar0.5.
Sem scheduler ou carregamento de pesos pré-treinados explícito.
Código escolhe CUDA se disponível; a saída publicada informa CPU.
RGB224×224, pixels[0,1], split80/10/10 aleatório; detalhes em BANCO_DE_DADOS.md.
Não há salvamento de checkpoint ou histórico por época implementado no notebook.

O comentário do DataLoader ainda diz500, mas os argumentos de código e a
saída de400/50/50 lotes correspondem a batch50. Apenas esse valor efetivo
foi documentado; não foi alterado o código da autora.
''')
    escrever(DESTINO/'REGISTRO_MODELO.md',f'''# Registro documental — Lisandra

Atualizado em06/11/2026, commit `{commit}` do repositório da autora.
Conferidos código e saídas salvas, sem executar células de instalação,
download ou treinamento. Referência atual única: CatsandDogs_classification.ipynb.

Rótulos pela pasta pai; Adam0.0001, batch50,5 épocas previstas; cabeça
4096→4096→1 com dois dropouts50%. Métodos, diagramas e informações dos dados
atualizados. Resultados de épocas completas ainda não publicados nesta revisão.
''')
    for c in n['cells']:
        if 'num_amostras = 10' not in ''.join(c.get('source',[])):continue
        for o in c.get('outputs',[]):
            imagem=o.get('data',{}).get('image/png')
            if imagem:
                if isinstance(imagem,list):imagem=''.join(imagem)
                (DESTINO/'amostra_rotulos.png').write_bytes(base64.b64decode(imagem))
    diagrama(DESTINO/'arquitetura.svg','Lisandra — AlexNetBinary vigente',[
        'RGB224×224','Conv11×11:96,s4,p2 → ReLU → Pool3/s2','Conv5×5:256,p2 → ReLU → Pool3/s2',
        'Conv3×3:384 → ReLU','Conv3×3:384 → ReLU','Conv3×3:256 → ReLU → Pool3/s2',
        'Flatten9216 → FC4096 → ReLU → Dropout50%','FC4096 → ReLU → Dropout50%',
        'FC1 → Sigmoid'], 'Fonte: notebook GitHub vigente, célula5.')
    banco();evolucao()
    escrever(RAIZ/'Lisandra/README.md','# Lisandra\n\n[Modelo principal vigente](modelo%20principal/README.md). [Repositório original](projeto/).\n')


if __name__=='__main__':consolidar()
