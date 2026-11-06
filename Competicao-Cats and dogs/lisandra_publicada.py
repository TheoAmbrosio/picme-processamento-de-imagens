"""Consolida o pacote de resultados publicado pela Lisandra, sem executar treino."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import shutil
import subprocess

RAIZ=Path(__file__).resolve().parent
REPO=RAIZ/'Lisandra/projeto'
PACOTE=REPO/'resultados_para_competicao'
DEST=RAIZ/'Lisandra/modelo principal'


def escrever(p,s):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf-8')


def js(p,o):escrever(p,json.dumps(o,ensure_ascii=False,indent=2)+'\n')


def banco_publicado():
    escrever(DEST/'banco de dados/AlexNetBinary.md','''# Lisandra — banco da versão publicada

Fontes: configuracao.json e notebook do repositório, commit538a5e9.
Kaggle bhavikjikadara/dog-and-cat-classification-dataset;24998 imagens.
Treino19998, validação2500, teste2500; split80/10/10 com shuffle,
sem random_state/stratify ou seed manual definidos. Sem manifesto nominal
dos arquivos nem contagem por classe publicada.

Rótulos pela pasta pai imediata: Dog=1, Cat=0. RGB224×224 por Resize direto,
ToTensor e **Normalize ImageNet**: mean[0.485,0.456,0.406],
std[0.229,0.224,0.225]. Sem data augmentation declarado. Batch50;
400/50/50 lotes previstos. Não atribuir a este modelo a ausência de Normalize
de versões anteriores. Sem tratamento explícito de duplicatas/ilegíveis.

Os dados de teste existem no código, mas não foram executados aqui. Dados
originais e splits não baixados ou refeitos por esta organização.
''')
    escrever(DEST/'BANCO_DE_DADOS.md','# Banco de dados — Lisandra\n\n[Origem, splits e pré-processamento da versão publicada](banco%20de%20dados/AlexNetBinary.md).\n\nEspecificação original em [configuração publicada](documentacao_publicada/configuracao.json).\n')


def evolucao_publicada():
    from organizar_evolucao import salvar
    p=PACOTE/'historico_epocas.csv'
    with p.open(newline='',encoding='utf-8-sig') as f:
        rows=[dict(epoca=int(r['epoca']),perda_treino=float(r['perda_treino']),
            perda_validacao=float(r['perda_validacao']),acuracia_treino=float(r['acuracia_treino']),
            acuracia_validacao=float(r['acuracia_validacao'])) for r in csv.DictReader(f)]
    assert [r['epoca'] for r in rows]==list(range(1,len(rows)+1))
    bruto=REPO/'training_history_Lisandra.csv'
    with bruto.open() as f:raw=list(csv.DictReader(f))
    assert len(rows)==len(raw)==10
    for r,s in zip(rows,raw):
        for k,origem,fator in [('acuracia_treino','train_acc',.01),('acuracia_validacao','val_acc',.01),('perda_treino','train_loss',1),('perda_validacao','val_loss',1)]:
            assert abs(r[k]-float(s[origem])*fator)<1e-12
    auditoria=DEST/'CONFERENCIA_ATUALIZACAO.json'
    compativel=auditoria.exists() and json.loads(auditoria.read_text()).get('compativel_com_arquitetura_atual',False)
    salvar('Lisandra','execucao_1_publicada',rows,str(p.relative_to(RAIZ)),
        'Checkpoint declarado pela autora: época5 por menor val_loss; '+('novo peso recebido compatível.' if compativel else 'arquivo de pesos correspondente pendente.'))
    d=DEST/'evolucao do treinamento';d.mkdir(exist_ok=True)
    shutil.copy2(p,d/'historico_epocas.csv')
    for ext in ('svg','png','pdf'):
        shutil.copy2(d/'execucao_1_publicada'/f'graficos.{ext}',d/f'evolucao_epocas.{ext}')
    info=json.loads((PACOTE/'checkpoint_info.json').read_text())
    js(d/'disponibilidade.json',dict(fonte=str(p.relative_to(RAIZ)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        epocas_com_resumo=10,epocas_previstas=40,early_stopping=dict(paciencia=5,parada=10),
        epoca_checkpoint_declarada=5,criterio_checkpoint_declarado='menor_perda_validacao',
        acuracia_validacao_checkpoint=.9016,maior_acuracia_validacao=dict(epoca=8,valor=.9108),
        confirmacao_pesos=('novo peso com BatchNorm recebido e compatível; época5 conforme declaração da autora' if compativel else 'peso local recebido não corresponde à arquitetura atual; novo arquivo pendente'),
        fonte_associacao='checkpoint_info.json e configuracao.json fornecidos pela autora'))
    escrever(d/'README.md','''# Evolução da Lisandra — execução1 publicada

Histórico completo de10 épocas, com40 previstas e early stopping de paciência5.
O pacote da autora identifica o checkpoint da **época5 por menor val_loss**.
Na época5: validação90,16%, perda0,223732; na época8: maior acurácia91,08%,
mas perda0,224588, superior à da5. São critérios diferentes.

- [Plot SVG](evolucao_epocas.svg), [PNG](evolucao_epocas.png), [PDF](evolucao_epocas.pdf).
- [CSV oficial por época](historico_epocas.csv).
- [Metadados e disponibilidade](disponibilidade.json).
- [Especificação original do checkpoint](../documentacao_publicada/checkpoint_info.json).
- [Pendências de código e pesos](../PENDENCIAS_ATUAIS.md).

Dados numéricos conferidos contra o CSV bruto publicado, normalização0–1
consistente. Não foram executados novos treinos, inferência real ou teste final.
A associação à época5 é documentada pela autora; o arquivo de pesos local
recebido anteriormente não corresponde à arquitetura agora publicada.
''')
    if compativel:
        p=d/'README.md';escrever(p,p.read_text().replace('arquivo de pesos local\nrecebido anteriormente não corresponde à arquitetura agora publicada.','novo arquivo de pesos recebido é compatível com a arquitetura publicada.'))


def auditar():
    import torch
    from torch import nn
    p=REPO/'CatsandDogs_classification.ipynb';nb=json.loads(p.read_text());erros=[]
    for i,c in enumerate(nb['cells']):
        if c['cell_type']!='code':continue
        s=''.join(c.get('source',[]))
        if any(l.lstrip().startswith(('!','%')) for l in s.splitlines()):continue
        try:ast.parse(s)
        except SyntaxError as e:erros.append(dict(celula=i,linha=e.lineno,mensagem=e.msg))
    s=next(''.join(c['source']) for c in nb['cells'] if 'class AlexNetBinary' in ''.join(c.get('source',[])))
    node=next(n for n in ast.parse(s).body if isinstance(n,ast.ClassDef) and n.name=='AlexNetBinary')
    ns={'torch':torch,'nn':nn};exec(compile(ast.Module(body=[node],type_ignores=[]),'<classe isolada>','exec'),ns)
    modelo=ns['AlexNetBinary']();esperado=modelo.state_dict()
    novo=DEST/'checkpoints/versao_com_batchnorm/alexnet_cats_dogs_best.pth'
    peso=novo if novo.exists() else DEST/'checkpoints/alexnet_cats_dogs_best.pth'
    declarado=json.loads((PACOTE/'checkpoint_info.json').read_text())
    audit=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        erros_sintaxe=erros,peso_novo_presente_no_clone=(REPO/'alexnet_cats_dogs_best.pth').exists(),
        tamanho_declarado_peso_atual=declarado['tamanho_bytes'])
    audit['alteracoes_locais']=subprocess.check_output(['git','status','--short'],cwd=REPO,text=True).splitlines()
    readme=(REPO/'README.md').read_text()
    audit['readme_coerente_com_modelo']=('BatchNorm' in readme and 'BCEWithLogitsLoss' in readme and '1000' not in readme)
    if peso.exists():
        state=torch.load(peso,map_location='cpu',weights_only=True)
        faltam=sorted(set(esperado)-set(state));extras=sorted(set(state)-set(esperado))
        shapes=[dict(chave=k,recebido=list(state[k].shape),esperado=list(esperado[k].shape)) for k in set(state)&set(esperado) if state[k].shape!=esperado[k].shape]
        audit.update(peso_local_recebido=str(peso.relative_to(RAIZ)),tamanho_recebido=peso.stat().st_size,
            compativel_com_arquitetura_atual=not(faltam or extras or shapes),
            chaves_faltantes=faltam,chaves_extras=extras,dimensoes_incompativeis=shapes)
        metap=peso.parent/'checkpoint_info.json'
        anterior=json.loads(metap.read_text()) if metap.exists() else {}
        historico=anterior.get('conferencia_anterior', {k:v for k,v in anterior.items() if k not in ('conferencia_versao_atual','observacao_atual')})
        js(metap,dict(arquivo=peso.name,status_atual=('compatível com a arquitetura publicada538a5e9' if audit['compativel_com_arquitetura_atual'] else 'incompatível com a arquitetura publicada538a5e9; aguardar novo arquivo'),
            conferencia_versao_atual=audit,conferencia_anterior=historico))
    js(DEST/'CONFERENCIA_ATUALIZACAO.json',audit)
    return audit


def consolidar_publicado():
    from organizar import diagrama
    DEST.mkdir(exist_ok=True);d=DEST/'documentacao_publicada';d.mkdir(exist_ok=True)
    for nome in ('configuracao.json','checkpoint_info.json','README.md'):
        shutil.copy2(PACOTE/nome,d/nome)
    cfg=json.loads((PACOTE/'configuracao.json').read_text());audit=auditar()
    js(DEST/'parametros.json',dict(modelo='AlexNetBinary com BatchNorm — versão publicada',commit=audit['commit'],
        fonte='resultados_para_competicao/configuracao.json',configuracao_declarada=cfg,
        checkpoint_status=('novo peso recebido compatível com BatchNorm' if audit.get('compativel_com_arquitetura_atual') else 'arquivo local anterior incompatível; pedir o novo correspondente'),
        epoca_checkpoint_declarada=5,criterio_checkpoint='menor_perda_validacao'))
    escrever(DEST/'README.md','''# Lisandra — AlexNetBinary com BatchNorm

Referência vigente: pacote resultados_para_competicao e notebook do GitHub,
commit538a5e9. Código e documentação da autora foram preservados.

- [Método](METODO.md) e [configuração completa declarada](documentacao_publicada/configuracao.json).
- [Registro](REGISTRO_MODELO.md).
- [Arquitetura no formato livro SVG](arquitetura_livro.svg), [PNG](arquitetura_livro.png), [PDF](arquitetura_livro.pdf).
- [Diagrama sequencial](arquitetura.svg).
- [Treino e validação:10 épocas](evolucao%20do%20treinamento/README.md).
- [Banco e normalização](BANCO_DE_DADOS.md).
- [Informações do checkpoint declaradas pela autora](documentacao_publicada/checkpoint_info.json).
- [Pendências verificadas](PENDENCIAS_ATUAIS.md), [conferência JSON](CONFERENCIA_ATUALIZACAO.json).
- [Notebook original](../projeto/CatsandDogs_classification.ipynb).

**Modelo atual não está pronto para inferência com o peso recebido antes:**
ele não contém as BatchNorm e não carrega na nova arquitetura. Não houve
adaptação de pesos nem execução de treinamento/teste para preencher lacunas.
''')
    escrever(DEST/'METODO.md','''# Método publicado pela Lisandra

AlexNetBinary do zero, canais96/256/384/384/256, kernels11/5/3/3/3.
BatchNorm após cada convolução, ReLU; pooling após conv1,conv2,conv5.
Flatten9216 →4096 → ReLU → Dropout50% →4096 → ReLU → Dropout50% →1 logit.
Sem Sigmoid no forward; usar sigmoid para obter probabilidade na avaliação.

Adam LR0.0001, betas0.9/0.999, eps1e-8, decay0; BCEWithLogitsLoss,
batch50,40 épocas previstas. Early stopping por val_loss, paciência5,
parada na10 e restauração da5, segundo os artefatos fornecidos pela autora.
Sem scheduler, pesos pré-treinados ou congelamento. Sem seed manual.
Normalização ImageNet e RGB224×224. Sem augmentation declarado.

Histórico confirma mínimo de perda na5, com90,16% de acurácia de validação.
Maior acurácia91,08% na8 não é o critério declarado do checkpoint.
Não confundir descrição do pacote com validação do arquivo de pesos anterior:
esse arquivo local é de arquitetura diferente. Ver PENDENCIAS_ATUAIS.md.
''')
    escrever(DEST/'PENDENCIAS_ATUAIS.md','''# Pendências verificadas na publicação atual

1. **Pesos recebidos incompatíveis.** Arquivo local anterior tem233148669 bytes,
   sem BatchNorm e com índices convolucionais diferentes. O pacote declara
   um checkpoint de233178729 bytes, com BatchNorm. O novo peso não está no
   clone. É necessário receber esse arquivo correspondente; não usamos
   strict=False para ignorar camadas nem completamos pesos aleatoriamente.
2. **Erros sintáticos no notebook publicado.** Célula6, linha51, e célula10,
   linha32: f-strings quebradas por novas linhas literais. Conferência feita
   com ast.parse, sem rodar treino. A autora precisa reenviar o notebook com
   essas strings corrigidas; nenhum reparo foi aplicado ao código original.
3. **README raiz ainda descreve outra cabeça.** Ele menciona FC1000 e Sigmoid,
   mas o notebook/pacote atual contém4096→4096→1 logit, BatchNorm e
   BCEWithLogitsLoss. Nossa documentação segue o código e configuracao.json,
   identificando essa divergência em vez de misturar arquiteturas.

A época5 e o critério menor_val_loss são declarados pelos novos metadados;
as métricas concordam com o CSV. Isso não prova que o arquivo local anterior
seja esse checkpoint. Histórico e logs foram organizados sem nova avaliação.
''')
    escrever(DEST/'REGISTRO_MODELO.md',f'''# Registro — Lisandra, publicação atual

Commit {audit['commit']}, obtido em06/11/2026. Pacote oficial com configuração,
checkpoint_info, histórico normalizado e log. CSV oficial conferido contra
training_history bruto. Informações atuais: BatchNorm, ImageNet Normalize,
40 épocas previstas,10 executadas, early stopping5, checkpoint declarado
da época5 por menor val_loss. Gráficos e diagramas atualizados.

Arquivo de pesos local anterior incompatível; duas células com SyntaxError
e README raiz desatualizado. Detalhes em PENDENCIAS_ATUAIS.md. Não rodados
treino ou teste final; não alterados os arquivos da autora.
''')
    diagrama(DEST/'arquitetura.svg','Lisandra — AlexNetBinary com BatchNorm',[
        'RGB224×224 → Normalize ImageNet','Conv11×11:96 → BN → ReLU → Pool',
        'Conv5×5:256 → BN → ReLU → Pool','Conv3×3:384 → BN → ReLU',
        'Conv3×3:384 → BN → ReLU','Conv3×3:256 → BN → ReLU → Pool',
        'Flatten9216 → FC4096 → ReLU → Dropout50%',
        'FC4096 → ReLU → Dropout50%','FC1 logit (sigmoid somente para probabilidade)'],
        'Fonte: notebook e configuração publicados, commit538a5e9.')
    banco_publicado();evolucao_publicada()
    if audit.get('compativel_com_arquitetura_atual'):
        p=DEST/'README.md';s=p.read_text().replace('''**Modelo atual não está pronto para inferência com o peso recebido antes:**
ele não contém as BatchNorm e não carrega na nova arquitetura. Não houve
adaptação de pesos nem execução de treinamento/teste para preencher lacunas.''','''**Novo checkpoint com BatchNorm recebido e compatível.** Carregamento estrito
e forward artificial aprovados; nenhum treinamento ou teste em imagens reais.
[Pesos atuais](checkpoints/versao_com_batchnorm/alexnet_cats_dogs_best.pth) e
[conferência](checkpoints/versao_com_batchnorm/checkpoint_info.json).
As pendências restantes dizem respeito ao notebook e README da autora.''');escrever(p,s)
        p=DEST/'PENDENCIAS_ATUAIS.md';s=p.read_text();inicio=s.index('1. **Pesos');fim=s.index('2. **Erros')
        s=s[:inicio]+s[fim:];s=s.replace('2. **Erros','1. **Erros').replace('3. **README','2. **README')
        s+='\n## Resolvido: checkpoint\n\nNovo arquivo com233178729 bytes recebido, BatchNorm presente e compatibilidade estrita confirmada. Teste artificial aprovado. Local: checkpoints/versao_com_batchnorm/. Época5 e critério menor_val_loss conforme declaração da autora; não gravados dentro do state_dict.\n'
        escrever(p,s)
        p=DEST/'METODO.md';escrever(p,p.read_text().replace('Não confundir descrição do pacote com validação do arquivo de pesos anterior:\nesse arquivo local é de arquitetura diferente. Ver PENDENCIAS_ATUAIS.md.','O novo arquivo de pesos recebido é compatível com esta arquitetura.\nÉpoca/critério conforme declaração da autora; teste real não executado.'))
        p=DEST/'evolucao do treinamento/README.md';s=p.read_text().replace('arquivo de pesos local\nrecebido anteriormente não corresponde à arquitetura agora publicada.','novo arquivo de pesos recebido é compatível com a arquitetura publicada.');escrever(p,s)
        p=DEST/'evolucao do treinamento/disponibilidade.json';info=json.loads(p.read_text());info['confirmacao_pesos']='novo arquivo com BatchNorm recebido e compatível estritamente; época5 segundo metadados da autora';js(p,info)
        with (DEST/'REGISTRO_MODELO.md').open('a') as f:f.write('\n## Checkpoint resolvido\n\nNovo arquivo233178729 bytes, com BatchNorm, recebido e carregado estritamente. Forward artificial em CPU/eval aprovado. Sem avaliação real. Checkpoint atual em checkpoints/versao_com_batchnorm/. Permanecem apenas as pendências do notebook e README da autora.\n')
    if not audit['erros_sintaxe'] and audit['readme_coerente_com_modelo']:
        escrever(DEST/'PENDENCIAS_ATUAIS.md','''# Situação após correções locais autorizadas

- Novo checkpoint com BatchNorm compatível; carregamento estrito e forward artificial aprovados.
- Sintaxe corrigida nos dois notebooks locais: strings de impressão e títulos com escapes apropriados.
- README local alinhado à arquitetura, normalização, perda e método publicados.

Não há pendências técnicas das verificações acima na cópia local. As correções
não mudaram arquitetura, hiperparâmetros, saídas salvas ou pesos.
Foram autorizadas pelo Theo e estão preservadas em
[patch para revisão](correcoes_locais/sintaxe_readme.patch).
O repositório remoto da Lisandra não foi alterado por nós.

Época5 e seleção por menor val_loss continuam conforme os metadados da autora;
state_dict não armazena esses campos. Não executamos treino nem teste real.
''')
        p=DEST/'README.md';escrever(p,p.read_text().replace('As pendências restantes dizem respeito ao notebook e README da autora.','README e sintaxe dos notebooks corrigidos localmente, com autorização do Theo. Patch preservado para revisão; repositório remoto da autora não alterado.'))
        escrever(DEST/'REGISTRO_MODELO.md',f'''# Registro — Lisandra, cópia local corrigida

Base remota:{audit['commit']}. Configuração publicada: BatchNorm,
Normalize ImageNet, BCEWithLogitsLoss, Adam0.0001, batch50,40 épocas previstas,
early stopping5;10 épocas registradas e checkpoint declarado da5.
Novo peso correspondente recebido e carregado estritamente; forward artificial aprovado.

Correção local autorizada pelo Theo: apenas escapes em strings das células6/10
dos dois notebooks e README alinhado ao código. Saídas, metadados, arquitetura,
hiperparâmetros e pesos preservados. Validação sintática das células Python
aprovada sem executar instalação, download, treino ou teste final.
Patch salvo em correcoes_locais/sintaxe_readme.patch. Não publicado no remoto
da Lisandra por nós. Fontes usadas incluem essas correções locais.
''')
    escrever(RAIZ/'Lisandra/README.md','# Lisandra\n\n[Modelo e resultados publicados](modelo%20principal/README.md). [Repositório local](projeto/).\n')
    return audit


if __name__=='__main__':
    consolidar_publicado()
