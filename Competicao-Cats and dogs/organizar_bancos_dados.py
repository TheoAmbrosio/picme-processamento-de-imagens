"""Documenta bancos a partir de código, metadados e saídas; não baixa imagens."""
from pathlib import Path
import collections
import csv
import json
import os

RAIZ=Path(__file__).resolve().parent
THEO=RAIZ.parent/'Cats and Dogs I e II'


def escrever(p,texto):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(texto,encoding='utf-8')


def salvar_json(p,obj):escrever(p,json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def link(p,alvo,texto):return f'[{texto}](<{os.path.relpath(alvo,p.parent)}>)'


def pasta(nome):
    p=RAIZ/nome/'modelo principal/banco de dados'
    p.mkdir(parents=True,exist_ok=True)
    return p


def indice(nome,principal,entradas):
    p=RAIZ/nome/'modelo principal/BANCO_DE_DADOS.md'
    escrever(p,f'# Banco de dados — {nome}\n\nDocumentado em06/11/2026 a partir das fontes locais, sem executar treinamento.\n\nModelo principal: **{principal}**.\n\n'+''.join(f'- {link(p,alvo,titulo)}.\n' for titulo,alvo in entradas)+'\nQuantidades salvas em saídas são observações da execução; contagens esperadas e configurações atuais são identificadas separadamente. Nenhum banco foi unificado ou redistribuído.\n')
    readme=p.parent/'README.md';s=readme.read_text()
    trecho='\n## Banco de dados do treinamento\n\n[Origem, quantidades, divisão e transformações por modelo](BANCO_DE_DADOS.md).\n'
    if '](BANCO_DE_DADOS.md)' not in s:escrever(readme,s+trecho)


def theo():
    p=pasta('Theo');alvo=p/'Aug8.md'
    config=THEO/'dataset_ampliado/224x224/divisao_balanceada/config_dataset.json'
    origem=THEO/'dataset_ampliado/224x224/divisao_balanceada/manifesto_originais.csv'
    contagens=collections.Counter()
    with origem.open() as f:
        for linha in csv.DictReader(f):
            if linha.get('status')=='ok':contagens[(linha['classe'],linha['split'])]+=1
    salvar_json(p/'divisao_manifesto.json',{'config_dataset':json.loads(config.read_text()),
        'contagens_validas':{f'{c}/{s}':v for (c,s),v in sorted(contagens.items())},
        'fonte':str(origem),'tipo':'contagem de entradas status=ok do manifesto preservado'})
    escrever(alvo,'''# Theo — banco do Aug8 principal

## Origem e disponibilidade

- Kaggle: `shaunthesheep/microsoft-catsvsdogs-dataset`, conforme preparador.
- Originais em `Cats and Dogs I e II/dataset/PetImages/{Cat,Dog}`.
- Fonte informa12500 arquivos por classe. Relatório da preparação:12499
  válidos por classe, um ilegível em cada;24998 válidos no total.
- Dados não foram copiados para a pasta da competição nem baixados novamente.

## Divisão antes de ampliar

Divisão aleatória balanceada80/20, semente42. Por classe:9999 treino,
2500 validação,0 teste. Total:19998 originais de treino e5000 de validação.
SHA256 agrupa conteúdo idêntico para manter derivados no mesmo lado;
duplicatas não são todas removidas. Rótulos conflitantes aceitos conforme
a configuração histórica. Cat=0, Dog=1. Teste final separado não executado.

## Entrada do modelo e avaliação

RGB224×224. Normalização ImageNet: média[0.485,0.456,0.406],
desvio[0.229,0.224,0.225]. Validação usa os JPEGs históricos preparados
por Resize256 bilinear + CenterCrop224; não recebe novas corrupções.

## Oito amostras virtuais por original de treino

1. Original esticada para224×224.
2. Ruído gaussiano, variância0.08, média0;2% dos pixels brancos.
3. Oito quadrados pretos, lados20–40px; centros na região central150×150.
4. Esticamento horizontal2× seguido de recorte central224.
5. Esticamento vertical2× seguido de recorte central224.
6. Ruído + quadrados.
7. Esticamento horizontal + ruído + quadrados.
8. Esticamento vertical + ruído + quadrados.

Total159984 amostras virtuais de treino por época. Sorteios novos na leitura;
isso não significa159984 arquivos físicos independentes. Variantes usam
o split do original; a validação não é multiplicada por8.

## Diferença para as etapas anteriores

Referência, repetição e primeiro fine-tuning usaram16 variantes físicas
por original:2 geometrias(recorte/esticada) ×2 espelhamentos ×4 rotações
(0/90/180/270). São319968 amostras de treino, com os mesmos5000 exemplos
de validação. A ampliação8× pertence ao Aug8, não a essas fases anteriores.
'''+f'\n## Fontes\n\n- {link(alvo,config,"Configuração da divisão")}.\n- {link(alvo,origem,"Manifesto dos originais")}.\n- {link(alvo,config.parent/"relatorio.txt","Relatório da preparação")}.\n- {link(alvo,THEO/"residuals networks/res_theo34_aug8/AMPLIACAO.md","Especificação Aug8")}.\n')
    grupos=collections.defaultdict(list)
    chaves=('usar_dados_ampliados','ampliacao_esquema','esquema_ampliacao',
        'epocas_iniciais_sem_ampliacao','frac_validacao','frac_teste','divisao',
        'aceitar_rotulos_duplicados','semente','manifesto','ruido_variancia',
        'ruido_gaussiano_variancia','pixels_brancos_fracao','quadrados_quantidade',
        'quadrado_min','quadrado_max','regiao_centros','fator_esticamento',
        'cutout_retangulos','cutout_lado_minimo','cutout_lado_maximo')
    for arquivo in sorted((THEO/'resultados_residuals_networks').rglob('parametros.json')):
        d=json.loads(arquivo.read_text());c=d.get('config',{});nome=d.get('modelo',arquivo.parent.parent.name)
        grupos[nome].append(dict(execucao=arquivo.parent.name,fonte=str(arquivo),
            estado=d.get('estado'),dados={k:c[k] for k in chaves if k in c},
            dataset_misto=d.get('dataset_misto')))
    salvar_json(p/'inventario_todos_modelos_theo.json',grupos)
    entradas=[('Aug8 e sua base de treinamento',alvo)]
    for nome,execucoes in sorted(grupos.items()):
        d=p/'modelos'/f'{nome}.md'
        texto=f'# Theo — dados de {nome}\n\nInventário documental, incluindo execuções arquivadas. Não promove esses modelos a principais. Valores ausentes não foram preenchidos com a configuração atual.\n'
        for e in execucoes:
            texto+=f'\n## {e["execucao"]}\n\n{link(d,Path(e["fonte"]),"Metadados originais")}\n\n```json\n{json.dumps(e["dados"],ensure_ascii=False,indent=2)}\n```\n'
            if e['dataset_misto']:texto+='\nConfiguração adicional da mistura:\n```json\n'+json.dumps(e['dataset_misto'],ensure_ascii=False,indent=2)+'\n```\n'
        escrever(d,texto);entradas.append((nome,d))
    indice('Theo','ResTheo34 Aug8',entradas)


def dulce():
    p=pasta('Dulce');repo=RAIZ/'Dulce/projeto/Cats vs Dogs'
    comum='''## Banco original e limpeza

Kaggle `bhavikjikadara/dog-and-cat-classification-dataset` via kagglehub.
Saídas salvas:12499 arquivos em Cat e12499 em Dog. Após remoção de
repetidos por hash:24966 imagens,24962 origens perceptuais.
Grupos com rótulos conflitantes são excluídos; grupos idênticos de mesma
classe ficam com um arquivo. Impressão visual9×8 em cinza agrupa imagens
com mesmo padrão de comparações vizinhas. Isso é o algoritmo documentado,
não prova que toda duplicata visual possível foi detectada.
Excluídos lados menores que94px:24822 originais restantes.
Classes salvas:['Cat','Dog']; Cat=0, Dog=1. Semente18.

## Divisão efetivamente implementada

O código separa15% para teste e depois15% **do restante** para validação.
Proporções teóricas72,25%/12,75%/15%, com arredondamentos e grupos inteiros;
não70%/15%/15%. Os números abaixo são os impressos nas saídas originais.
Não foi encontrado manifesto completo dos arquivos de cada split no clone.
'''
    entradas=[]
    for nome in ('cats_and_dogs','cats_and_dogs_224','cats_and_dogs_224_v2','cats_and_dogs_resnet'):
        nb=repo/(nome+'.ipynb');n=json.loads(nb.read_text());alvo=p/(nome+'.md')
        evidencias=[]
        for i,c in enumerate(n['cells']):
            s=''.join(c.get('source',[]))
            if c['cell_type']=='code' and any(k in s for k in ('separar_por_grupo','LARGURA, ALTURA','def aumentar','def adicionar_ruido','vao_girar','MEDIA =','dataset_download')):
                evidencias.append(dict(celula=i,codigo=s,saidas=[''.join(o.get('text',[])) for o in c.get('outputs',[]) if o.get('text')]))
        salvar_json(p/'fontes'/(nome+'.json'),evidencias)
        texto=f'# Dulce — {nome}\n\n'+comum
        if nome=='cats_and_dogs':
            texto+='''
## Versão inicial100×94

Redimensionamento bilinear para largura100, altura94, determinado pela menor
área restante. Pixels[0,1], tensor N×3×94×100. Antes de dividir,10% dos
originais recebem uma cópia girada15° e outros10% uma com ruído gaussiano
(desvio2 em escala0–255). Total29786 versões. Divisão por origem após essas
cópias:21520 treino,3798 validação,4468 teste. Portanto nesta versão
validação/teste também podem conter derivados da mesma família de transformações,
mantidos no mesmo grupo do original. Não atribuir a ela a estratégia posterior
de ampliar somente o treino.
'''
        else:
            texto+='''
## Originais separados antes de ampliar

17934 treino original;3165 validação;3723 teste. RGB224×224 com bilinear,
preservação da proporção e corte central.2785 originais exigem ampliação
porque o lado menor é menor que224. Validação/teste somente originais.
'''
            if nome=='cats_and_dogs_resnet':
                texto+='''
Sem cópias fixas de rotação/ruído neste notebook:17934 imagens de treino.
Augmentation durante os lotes: flip horizontal50%, zoom0.7–1, rotação±15°,
deslocamento dependente do zoom, brilho/contraste0.8–1.2, interpolação bilinear
com padding por reflexão. Entrada[0,1] e depois normalização ImageNet dentro
do modelo: média0.485/0.456/0.406, desvio0.229/0.224/0.225.
'''
            else:
                texto+='''
Cópias fixas só do treino:1793 giradas15° e1793 com ruído gaussiano
(média0, desvio2 na escala0–255);21520 amostras de treino e28408 no tensor
global, contando validação/teste. Tensor uint8, convertido para[0,1] no lote,
sem Normalize por canal. Há augmentation online: flip50%, zoom0.7–1,
rotação±15°, deslocamento, brilho/contraste0.8–1.2. No notebook224 anterior
existem loops de treino sucessivos; augmentation online só se aplica aos
loops que chamam aumentar(), não automaticamente a todas as saídas antigas.
'''
        texto+=f'\n## Fontes\n\n- {link(alvo,nb,"Notebook original")}.\n- {link(alvo,p/"fontes"/(nome+".json"),"Células e saídas usadas como evidência")}.\n\nImagens e splits não foram baixados nem recriados nesta organização.\n'
        escrever(alvo,texto);entradas.append((nome,alvo))
    indice('Dulce','CNN224 v2',entradas)


def bandeira():
    p=pasta('Bandeira');repo=RAIZ/'Bandeira/projeto'
    entradas=[]
    for v in range(1,5):
        alvo=p/f'V{v}.md'
        texto=f'''# Bandeira — banco V{v}

## Origem, rótulos e divisão

README declara Kaggle Dogs vs Cats:25000 imagens rotuladas de treino e12500
imagens de teste. Código procura `data/dogs-vs-cats/train/*.jpg` e
`test1/`; data/data-setting.ipynb baixa a competição Kaggle `dogs-vs-cats`
via kagglehub.competition_download.
Essas imagens não estão no clone recebido. Não confundir essa fonte com
os espelhos PetImages usados pelos demais.

Cat=0 para nome começando `cat.`; demais nomes recebem1.
Divisão80/20 estratificada, random_state42, conforme código disponível e README.
Se o diretório contiver25000 imagens, espera-se20000 treino e5000 validação;
isso é uma contagem esperada, não uma leitura de arquivos locais.
A análise V{v} salva mostra2500 gatos e2500 cachorros na validação.
O diretório test1 é separado e pode não ter rótulos; não foi avaliado aqui.
Há análise de um conjunto externo Pets IMPATech de64 imagens; é outro banco,
não parte confirmada do treinamento. Nenhuma imagem desse conjunto foi usada aqui.

## Transformações por versão

Há preparação offline em data/data-setting.ipynb: converte RGB, redimensiona
mantendo proporção (lado menor igual ao alvo), recorta o centro e sobrescreve
os JPEGs. Célula7 usa224; célula8 usa384 para V3. Isso explica a ausência
de Resize em alguns loaders que já recebem arquivos preparados. Não há
manifesto dos arquivos preparados por execução para confirmar a sequência
exata realizada em cada máquina.

'''
        if v==1:
            texto+='''README histórico: entrada224, normalização ImageNet
(mean0.485/0.456/0.406,std0.229/0.224/0.225), sem augmentation declarado.
O código V1 completo de preparação não está preservado como configuração
imutável no estado atual; detalhes de resize/crop e limpeza não confirmados.
'''
        elif v==2:
            texto+='''README histórico: entrada224, RandomResizedCrop224,
flip horizontal, rotação±15°, ColorJitter0.2. Normalização própria em
data/norm_stats.json: média0.49072049/0.45525817/0.41692111,
desvio0.25939278/0.25240866/0.25409759. Não extrapolado o código V4
de validação para essa versão antiga sem fonte da execução.
'''
        else:
            texto+='''Entrada384×384. Treino: RandomResizedCrop384(scale0.8–1),
flip horizontal, RandomRotation15, brilho/contraste0.2, ToTensor, Normalize.
Mean em norm_stats_3.0.json:0.49077199/0.45522934/0.41697363;
std0.26066301/0.25364598/0.25537652. Estatísticas documentadas como do treino.
Não há multiplicação fixa em arquivos: transformações são sorteadas na leitura.
'''
            if v==4: texto+='\nCódigo vigente V4: validação Resize((384,384)) estático, ToTensor e mesma normalização.\n'
            else: texto+='\nO src_old não redimensiona no loader de validação; a preparação offline384 em data-setting.ipynb fornece essa dimensão. Não se deve concluir que isso seja erro de execução apenas pela ausência de Resize no loader.\n'
        texto+='''
## Disponibilidade e pendências

Históricos training_history, predições val_predictions e checkpoints V1–V4
agora publicados no clone atualizado. Sem imagens ou manifestos completos
associando nomes de arquivos aos splits; predições contêm rótulos/probabilidades.
Não encontrado tratamento explícito de duplicatas/agrupamento por conteúdo na
rotina de divisão disponível. Isso descreve o código lido, não atesta a existência
de vazamento. Não foram contadas classes de treino nem refeita a divisão.
'''
        texto+=f'\n## Fontes\n\n- {link(alvo,repo/"README.md","README histórico")}.\n- {link(alvo,repo/"Analysis"/f"resultsV{v}.ipynb","Análise salva da versão")}.\n- {link(alvo,repo/"src/dataset.py","Código de dados vigente V4")}.\n- {link(alvo,repo/"src_old/dataset.py","Código anterior disponível")}.\n'
        escrever(alvo,texto);entradas.append((f'CNN V{v}',alvo))
    salvar_json(p/'normalizacao_publicada.json',{p.name:json.loads(p.read_text()) for p in (repo/'data').glob('norm_stats*.json')})
    indice('Bandeira','CNN V4',entradas)


def lisandra():
    from lisandra_atualizada import banco
    banco()


def main():
    theo();dulce();bandeira();lisandra()
    escrever(RAIZ/'BANCOS_DE_DADOS.md', '''# Bancos de dados da competição

Documentação baseada em código, metadados e saídas salvas em06/11/2026.
Não foram baixadas imagens, executados notebooks ou refeitos splits.

| Participante | Modelos/bancos documentados | Índice |
|---|---|---|
| Theo | Aug8 detalhado e inventário por modelo/execução, inclusive arquivados | [Abrir](Theo/modelo%20principal/BANCO_DE_DADOS.md) |
| Dulce | versão inicial,224,224_v2,ResNet18 | [Abrir](Dulce/modelo%20principal/BANCO_DE_DADOS.md) |
| Bandeira | V1,V2,V3,V4 com origem de cada informação | [Abrir](Bandeira/modelo%20principal/BANCO_DE_DADOS.md) |
| Lisandra | AlexNetBinary vigente no GitHub | [Abrir](Lisandra/modelo%20principal/BANCO_DE_DADOS.md) |

O mesmo nome informal Cats and Dogs não significa o mesmo split. Foram
preservadas diferenças entre fonte Kaggle, limpeza, contagens, entrada,
normalização e augmentation. Contagem de variantes não equivale à contagem
de animais originais independentes. Fontes ausentes estão indicadas nas fichas.
''')
    print('Bancos documentados para os quatro participantes e variantes encontradas.')


if __name__=='__main__':main()
