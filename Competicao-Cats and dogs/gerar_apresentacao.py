"""SVGs de apresentação: métodos e históricos existentes, sem treinar."""
from pathlib import Path
import copy, csv, html, json
import xml.etree.ElementTree as ET

RAIZ=Path(__file__).resolve().parent
DEST=RAIZ/'coisas para apresentacao'
NOMES=('Theo','Lisandra','Dulce','Bandeira')
CORES={'Theo':'#286b78','Lisandra':'#8b5290','Dulce':'#b96238','Bandeira':'#4367a6'}
NS='http://www.w3.org/2000/svg'
ET.register_namespace('',NS)

class Tela:
    def __init__(self,titulo,grupo,sub=''):
        self.s=['<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900" role="img">',
            '<title>'+html.escape(titulo)+'</title>',
            '<defs><marker id="seta" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#657384"/></marker></defs>',
            '<rect width="1600" height="900" fill="#fcfbf8"/>']
        self.text(65,54,grupo.upper(),17,'#657384',600)
        self.text(65,112,titulo,40,'#202c39',700)
        if sub:self.text(65,153,sub,21,'#657384')
    def text(self,x,y,s,tam=24,cor='#202c39',peso=400,ancora='start'):
        self.s.append(f'<text x="{x}" y="{y}" font-family="DejaVu Sans,sans-serif" font-size="{tam}" fill="{cor}" font-weight="{peso}" text-anchor="{ancora}">{html.escape(str(s))}</text>')
    def rect(self,x,y,w,h,cor='white',borda='#e0e3e4',raio=18):
        self.s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{raio}" fill="{cor}" stroke="{borda}"/>')
    def line(self,x1,y1,x2,y2,cor='#dce1e3',esp=2,traco='',seta=False):
        attrs=(f' stroke-dasharray="{traco}"' if traco else '')+(' marker-end="url(#seta)"' if seta else '')
        self.s.append(f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{cor}" stroke-width="{esp}"{attrs}/>')
    def curve(self,pts,cor):
        pontos=' '.join(f'{x:.2f},{y:.2f}' for x,y in pts)
        self.s.append(f'<polyline fill="none" stroke="{cor}" stroke-width="3" stroke-linejoin="round" points="{pontos}"/>')
    def card(self,x,y,w,h,titulo,linhas,cor):
        self.rect(x,y,w,h);self.rect(x+23,y+27,5,38,cor,cor,2)
        self.text(x+46,y+57,titulo,29,cor,700)
        for i,l in enumerate(linhas):self.text(x+28,y+106+i*39,l,23)
    def save(self,p):
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text('\n'.join(self.s+['</svg>']),encoding='utf-8');ET.parse(p)

TREINO={
 'Theo':dict(origem='Checkpoint 33 da repetição',linhas=['AdamW · weight decay 0,01','LRs 1e−5 / 5e−5 / 1e−4','Batch 64 · 30 épocas no Aug8','Cosseno · dropout 10% antes de fc','BCEWithLogitsLoss'],
     fluxo=['Originais de treino','8 ampliações virtuais','Fine-tuning completo','Validação e checkpoint'],
     selecao=['Maior acurácia de validação','Época 29']),
 'Lisandra':dict(origem='Inicialização do zero',linhas=['Adam · LR 0,0001','Batch 50 · 40 épocas previstas','10 épocas · early stopping 5','BatchNorm · dois dropouts de 50%','BCEWithLogitsLoss'],
     fluxo=['RGB 224 × 224','Normalize ImageNet','Treino com early stopping','Restaurar melhor época'],
     selecao=['Menor perda de validação','Época 5']),
 'Dulce':dict(origem='Inicialização do zero',linhas=['Adam · LR 0,0003','Batch 32 · 120 épocas','Cópias fixas + augmentation online','Dropout 20% nos cinco blocos','Entropia cruzada'],
     fluxo=['Split por origem','Cópias de giro e ruído','Augmentation nos lotes','Validação e checkpoint'],
     selecao=['Maior acurácia de validação','Época 112']),
 'Bandeira':dict(origem='Fine-tuning do checkpoint V4 / época 30',linhas=['AdamW · LR 0,00001 · decay 0,0001','Batch 64 · 10 épocas finais','Cosseno · early stopping 4','Dropout 50% na cabeça','BCEWithLogitsLoss'],
     fluxo=['Carregar checkpoint V4','Congelar blocos 1–3','Treinar blocos 4–5 e cabeça','Validação e checkpoint'],
     selecao=['Menor perda de validação','Época 10 do fine-tuning'])}

def grade(titulo,sub,cartas,arquivo):
    t=Tela(titulo,'Geral',sub)
    for i,n in enumerate(NOMES):t.card(65+(i%2)*755,200+(i//2)*325,715,305,n,cartas[n],CORES[n])
    t.save(DEST/'Geral'/arquivo)

def geral():
    grade('Quatro modelos para gatos e cachorros','',{
     'Theo':['ResTheo34 Aug8','ResNet-34 sem layer4','64 → 128 → 256 canais','Resíduos + média global'],
     'Lisandra':['AlexNetBinary com BatchNorm','96 → 256 → 384 → 384 → 256','Cabeça 4096 → 4096 → 1','Dois dropouts de 50%'],
     'Dulce':['CNN224 v2','32 → 64 → 64 → 128 → 128','Cabeça 1024 → 32 → 2','Cinco blocos com dropout de 20%'],
     'Bandeira':['CNN V4 Final','32 → 64 → 128 → 256 → 512','Média global → 1 saída','Cinco blocos convolucionais']},'01_modelos.svg')
    grade('Como cada modelo foi treinado','',{n:[d['origem'],*d['linhas'][:4]] for n,d in TREINO.items()},'02_treinamentos.svg')
    grade('Dados e ampliações','Conjuntos de validação distintos entre os participantes',{
     'Theo':['19.998 originais de treino · 5.000 validação','8 versões por original no Aug8','Ruído, quadrados e esticamentos','RGB 224 × 224 · ImageNet Normalize'],
     'Lisandra':['19.998 treino · 2.500 validação','Sem augmentation declarado','Resize direto para 224 × 224','ImageNet Normalize'],
     'Dulce':['17.934 originais de treino · 3.165 validação','21.520 amostras com cópias fixas','Giros, ruído e augmentation online','RGB 224 × 224 · pixels em [0,1]'],
     'Bandeira':['20.000 treino esperados · 5.000 validação','Crop aleatório, flip, giro e cor','RGB 384 × 384','Normalização própria do treino']},'03_dados_e_ampliacoes.svg')
    t=Tela('Ensaios e versões dos projetos','Geral','Execuções exibidas; não uma sequência única de pesos')
    etapas={
     'Theo':[('Referência','21 épocas'),('Repetição','36 épocas'),('Aug8','30 épocas')],
     'Lisandra':[('Execução publicada','10 épocas'),('Seleção por val_loss','Checkpoint da época 5')],
     'Dulce':[('Primeiro modelo','100 épocas'),('CNN224','50 / 200 épocas'),('CNN224 v2','120 épocas'),('ResNet18','5 épocas')],
     'Bandeira':[('V1','15 épocas'),('V2','23 épocas'),('V3','30 épocas'),('V4 → Final','30 + 10 épocas')]}
    for j,n in enumerate(NOMES):
        y=220+j*153;t.text(65,y+53,n,27,CORES[n],700)
        for i,(a,b) in enumerate(etapas[n]):
            x=290+i*308;t.rect(x,y,285,110);t.text(x+20,y+43,a,22,peso=600);t.text(x+20,y+81,b,20,'#657384')
    t.save(DEST/'Geral/04_evolucao_geral.svg')

def arquitetura(n):
    root=copy.deepcopy(ET.parse(RAIZ/n/'modelo principal/arquitetura_livro.svg').getroot())
    for pai in root.iter():
        for filho in list(pai):
            if filho.tag==f'{{{NS}}}text' and ''.join(filho.itertext()).startswith(n+' —'):pai.remove(filho)
    for k,v in dict(x='55',y='160',width='1490',height='680',preserveAspectRatio='xMidYMid meet').items():root.set(k,v)
    t=Tela('Arquitetura — '+n,n);t.s.append(ET.tostring(root,encoding='unicode'));t.save(DEST/n/'01_arquitetura.svg')

def treino(n):
    d=TREINO[n];t=Tela('Treinamento — '+n,n,d['origem'])
    for i,s in enumerate(d['fluxo']):
        x=65+i*385;t.rect(x,220,330,110);t.text(x+165,280,s,21,CORES[n],600,'middle')
        if i<3:t.line(x+337,275,x+373,275,seta=True)
    t.card(65,395,715,350,'Configuração',d['linhas'],CORES[n])
    t.card(820,395,715,350,'Seleção do modelo',d['selecao']+['Avaliação sem dropout','Histórico por época'],CORES[n])
    t.save(DEST/n/'02_treinamento.svg')

def ler_csv(p,mapa):
    with p.open(encoding='utf-8-sig',newline='') as f:return [{dest:float(r[orig]) for dest,orig in mapa.items()} for r in csv.DictReader(f)]

def chart(t,x,y,w,h,series,medida,titulo,selecionada=None):
    t.text(x,y-34,titulo,28,peso=600)
    vals=[r[medida+'_'+f] for _,rows in series for r in rows for f in ('treino','validacao')]
    xmax=max(r['epoca'] for _,rows in series for r in rows)
    lo,hi=(.4,1.) if medida=='acc' else (0,max(vals)*1.08)
    px=lambda v:x+(v-1)/max(1,xmax-1)*w
    py=lambda v:y+h-(v-lo)/(hi-lo)*h
    for i in range(5):
        v=lo+(hi-lo)*i/4;yy=py(v);t.line(x,yy,x+w,yy,esp=1)
        t.text(x-16,yy+6,f'{v:.0%}' if medida=='acc' else f'{v:.2f}',18,'#657384',ancora='end')
    for v in sorted(set([1,*[round(1+(xmax-1)*i/4) for i in range(1,5)]])):t.text(px(v),y+h+30,v,18,'#657384',ancora='middle')
    for _,rows in series:
        for f,c in (('treino','#2877ad'),('validacao','#da7639')):t.curve([(px(r['epoca']),py(r[medida+'_'+f])) for r in rows],c)
    if selecionada is not None:t.line(px(selecionada),y,px(selecionada),y+h,cor='#866294',traco='6 6')
    t.text(x+w/2,y+h+68,'Época',20,'#657384',ancora='middle')

def evolucao(n,rows,ep,sub,arquivo='03_evolucao.svg'):
    t=Tela('Treino e validação — '+n,n,sub)
    for x,c,s in ((70,'#2877ad','Treino'),(265,'#da7639','Validação'),(500,'#866294',f'Checkpoint: época {ep}')):
        t.line(x,201,x+45,201,cor=c,esp=3,traco='6 6' if x==500 else '');t.text(x+57,208,s,21)
    series=[(n,rows)];chart(t,120,320,620,385,series,'acc','Acurácia',ep);chart(t,920,320,590,385,series,'loss','Perda',ep)
    t.save(DEST/n/arquivo)

def historicos():
    mt={'epoca':'epoca','loss_treino':'bce_treino','loss_validacao':'bce_validacao','acc_treino':'acuracia_treino','acc_validacao':'acuracia_validacao'}
    rows=ler_csv(RAIZ/'Theo/modelo principal/treinamento_aug8/historico.csv',mt)
    evolucao('Theo',rows,29,'Aug8 · 30 épocas · seleção por acurácia de validação','03_evolucao_aug8.svg')
    h=json.loads((RAIZ/'Dulce/projeto/Cats vs Dogs/historico/historico_treino_224_v2.json').read_text())
    rows=[dict(epoca=i+1,loss_treino=h['perda_treino'][i],loss_validacao=h['perda_val'][i],acc_treino=h['acc_treino'][i],acc_validacao=h['acc_val'][i]) for i in range(120)]
    evolucao('Dulce',rows,112,'CNN224 v2 · 120 épocas · seleção por acurácia de validação')
    rows=ler_csv(RAIZ/'Lisandra/projeto/resultados_para_competicao/historico_epocas.csv',{'epoca':'epoca','loss_treino':'perda_treino','loss_validacao':'perda_validacao','acc_treino':'acuracia_treino','acc_validacao':'acuracia_validacao'})
    evolucao('Lisandra',rows,5,'10 épocas registradas · seleção por menor perda de validação')
    rows=ler_csv(RAIZ/'Bandeira/projeto/data/training_history_4.0_Final.csv',{'epoca':'epoch','loss_treino':'train_loss','loss_validacao':'val_loss','acc_treino':'train_acc','acc_validacao':'val_acc'})
    evolucao('Bandeira',rows,10,'V4 Final · 10 épocas de fine-tuning · seleção por menor perda de validação','03_evolucao_v4.svg')
    mapa={'epoca':'epoch','loss_treino':'train_loss','loss_validacao':'val_loss','acc_treino':'train_acc','acc_validacao':'val_acc'}
    base=ler_csv(RAIZ/'Bandeira/projeto/data/training_history_4.0.csv',mapa)
    finais=[dict(r,epoca=r['epoca']+30) for r in rows]
    t=Tela('V4 → V4 Final','Bandeira','30 épocas do zero + 10 de fine-tuning · mudança de etapa após a época 30')
    series=[('V4 + Final',base+finais)]
    chart(t,120,320,620,385,series,'acc','Acurácia',30.5)
    chart(t,920,320,590,385,series,'loss','Perda',30.5)
    for x,c,label in ((90,'#2877ad','Treino'),(320,'#da7639','Validação'),(630,'#866294','Linha vertical: início do fine-tuning')):
        t.line(x,215,x+45,215,cor=c,esp=3);t.text(x+57,222,label,21)
    t.save(DEST/'Bandeira/05_linhagem_v4_final.svg')
    etapas=json.loads((RAIZ/'Theo/modelo principal/graficos/resumo_trajetoria.json').read_text())
    series=[];offset=0;limites=[]
    for e,label in zip(etapas,('Referência','Repetição','Aug8')):
        rows=ler_csv(RAIZ.parent/'Cats and Dogs I e II/resultados_residuals_networks'/e['execucao']/'historico.csv',mt)
        for row in rows:row['epoca']+=offset
        series.append((label,rows));offset+=len(rows);limites.append(offset)
    t=Tela('Da referência ao Aug8','Theo','Fases da apresentação · reinícios e ramos independentes')
    for i,(s,rows) in enumerate(series):t.text(85+i*375,205,f'{i+1}. {s} · {len(rows)} épocas',22,CORES['Theo'],600)
    chart(t,120,320,620,385,series,'acc','Acurácia');chart(t,920,320,590,385,series,'loss','Perda')
    for x,w in ((120,620),(920,590)):
        for b in limites[:-1]:
            xx=x+b/(offset-1)*w;t.line(xx,320,xx,705,cor='#8a9198',esp=1,traco='4 5')
    for x,c,s in ((90,'#2877ad','Treino'),(300,'#da7639','Validação')):
        t.line(x,824,x+45,824,cor=c,esp=4);t.text(x+57,832,s,20)
    t.save(DEST/'Theo/04_evolucao_completa.svg')

def extras():
    t=Tela('De onde vieram os pesos do Aug8','Theo');c=CORES['Theo']
    t.card(65,275,350,240,'Referência',['ImageNet','21 épocas observadas','Referência de método'],c)
    t.card(530,275,370,240,'Repetição',['Reinício ImageNet','36 épocas observadas','Checkpoint da época 33'],c)
    t.card(1040,275,490,240,'Aug8 principal',['Parte do checkpoint 33','30 épocas · 8 ampliações','Checkpoint escolhido: 29'],c)
    t.line(425,392,515,392,traco='6 5',seta=True);t.line(910,392,1023,392,seta=True)
    t.save(DEST/'Theo/05_linhagem.svg')
    for n,titulo,sub,dados in [
        ('Dulce','Experimentos documentados','Cada execução preservada separadamente',[
          ('Primeiro modelo',['SGD · LR 0,01','100 épocas na execução original','Repetição independente com mais métricas']),
          ('CNN224',['Execução de 50 épocas','Outra execução de 200 épocas','Históricos distintos']),
          ('CNN224 v2',['Adam · LR 0,0003','120 épocas','Modelo final declarado pela autora']),
          ('ResNet18',['ImageNet + fine-tuning','5 épocas','Experimento separado'])]),
        ('Bandeira','Evolução das versões','V1–V4: do zero · Final: continua o checkpoint 30 da V4',[
          ('V1',['224 × 224 · Adam','15 épocas','Checkpoint: época 15']),
          ('V2',['224 × 224 · AdamW','23 épocas executadas','Checkpoint: época 18']),
          ('V3',['384 × 384 · AdamW','30 épocas','Checkpoint: época 28']),
          ('V4 → Final',['30 épocas + 10 de fine-tuning','Congelamento dos blocos 1–3 na fase final','Checkpoint final: época 10'])])]:
        t=Tela(titulo,n,sub)
        for i,(s,ls) in enumerate(dados):t.card(65+(i%2)*755,205+(i//2)*320,715,290,s,ls,CORES[n])
        t.save(DEST/n/'04_experimentos.svg')

def indices():
    arquivos=[]
    for pasta in ('Geral',*NOMES):
        ps=sorted((DEST/pasta).glob('*.svg'));arquivos+=ps
        ls=['# '+pasta+' — apresentação','']
        for p in ps:ls.append(f'- [{p.stem.replace("_"," ")}]({p.name})')
        (DEST/pasta/'README.md').write_text('\n'.join(ls)+'\n')
    (DEST/'README.md').write_text('''# Coisas para apresentação

SVGs prontos para navegador e slides, em páginas1600×900.

- [Geral](Geral/README.md): panorama dos modelos, métodos, dados e versões.
- [Theo](Theo/README.md): arquitetura, treino, curvas e linhagem até Aug8.
- [Lisandra](Lisandra/README.md): arquitetura com BatchNorm, treino e10 épocas.
- [Dulce](Dulce/README.md): CNN224 v2, treino, curvas e experimentos.
- [Bandeira](Bandeira/README.md): CNN V4 Final, fine-tuning, curvas e versões.
- [Galeria de todos os SVGs](galeria.html).
- [Relatório Pets publicado: métodos, ranking e gráficos](<Resumo publico Pets/README.md>).
- [Avaliação completa local: CSVs e fotos](<Resultados Pets/README.md>).

Arquiteturas sem rodapés explicativos; blocos, dimensões e resíduos preservados.
Detalhes ficam nos documentos principais para apoiar a fala de vocês.

As validações são distintas; não é um ranking em teste comum. Fases do Theo
incluem uma referência de método; o Aug8 parte diretamente do checkpoint33 da repetição.
No desenho, a cabeça densa tem escala esquemática e layer3 do Theo é abreviada.

Fontes: CSV/JSON dos treinamentos e configurações preservadas. Gerador:
../gerar_apresentacao.py. Os SVGs de treinamento usam históricos existentes. A avaliação Pets está separada e documentada, sem novo treinamento.
''')
    cards=[]
    for p in arquivos:
        rel=str(p.relative_to(DEST))
        cards.append(f'<section><h2>{html.escape(rel.replace("_"," "))}</h2><a href="{html.escape(rel)}"><img loading="lazy" src="{html.escape(rel)}" alt="{html.escape(p.stem)}"></a></section>')
        ET.parse(p)
    (DEST/'galeria.html').write_text('<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Coisas para apresentação</title><style>body{margin:32px;font-family:system-ui;background:#eceeec;color:#243343}section{background:white;padding:18px;margin:28px auto;max-width:1400px;border-radius:14px}img{width:100%;height:auto}h2{font-size:18px}</style><h1>Coisas para apresentação</h1><p><a href="Resultados%20Pets/20261106_190610/index.html">Resultados Pets — ranking, relatório e predições</a></p>'+'\n'.join(cards)+'</html>')
    return arquivos

def main():
    for n in NOMES:(DEST/n).mkdir(parents=True,exist_ok=True);arquitetura(n);treino(n)
    geral();historicos();extras()
    print(f'{len(indices())} SVGs gerados em {DEST}')

if __name__=='__main__':main()
