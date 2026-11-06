"""Consolida a publicação V4 Final do autor; não executa treino ou avaliação."""
from pathlib import Path
import json, subprocess, base64

RAIZ=Path(__file__).resolve().parent

def atualizar():
    repo=RAIZ/'Bandeira/projeto';d=RAIZ/'Bandeira/modelo principal'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    p=d/'parametros.json';anterior=json.loads(p.read_text())
    if anterior['versao']=='V4, código vigente':
        (d/'parametros_v4_base.json').write_text(json.dumps(anterior,ensure_ascii=False,indent=2)+'\n')
    params=dict(anterior,versao='V4 Final — fine-tuning',learning_rate=1e-5,epocas_maximas=10,
        early_stopping_patience=4,T_max=10,eta_min=1e-7,commit=commit,
        checkpoint='projeto/data/best_model_4.0_Final.pth',melhor_epoca_por_val_loss=10,
        historico='projeto/data/training_history_4.0_Final.csv',
        fonte='projeto/src/model.py, fine_tuning.py, dataset.py',
        checkpoint_inicial='projeto/data/best_model_4.0.pth',epoca_checkpoint_inicial=30,
        congelados='features[:12]: parâmetros dos blocos 1–3',
        treinaveis='features[12:] e classifier: blocos 4–5 e cabeça',
        batchnorm='model.train() mantém estatísticas correntes dos blocos congelados atualizáveis',
        augmentation=dict(random_resized_crop=dict(size=384,scale=[.9,1.0]),flip_horizontal=.5,
            rotacao_graus=5,color_jitter=dict(brightness=.05,contrast=.05)),
        validacao='Resize((384,384)), ToTensor, Normalize; preparação offline documentada separadamente')
    p.write_text(json.dumps(params,ensure_ascii=False,indent=2)+'\n')
    (d/'README.md').write_text('''# Bandeira — CNN V4 Final

Modelo atual: fine-tuning do checkpoint da época 30 da V4, publicado no commit `'''+commit[:7]+'''`.
A arquitetura foi preservada: cinco blocos 32→64→128→256→512, média global e uma saída.

- [Método e fases](METODO.md) · [Registro](REGISTRO_MODELO.md).
- [Parâmetros finais](parametros.json) · [Parâmetros da V4 base](parametros_v4_base.json).
- [Arquitetura no formato livro](arquitetura_livro.svg).
- [Evolução por época e painel geral](evolucao%20do%20treinamento/README.md).
- [Banco e transformações](BANCO_DE_DADOS.md).
- [Conferência final](CONFERENCIA_FINAL.json) · [Pontos de atenção](CONFERENCIA_ATUALIZACAO.md).
- [Checkpoint final](../projeto/data/best_model_4.0_Final.pth).
- [Histórico final](../projeto/data/training_history_4.0_Final.csv).
- [Predições salvas](../projeto/data/val_predictions_4.0_Final.csv).
- [Código do fine-tuning](../projeto/src/fine_tuning.py).

Checkpoint escolhido por menor perda: época 10 do fine-tuning, acurácia 94,76%,
BCE 0,13356867 e F1 0,94726248. Maior acurácia da etapa: 94,80% nas épocas 3 e 9.
V1–V4 continuam preservadas como execuções anteriores; a etapa Final deriva apenas da V4.
Sem novo treinamento ou avaliação no conjunto de imagens.
''')
    (d/'METODO.md').write_text('''# Método — V4 e ajuste final

A CNN mantém entrada RGB 384×384 e cinco blocos Conv3×3 padding1, BatchNorm,
ReLU e MaxPool2, com 32/64/128/256/512 canais. GAP, Flatten, Dropout50% e Linear512→1.
Perda BCEWithLogitsLoss; sigmoid e limiar >0,5 na avaliação.

## V4 base — 30 épocas

Inicialização do zero, AdamW LR0,001, decay0,0001, batch64, cosseno T_max30,
eta_min1e-6, early stopping7. Checkpoint por menor perda na época30:94,64%.
Treino: RandomResizedCrop384(scale0,8–1), flip, rotação±15° e jitter0,2.

## V4 Final — 10 épocas adicionais

Carrega best_model_4.0.pth (época30). Novo AdamW LR1e-5, decay1e-4, batch64,
cosseno T_max10 até1e-7, early stopping4. Mantém Dropout50% e BCEWithLogitsLoss.
Parâmetros dos blocos1–3 congelados; blocos4–5 e cabeça treináveis.
O código chama model.train(): buffers de média/variância das BatchNorm dos
blocos congelados continuam sendo atualizados; requires_grad=False congela seus parâmetros.

Augmentation suave online: RandomResizedCrop384(scale0,9–1), flip horizontal0,5,
rotação±5°, jitter brilho/contraste0,05. Normalização própria do treino preservada.
Split80/20 estratificado, seed42. Validação sem augmentation aleatório,
Resize384×384, ToTensor e mesma normalização. A preparação offline das imagens
está documentada no banco de dados; há um ponto de confirmação para imagens novas.

Executou10 épocas. Menor perda na10:0,1335686746; acurácia94,76%, F1≈0,94726248.
Maior acurácia94,80% nas épocas3 e9, sem substituir o critério de checkpoint do autor.
''')
    nota='''\n\n## 06/11/2026 — V4 Final publicada\n\nAtualização para commit `'''+commit[:7]+'''`: fine-tuning de10 épocas a partir da V4/30.
Arquitetura preservada. Checkpoint Final/10 por menor perda;94,76% e F1≈0,94726248.
Histórico, métodos, banco, gráficos e apresentação atualizados. Peso carregado
estritamente e forward artificial aprovados; métricas recalculadas exclusivamente
nas predições publicadas, sem avaliação real, treino ou teste final.
'''
    for p in [d/'REGISTRO_MODELO.md',RAIZ/'REGISTRO_ORGANIZACAO.md']:
        if 'V4 Final publicada' not in p.read_text():
            with p.open('a') as f:f.write(nota)
    with (d/'CONFERENCIA_ATUALIZACAO.md').open('a') as f:
        if 'V4 Final — bd92c13' not in (d/'CONFERENCIA_ATUALIZACAO.md').read_text():f.write('''\n\n## V4 Final — bd92c13\n\nNova etapa de10 épocas, derivada do checkpoint V4/30. Checkpoint por menor
val_loss na10 (94,76%), máximo de acurácia94,80% nas3 e9. Conferência detalhada
em CONFERENCIA_FINAL.json: carregamento estrito, pesos finitos, forward artificial
1×3×384×384→1×1;5000 predições conferidas com o histórico. Não avaliado nas imagens.
O ponto sobre preparação offline versus imagem nova descrito acima permanece.
A congelamento anunciado se aplica aos parâmetros; os buffers de BatchNorm
continuam atualizáveis no código do autor. Nenhum código original foi alterado.
''')
    (d/'banco de dados/V4_Final.md').write_text('''# Bandeira — dados do fine-tuning V4 Final

Mesma origem, split estratificado80/20 e seed42 da V4:20.000 treino e5.000
validação esperados para25.000 JPEGs. As predições publicadas contêm5.000 exemplos.
Não há multiplicação fixa da base: augmentation online a cada leitura.

Treino: RandomResizedCrop384(scale0,9–1; ratio padrão3/4–4/3), flip0,5,
rotação±5°, ColorJitter brilho/contraste0,05, ToTensor e normalização do treino
em norm_stats_3.0.json. Validação: Resize((384,384)), ToTensor, mesma normalização.
Os arquivos podem já ter sido preparados offline; consulte V4.md.

Fonte: ../projeto/src/dataset.py e src/fine_tuning.py, commit bd92c13.
''')
    p=d/'BANCO_DE_DADOS.md';s=p.read_text()
    if 'V4_Final.md' not in s:s+='\n- [V4 Final — dados do fine-tuning](<banco de dados/V4_Final.md>).\n'
    p.write_text(s)
    fontes=RAIZ/'FONTES.json';dados=json.loads(fontes.read_text())
    for v in dados:
        if v['participante']=='Bandeira':v.update(commit=commit,modelo='CNN V4 Final',checkpoint='data/best_model_4.0_Final.pth')
    fontes.write_text(json.dumps(dados,ensure_ascii=False,indent=2)+'\n')
    (RAIZ/'Bandeira/README.md').write_text('# Bandeira\n\n[Modelo atual: CNN V4 Final](modelo%20principal/README.md). Fine-tuning da V4; versões anteriores preservadas.\n')
    p=RAIZ/'README.md';s=p.read_text().replace('CNN V4 vigente em src/model.py','CNN V4 Final — fine-tuning de10 épocas').replace('**Bandeira:** fonte V4 e análises V1–V4.','**Bandeira:** V4 Final, derivada da V4;10 épocas de fine-tuning. Análises V1–V4 preservadas.');p.write_text(s)
    for p in [RAIZ/'EVOLUCAO_TREINAMENTOS.md']:
        s=p.read_text().replace('Históricos completos V1–V4, CSVs e gráficos numéricos','Históricos V1–V4 e V4 Final (10 épocas), CSVs e gráficos numéricos');p.write_text(s)
    # Figuras originais publicadas, preservando as versões anteriores.
    nb=json.loads((repo/'Analysis/results_final.ipynb').read_text());n=0
    for cell in nb['cells']:
        for out in cell.get('outputs',[]):
            image=out.get('data',{}).get('image/png')
            if image:
                n+=1;(d/'graficos'/f'results_final_{n}.png').write_bytes(base64.b64decode(''.join(image) if isinstance(image,list) else image))
    p=d/'graficos/README.md';s=p.read_text()
    if 'results_final_1.png' not in s:
        s+='\n## Figuras originais da etapa Final\n\n'+''.join(f'- [Figura {i}](results_final_{i}.png)\n' for i in range(1,n+1));p.write_text(s)
    print('Bandeira V4 Final documentada: '+commit[:7])

if __name__=='__main__':atualizar()
