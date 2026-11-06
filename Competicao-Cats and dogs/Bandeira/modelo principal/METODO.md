# Método documentado no código V4

CNN do zero com cinco blocos, canais32/64/128/256/512, entrada384×384.
Cada bloco: uma Conv3×3 padding1, BatchNorm, ReLU e MaxPool2.
GAP → Flatten → Dropout50% → Linear512→1. BCEWithLogitsLoss.
AdamW LR0.001, decay0.0001, batch64, até30 épocas, paciência7,
checkpoint por menor perda de validação. Cosseno30 até1e-6 e AMP em CUDA.

Divisão80/20 estratificada, seed42. Treino: RandomResizedCrop384
(scale0.8–1), flip horizontal, rotação±15°, brilho/contraste0.2.
Validação: Resize384×384. Normalização própria preservada em parametros.json.

A análise V4 salva informa acurácia0.9464 e F1 binário0.9461 no conjunto
de validação de5000 imagens. São saídas já existentes, não recalculadas.
Atualização publicada: histórico V4 completo com30 épocas; menor perda de
validação na época30. CSVs de predições/histórico e pesos agora disponíveis.
As predições salvas reproduzem94,64% de acurácia e F1 binário0,9461;
pesos V4 carregados estritamente no modelo e forward artificial aprovado.
