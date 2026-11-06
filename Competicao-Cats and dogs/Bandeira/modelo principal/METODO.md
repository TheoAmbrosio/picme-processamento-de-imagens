# Método — V4 e ajuste final

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
