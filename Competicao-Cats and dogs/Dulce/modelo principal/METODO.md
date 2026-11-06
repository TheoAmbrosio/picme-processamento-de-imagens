# Método documentado

Cinco blocos Conv5×5(padding1) → ReLU → MaxPool2 → Dropout0.2,
canais32/64/64/128/128; mapas após pooling111/54/26/12/5.
Flatten3200 →1024 →32 →2, ReLU nas duas camadas intermediárias.
Treino com logits e entropia cruzada; forward expõe softmax.
Adam LR0.0003, batch32,120 épocas, seed18. Sem pesos pré-treinados neste modelo.

Filtragem de duplicatas/quase duplicatas antes de dividir por grupos:
15% para teste e15% do restante para validação. Saída registrada:
17934/3165/3723 originais, aproximadamente72,25%/12,75%/15%;
descartadas imagens com lado menor que94px. Resize bilinear mantendo proporção
e corte central224. Pixels convertidos para[0,1]. Treino ganha versões fixas:
10% com rotação15° e10% com ruído gaussiano (desvio2 na escala0–255).
Em cada lote: espelhamento, zoom0.7–1, deslocamento, rotação±15°,
brilho e contraste0.8–1.2. Validação somente originais.

Histórico encontrado:120 épocas. Melhor acurácia de validação
96.9036%, época112. Critério de checkpoint: maior
acurácia de validação. Métricas de teste já presentes no repositório não
foram usadas para escolher ou alterar configurações nesta organização.
