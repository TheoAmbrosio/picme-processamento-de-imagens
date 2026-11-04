# Theo Ajuste

Fine-tuning independente dos pesos da execução histórica de 04/11/2026 às
15:54:32, checkpoint `ultima_epoca.pt` da época 50. O estado do otimizador
começa novo. O manifesto deve ter exatamente o mesmo hash da execução original.

40 épocas, batch 16, AdamW (`weight_decay=0.01`), warmup linear de 1e-6 na
época 1 até 0.001 na 25, mantido depois. Dropout: 0 na época 1, 15% na 15,
redução linear a 5% na 25, mantido até 40. Toda a rede aprende.

Somente treino recebe cinco versões, todas na orientação original: normal,
cinza, ruído gaussiano de variância 0.08 + 2% de pixels brancos, oito quadrados
de cores RGB sorteadas e oito quadrados pretos. Cada quadrado tem lado 20–40 px;
posições e cores são sorteadas novamente a cada leitura. Os sorteios são
independentes entre as duas variantes de quadrados. Sobreposições são possíveis.
Validação usa somente originais esticadas, mantendo a separação por origem.

F1 binário usa Dog=1, limiar 0.5, calculado a partir de TP/FP/FN de toda a época.
Gráficos de BCE, acurácia e F1 usam somente épocas no eixo horizontal.
Os resultados ficam em `resultados_residuals_networks/theo_ajuste/`.

Execute `bash iniciar_theo_ajuste.sh` depois de terminar o treino original.
