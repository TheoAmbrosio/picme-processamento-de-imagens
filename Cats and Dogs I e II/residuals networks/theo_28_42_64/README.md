# Theo 28/42/64

Variante da arquitetura principal com conv1=28, layer1=28, layer2=42 e layer3=64;
dois BasicBlocks por estágio, projeções residuais compatíveis, média global e
Linear(64,1). Layer4 continua removida. Inicialização He normal fan_out.
`criar_modelo()` instala o dropout nas posições configuradas.
Treino do zero em 80 épocas, batch 16, AdamW 0.001, weight decay 0.01.
Épocas 1–50: originais, LR fixa; dropout 0% na 1, 20% na 25, 10% na 50.
Épocas 51–80: cinco variantes sem rotação da Theo Ajuste; warmup 1e-6 na 51
até 0.001 na 75; dropout cai linearmente de 10% na 50 até 5% na 75,
mantendo 5% até 80. O mesmo otimizador continua entre as fases.
Validação original e balanceada, sem ampliação. F1, BCE e acurácia por época.
Execute `bash iniciar_theo_28_42_64.sh` na raiz. Resultados em
`resultados_residuals_networks/theo_28_42_64/<execução>/`.
