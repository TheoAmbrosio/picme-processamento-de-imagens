# theo_sem_res_28_42_64_88

CNN sequencial, sem resíduos: entrada RGB 224×224, Conv7×7 3→28 com stride 2,
BN/ReLU/MaxPool; estágios 28 → 42 → 64 → 88, cada um com uma única Conv3×3,
BN e ReLU. Primeiro estágio stride 1, demais stride 2. Média global, Flatten
e Linear para um logit (Cat=0, Dog=1). Inicialização He normal fan_out.

Mesma agenda da Theo 28/42/64: 80 épocas, batch 16, AdamW LR 0.001,
weight decay 0.01. Épocas 1–50 originais; 51–80 com as cinco variantes
sem rotação da Theo Ajuste (ruído variância 0.08 e novos quadrados sorteados).
Validação original balanceada e separada dos derivados de treino.

Warmup somente na fase ampliada: 1e-6 na época 51 até 0.001 na 75.
Dropout: 0% na 1, 20% na 25, 10% na 50, 5% na 75 e até 80; aplicado
após Conv/BN/ReLU internas, BN/ReLU/pooling inicial e pooling/Flatten finais.
Conv7×7 inicial e classificador ficam sem dropout. Avaliação desativa dropout.

Execute na raiz: `bash iniciar_theo_sem_res_28_42_64_88.sh`.
Resultados em `resultados_residuals_networks/theo_sem_res_28_42_64_88/<execução>/`,
incluindo F1, gráficos por época, fontes, parâmetros e melhor/último checkpoint.
O script inicia em segundo plano e confirma CUDA. Use um treino por vez.
