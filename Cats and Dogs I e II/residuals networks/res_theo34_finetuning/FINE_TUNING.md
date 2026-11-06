# Fine-tuning discriminativo da época 33

Pesos do melhor ResTheo34 repetição (97,34% validação na época33). Nova execução
de30 épocas, numeradas1–30 e acumuladas34–63. Toda a rede aprende.
Layer4 permanece removida; grupo alto contém somente layer3.

AdamW novo, decay0.01 em todos os parâmetros. Grupos:
conv1/bn1/layer1/layer2 LR1e-5; layer3 LR5e-5; fc LR1e-4.
CosineAnnealingLR T_max30, eta_min1e-7, step após cada época.
Histórico registra taxas usadas; checkpoint preserva scheduler e otimizador.

Dropout fixo7% após Flatten, antes de fc; sem dropout interno. Batch64,
16 variantes por original de treino e validação determinística do mesmo manifesto.
Novos checkpoints último/melhor e snapshots em resultados_residuals_networks/
res_theo34_finetuning/. O melhor da fase anterior permanece na pasta de origem.
