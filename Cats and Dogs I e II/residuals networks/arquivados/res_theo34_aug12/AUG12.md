# ResTheo34 Aug12 — dois logits e CrossEntropy

Origem: melhor Aug8 da época29,98.32% validação. Treino com barras foi encerrado
por autorização e seus dados/checkpoints permanecem preservados.

Mantidos:30 épocas,batch64,AdamW novo com decay0.01; grupos base1e-5,
layer3 5e-5,fc1e-4; CosineAnnealingLR T_max30,eta_min1e-7; dropout10%
após Flatten,sem layer4. A cabeça passou de Linear256→1 para Linear256→2.

Conversão inicial preserva probabilidade: primeira linha de peso/bias em0,
segunda linha copiada do logit antigo. Assim softmax([0,z])[Dog]=sigmoid(z).
Depois ambas as linhas são treináveis. A perda é CrossEntropyLoss,com classes
inteiras Cat0/Dog1 e sem softmax antes da perda. Avaliação usa softmax/argmax.

12 versões virtuais por original de treino:8 versões Aug8 anteriores; uma
delas sorteada novamente e acrescida de faixa horizontal preta,largura0–50px,
posição vertical aleatória; original com blur médio10×10; ruído com esse blur;
quadrados com esse blur. Média10×10 usa padding refletido4px antes/5px depois
para preservar224×224; blur aplicado depois da corrupção e antes da normalização.
Largura0 permite a versão sem faixa. Sem alteração nas5000 imagens de validação.

Total239976 amostras virtuais de treino,12 por original. Mesma divisão80/20,
RGB224×224 e normalização ImageNet. F1,acurácia e CE por época; parâmetros,
pesos,scheduler e melhor/último checkpoint preservados na pasta de execução.
