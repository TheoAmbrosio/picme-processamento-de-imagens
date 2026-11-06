# Fine-tuning com ampliação8× e dropout10%

Origem: melhor checkpoint da repetição, época33, 97.34% validação. Nova fase
de30 épocas, AdamW decay0.01, grupos LR1e-5/5e-5/1e-4 e cosseno T_max30,
eta_min1e-7. Dropout10% na entrada de fc; a arquitetura permanece até layer3.

Oito versões virtuais por original de treino, sem alterar a divisão:
original RGB esticada224×224; ruído; quadrados; horizontal2×+crop central;
vertical2×+crop central; ruído+quadrados; horizontal+ruído+quadrados;
vertical+ruído+quadrados. Novo ruído/novas posições em cada leitura.

Ruído gaussiano variância0.08, média0, clamp[0,1] e2% dos pixels brancos.
Oito quadrados pretos, lados20–40px, centros uniformes em [37,187) em cada
eixo (região central150×150). Quadrados podem se sobrepor.
Esticamento horizontal224×448 e vertical448×224, seguido de CenterCrop224.
Corrupções ocorrem antes da normalização ImageNet.

Treino:159984 amostras; validação:5000 JPEGs determinísticos históricos,
sem novas corrupções. Arquivos anteriores e checkpoint de origem preservados.
