# Theo Ajuste — continuação independente

Configuração vigente documentada em [CONTINUACAO.md](CONTINUACAO.md).
O texto abaixo descreve a origem histórica da arquitetura, não a agenda atual.

Restauração exata da arquitetura usada na execução histórica `155432`:

- conv1 7×7 com 32 canais, sem pesos ImageNet;
- BasicBlocks `[2,2,2]` com canais `32/64/128`;
- `layer4` removida;
- saída `Linear(128,1)`;
- He normal `fan_out`; dropout 0→20% entre épocas 20–50, mantido até a 80
  e reduzido linearmente a 5% na época 100;
- AdamW com weight decay 0,01, batch 16 e 110 épocas; warmup linear de
  `0,000001` na época 1 até `0,001` na época 30, mantendo `0,001` depois;
- divisão balanceada por original: 80% treino e 20% validação;
- treino com cinco versões determinísticas por original: esticada, uma rotação
  entre 90°/180°/270°, cinza, ruído gaussiano de variância 0,1 com 2% dos
  pixels brancos e cutout com oito quadrados pretos de lados entre 20–40 px;
- cinza, ruído e cutout partem aleatoriamente da original ou da rotacionada;
- validação usa somente a imagem original esticada para RGB 224×224.
- nas épocas 1–20, o treino também usa somente os originais; as cinco variantes
  entram a partir da época 21.

```bash
bash iniciar_theo_ajuste.sh
```
