# res_theo_especial50

ResNet-50 Bottleneck completa, com blocos `[3,4,6,3]` e `conv1=64`.
As larguras internas são `16/32/64/128` (1/4 das originais), com saídas
`64/128/256/512`. A `layer4` é mantida e o classificador é `Linear(512,1)`.

Somente `conv1` recebe pesos ImageNet V1. Ela fica congelada nas épocas 1–5 e
treina a partir da época 6. As outras camadas usam He normal `fan_out`.

O dropout incremental está ativado: 0% até a época 20 e aumento linear até 10%
na época 40. Dados originais são esticados para RGB 224×224.

```bash
bash iniciar_res_theo_especial50.sh
```
