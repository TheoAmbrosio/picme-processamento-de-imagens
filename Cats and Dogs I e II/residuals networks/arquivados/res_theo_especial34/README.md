# res_theo_especial34

ResNet-34 completa e estreita: blocos `[3,4,6,3]`, canais
`64/32/64/128/256`, layer4 mantida e saída `Linear(256,1)`.

Conv1 usa ImageNet V1 e fica congelada nas épocas 1–5. Dropout incremental vai
de 0% até 10% entre as épocas 20 e 40. Total: 60 épocas.
