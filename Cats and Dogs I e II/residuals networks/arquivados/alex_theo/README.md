# AlexTheo

AlexNet oficial com pesos ImageNet V1. A estrutura de `features`, `avgpool` e do
classificador é preservada; somente `classifier[6]` muda de `Linear(4096,1000)`
para `Linear(4096,1)` para classificação binária Cat=0/Dog=1.

Configuração copiada da ResTheoEspecial50: Adam 0.001, batch 16, 60 épocas,
dados originais RGB 224×224, divisão 80/20 e dropout do classificador crescente
de 0% (até época 20) a 10% (época 40). A primeira convolução 11×11 fica
congelada nas épocas 1–5 e é liberada na época 6.

```bash
bash iniciar_alex_theo.sh
```
