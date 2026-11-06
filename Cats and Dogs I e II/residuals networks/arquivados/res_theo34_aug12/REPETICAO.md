# Repetição histórica 115733

Recria a execução `res_theo34/complementados,dropout,com_back`, usando parâmetros
recuperados do checkpoint da época 21 e confirmados pelo log original.
O nome da pasta não define a metodologia: o dropout efetivo era zero.

ResNet34 ImageNet V1 até layer3, sem layer4, fc 256→1 com peso e bias Normal(0,1).
Adam 0.001 com defaults, batch 64, 50 épocas, backbone congelado nas épocas 1–5,
liberado na 6. Semente 42. 16 variantes por original de treino, validação usual
do manifesto histórico (Resize256 + CenterCrop224), divisão balanceada 80/20.
AMP ligado, workers4, sem warmup/compile/channels_last/cuDNN benchmark.

Uma nova execução parte novamente do ImageNet, não do checkpoint treinado.
Preserva ultima_epoca.pt e melhor_modelo.pt por maior acurácia de validação,
além de configuração, fontes, histórico e log em pasta própria. A repetição
não garante igualdade numérica entre versões/máquinas/operações CUDA.
