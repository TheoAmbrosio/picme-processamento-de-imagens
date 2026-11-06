# Método publicado pela Lisandra

AlexNetBinary do zero, canais96/256/384/384/256, kernels11/5/3/3/3.
BatchNorm após cada convolução, ReLU; pooling após conv1,conv2,conv5.
Flatten9216 →4096 → ReLU → Dropout50% →4096 → ReLU → Dropout50% →1 logit.
Sem Sigmoid no forward; usar sigmoid para obter probabilidade na avaliação.

Adam LR0.0001, betas0.9/0.999, eps1e-8, decay0; BCEWithLogitsLoss,
batch50,40 épocas previstas. Early stopping por val_loss, paciência5,
parada na10 e restauração da5, segundo os artefatos fornecidos pela autora.
Sem scheduler, pesos pré-treinados ou congelamento. Sem seed manual.
Normalização ImageNet e RGB224×224. Sem augmentation declarado.

Histórico confirma mínimo de perda na5, com90,16% de acurácia de validação.
Maior acurácia91,08% na8 não é o critério declarado do checkpoint.
O novo arquivo de pesos recebido é compatível com esta arquitetura.
Época/critério conforme declaração da autora; teste real não executado.
