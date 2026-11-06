# AlexNet Binary Image Classification: Cats vs Dogs

Repositório contendo a implementação e avaliação de uma Rede Neural Convolucional inspirada na arquitetura AlexNet (Krizhevsky et al., 2012), adaptada para a classificação binária de imagens (Gatos vs. Cachorros) em PyTorch.

## Visão Geral

- **Tarefa:** Classificação binária (0 = Gato, 1 = Cachorro).
- **Framework:** PyTorch & Torchvision.
- **Dataset:** `bhavikjikadara/dog-and-cat-classification-dataset` (divisão aleatória: 80% treino, 10% validação, 10% teste via `train_test_split`).
- **Dimensão de Entrada:** Imagens RGB de 224 × 224 × 3.

## Arquitetura da Rede (`AlexNetBinary`)

A rede possui 5 camadas convolucionais com BatchNorm e 3 camadas totalmente conectadas. O forward retorna um logit por imagem; a probabilidade é obtida aplicando sigmoide externamente:

- **Conv 1:** 96 filtros 11 × 11, stride 4, padding 2 → BatchNorm → ReLU → MaxPool (3 × 3, stride 2) → Saída: 27 × 27 × 96
- **Conv 2:** 256 filtros 5 × 5, stride 1, padding 2 → BatchNorm → ReLU → MaxPool (3 × 3, stride 2) → Saída: 13 × 13 × 256
- **Conv 3:** 384 filtros 3 × 3, stride 1, padding 1 → BatchNorm → ReLU → Saída: 13 × 13 × 384
- **Conv 4:** 384 filtros 3 × 3, stride 1, padding 1 → BatchNorm → ReLU → Saída: 13 × 13 × 384
- **Conv 5:** 256 filtros 3 × 3, stride 1, padding 1 → BatchNorm → ReLU → MaxPool (3 × 3, stride 2) → Saída: 6 × 6 × 256
- **Vetorização (Flatten):** 6 × 6 × 256 = 9216 elementos.
- **Classificador (FC):**
  - **FC 1:** Linear (9216 → 4096) + ReLU + Dropout (0,5)
  - **FC 2:** Linear (4096 → 4096) + ReLU + Dropout (0,5)
  - **FC 3:** Linear (4096 → 1), sem Sigmoid no modelo

## Treinamento e pré-processamento

Adam com learning rate `0.0001`, batch `50` e `BCEWithLogitsLoss`.
São previstas 40 épocas; os resultados publicados registram 10, com early
stopping de paciência 5 pela menor perda de validação e restauração da época 5.
Na época selecionada, a acurácia de validação é 90,16%; o máximo de acurácia
da série é 91,08% na época 8, que não é o critério de escolha dos pesos.

Imagens RGB: `Resize((224, 224))`, `ToTensor` e normalização ImageNet
(`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`). Sem data augmentation
ou seed manual definidos. Rótulos pela pasta pai imediata: Cat=0 e Dog=1.
Durante a classificação, logit >= 0 equivale a probabilidade >= 0,5.

Configuração, histórico e informações declaradas do checkpoint estão em
[resultados_para_competicao](resultados_para_competicao/README.md).
