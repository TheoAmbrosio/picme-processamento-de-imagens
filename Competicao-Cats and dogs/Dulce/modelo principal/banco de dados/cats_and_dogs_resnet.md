# Dulce — cats_and_dogs_resnet

## Banco original e limpeza

Kaggle `bhavikjikadara/dog-and-cat-classification-dataset` via kagglehub.
Saídas salvas:12499 arquivos em Cat e12499 em Dog. Após remoção de
repetidos por hash:24966 imagens,24962 origens perceptuais.
Grupos com rótulos conflitantes são excluídos; grupos idênticos de mesma
classe ficam com um arquivo. Impressão visual9×8 em cinza agrupa imagens
com mesmo padrão de comparações vizinhas. Isso é o algoritmo documentado,
não prova que toda duplicata visual possível foi detectada.
Excluídos lados menores que94px:24822 originais restantes.
Classes salvas:['Cat','Dog']; Cat=0, Dog=1. Semente18.

## Divisão efetivamente implementada

O código separa15% para teste e depois15% **do restante** para validação.
Proporções teóricas72,25%/12,75%/15%, com arredondamentos e grupos inteiros;
não70%/15%/15%. Os números abaixo são os impressos nas saídas originais.
Não foi encontrado manifesto completo dos arquivos de cada split no clone.

## Originais separados antes de ampliar

17934 treino original;3165 validação;3723 teste. RGB224×224 com bilinear,
preservação da proporção e corte central.2785 originais exigem ampliação
porque o lado menor é menor que224. Validação/teste somente originais.

Sem cópias fixas de rotação/ruído neste notebook:17934 imagens de treino.
Augmentation durante os lotes: flip horizontal50%, zoom0.7–1, rotação±15°,
deslocamento dependente do zoom, brilho/contraste0.8–1.2, interpolação bilinear
com padding por reflexão. Entrada[0,1] e depois normalização ImageNet dentro
do modelo: média0.485/0.456/0.406, desvio0.229/0.224/0.225.

## Fontes

- [Notebook original](<../../projeto/Cats vs Dogs/cats_and_dogs_resnet.ipynb>).
- [Células e saídas usadas como evidência](<fontes/cats_and_dogs_resnet.json>).

Imagens e splits não foram baixados nem recriados nesta organização.
