# Dulce — cats_and_dogs

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

## Versão inicial100×94

Redimensionamento bilinear para largura100, altura94, determinado pela menor
área restante. Pixels[0,1], tensor N×3×94×100. Antes de dividir,10% dos
originais recebem uma cópia girada15° e outros10% uma com ruído gaussiano
(desvio2 em escala0–255). Total29786 versões. Divisão por origem após essas
cópias:21520 treino,3798 validação,4468 teste. Portanto nesta versão
validação/teste também podem conter derivados da mesma família de transformações,
mantidos no mesmo grupo do original. Não atribuir a ela a estratégia posterior
de ampliar somente o treino.

## Fontes

- [Notebook original](<../../projeto/Cats vs Dogs/cats_and_dogs.ipynb>).
- [Células e saídas usadas como evidência](<fontes/cats_and_dogs.json>).

Imagens e splits não foram baixados nem recriados nesta organização.

## Associação aos históricos confirmada pela autora

As duas séries de100 épocas em historico_treino.json são deste primeiro modelo, SGD LR0.01. A série acc_val_modelo_fechado é a execução original (só validação); repeticao é uma execução posterior com mesma configuração e seed, para registrar perda/acurácia de treino e validação.
