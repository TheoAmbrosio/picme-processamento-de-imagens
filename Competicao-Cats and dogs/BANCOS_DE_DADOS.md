# Bancos de dados da competição

Documentação baseada em código, metadados e saídas salvas em06/11/2026.
Não foram baixadas imagens, executados notebooks ou refeitos splits.

| Participante | Modelos/bancos documentados | Índice |
|---|---|---|
| Theo | Aug8 detalhado e inventário por modelo/execução, inclusive arquivados | [Abrir](Theo/modelo%20principal/BANCO_DE_DADOS.md) |
| Dulce | versão inicial,224,224_v2,ResNet18 | [Abrir](Dulce/modelo%20principal/BANCO_DE_DADOS.md) |
| Bandeira | V1,V2,V3,V4 com origem de cada informação | [Abrir](Bandeira/modelo%20principal/BANCO_DE_DADOS.md) |
| Lisandra | AlexNetBinary vigente no GitHub | [Abrir](Lisandra/modelo%20principal/BANCO_DE_DADOS.md) |

O mesmo nome informal Cats and Dogs não significa o mesmo split. Foram
preservadas diferenças entre fonte Kaggle, limpeza, contagens, entrada,
normalização e augmentation. Contagem de variantes não equivale à contagem
de animais originais independentes. Fontes ausentes estão indicadas nas fichas.

## Quantidades dos modelos principais

| Modelo | Originais de treino | Amostras de treino | Validação | Teste interno |
|---|---:|---:|---:|---:|
| Theo Aug8 |19998|159984 virtuais/época|5000|0 nesta divisão|
| Dulce CNN224 v2 |17934|21520 com cópias fixas + augmentation online|3165|3723|
| Bandeira V4 |20000 esperados, não contados localmente|sem multiplicador fixo; augmentation online|5000 nas saídas salvas|test1 separado, tamanho12500 declarado no README|
| Lisandra |19998 nas saídas salvas|19998, sem augmentation explícito|2500|2500|

Os números não tornam os conjuntos equivalentes. Na versão vigente da Lisandra,
a regra de rótulo pela pasta pai está corrigida, mas não há contagem de classes
publicada nem manifesto completo dos splits.
