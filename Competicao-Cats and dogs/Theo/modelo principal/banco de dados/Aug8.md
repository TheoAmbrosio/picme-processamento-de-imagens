# Theo — banco do Aug8 principal

## Origem e disponibilidade

- Kaggle: `shaunthesheep/microsoft-catsvsdogs-dataset`, conforme preparador.
- Originais em `Cats and Dogs I e II/dataset/PetImages/{Cat,Dog}`.
- Fonte informa12500 arquivos por classe. Relatório da preparação:12499
  válidos por classe, um ilegível em cada;24998 válidos no total.
- Dados não foram copiados para a pasta da competição nem baixados novamente.

## Divisão antes de ampliar

Divisão aleatória balanceada80/20, semente42. Por classe:9999 treino,
2500 validação,0 teste. Total:19998 originais de treino e5000 de validação.
SHA256 agrupa conteúdo idêntico para manter derivados no mesmo lado;
duplicatas não são todas removidas. Rótulos conflitantes aceitos conforme
a configuração histórica. Cat=0, Dog=1. Teste final separado não executado.

## Entrada do modelo e avaliação

RGB224×224. Normalização ImageNet: média[0.485,0.456,0.406],
desvio[0.229,0.224,0.225]. Validação usa os JPEGs históricos preparados
por Resize256 bilinear + CenterCrop224; não recebe novas corrupções.

## Oito amostras virtuais por original de treino

1. Original esticada para224×224.
2. Ruído gaussiano, variância0.08, média0;2% dos pixels brancos.
3. Oito quadrados pretos, lados20–40px; centros na região central150×150.
4. Esticamento horizontal2× seguido de recorte central224.
5. Esticamento vertical2× seguido de recorte central224.
6. Ruído + quadrados.
7. Esticamento horizontal + ruído + quadrados.
8. Esticamento vertical + ruído + quadrados.

Total159984 amostras virtuais de treino por época. Sorteios novos na leitura;
isso não significa159984 arquivos físicos independentes. Variantes usam
o split do original; a validação não é multiplicada por8.

## Diferença para as etapas anteriores

Referência e repetição usaram16 variantes físicas
por original:2 geometrias(recorte/esticada) ×2 espelhamentos ×4 rotações
(0/90/180/270). São319968 amostras de treino, com os mesmos5000 exemplos
de validação. A ampliação8× pertence ao Aug8, não a essas fases anteriores.

## Fontes

- [Configuração da divisão](<../../../../Cats and Dogs I e II/dataset_ampliado/224x224/divisao_balanceada/config_dataset.json>).
- [Manifesto dos originais](<../../../../Cats and Dogs I e II/dataset_ampliado/224x224/divisao_balanceada/manifesto_originais.csv>).
- [Relatório da preparação](<../../../../Cats and Dogs I e II/dataset_ampliado/224x224/divisao_balanceada/relatorio.txt>).
- [Especificação Aug8](<../../../../Cats and Dogs I e II/residuals networks/res_theo34_aug8/AMPLIACAO.md>).
