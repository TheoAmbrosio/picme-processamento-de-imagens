# Registro de trabalho — Cats and Dogs I e II

Este arquivo acompanha as tentativas e os avanços do projeto. Os resultados abaixo descrevem o que foi implementado ou verificado; não equivalem a uma avaliação final do classificador.

## 28/10/2026 — Reconstrução do projeto

- **Objetivo:** recuperar o trabalho feito na conversa anterior e deixá-lo executável em notebooks.
- **Motivo expresso:** a conversa compartilhada não fornecia o código diretamente; o texto colado pelo usuário serviria para reconstruí-lo.
- **Mudanças:** foram criados `modelo.ipynb`, `testes_k_svd.ipynb` e instruções da competição.
- **Resultado observado:** o histórico do Git registra a adição dos notebooks e das instruções no commit `511d12b`. O desempenho do modelo não foi medido nesse marco.

## 29/10/2026 — Padronização de imagens e representação SVD

- **Objetivo:** tornar as imagens comparáveis para a SVD e escolher entre escala de cinza e RGB.
- **Motivo expresso:** o usuário queria preservar a proporção ao recortar o centro, redimensionar para 300 × 300 e, no modo RGB, fatorar separadamente os três canais. Ele também queria ajustar `k` no início do código.
- **Mudanças:** os notebooks passaram a recortar e redimensionar as imagens antes da SVD; o modo RGB produz fatores para cada canal. A rede usa as dimensões derivadas de `k`, camadas ocultas com largura reduzida a `floor(2/3)` da anterior, saída sigmoide e MSE.
- **Resultado observado:** as mudanças foram publicadas no commit `2b44996`. Esse registro não contém métricas de classificação.

## 29/10/2026 — Cache, amostragem e tempo de execução

- **Objetivo:** evitar refazer a SVD e tornar os experimentos controláveis.
- **Motivo expresso:** o pré-cálculo era demorado, ocorreu um erro `ArpackError`, e o usuário queria selecionar `N` imagens com metade de cada classe e acompanhar o tempo restante do treino.
- **Mudanças:** foi acrescentada uma alternativa de SVD densa quando o ARPACK falha; os vetores foram separados por `k` e modo de cor em `datasets_processados/`; o modelo recebeu `usar_todos_dados`, `N` e estimativa de tempo restante por época.
- **Resultado observado:** 10.667 vetores já calculados para `k=20` em cinza foram preservados no novo local. O cache e as imagens ficam fora do Git. As alterações foram publicadas no commit `42c6422`; o treinamento completo não foi executado pelo assistente.

## 29/10/2026 — Execução em outro computador

- **Objetivo:** permitir que o outro PC obtenha os arquivos e prepare as imagens localmente.
- **Motivo expresso:** o usuário considerou melhor baixar as imagens no próprio PC de destino e gerar ali os fatores SVD, sem colocar as imagens no GitHub. Depois pediu um único script para configurar, treinar e plotar.
- **Mudanças:** o notebook ganhou uma célula de download pelo Kaggle; depois foram criados `preparar_dados.py`, `treinar_modelo.py`, `requirements.txt` e um README com instruções de execução.
- **Resultado observado:** os scripts e o guia foram publicados no commit `4e7a2ee`. A estrutura e o diff foram verificados, mas o download e o treino não foram executados no outro PC pelo assistente.

## Próximos passos

- Executar `treinar_modelo.py` no outro computador seguindo o `README.md`.
- Confirmar que o download produz `dataset/PetImages/Cat` e `Dog`, que o cache SVD é reutilizado e que o gráfico aparece ao fim do treino.
- Registrar as configurações usadas e as métricas observadas antes de comparar experimentos.
