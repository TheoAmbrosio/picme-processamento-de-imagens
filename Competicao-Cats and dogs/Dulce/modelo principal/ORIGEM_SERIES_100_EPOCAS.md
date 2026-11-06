# Origem das séries de100 épocas

Esclarecimento da autora, transmitido pelo Theo em06/11/2026:

> As séries de 100 épocas são do primeiro modelo (cats_and_dogs.ipynb, SGD, lr 1e-2). No historico_treino.json, acc_val_modelo_fechado é a execução original (só validação) e repeticao é uma reexecução com a mesma configuração e seed, feita depois pra ter perda e acurácia de treino e validação.

| Chave em historico_treino.json | Execução | Métricas disponíveis |
|---|---|---|
| acc_val_modelo_fechado | Original do primeiro modelo | Acurácia de validação por época |
| repeticao | Nova execução do primeiro modelo, mesma configuração e seed | Perda e acurácia de treino e validação por época |

Ambas referem-se a cats_and_dogs.ipynb, SGD com LR0.01. A repetição foi
feita depois para produzir o histórico mais completo. Não representa outras
100 épocas continuadas a partir do primeiro treino, nem permite atribuir
as perdas e acurácias de treino da repetição à execução original.

Fonte desta associação: confirmação da autora; dados numéricos permanecem
os publicados no JSON original. Não foi executado novo treinamento aqui.
