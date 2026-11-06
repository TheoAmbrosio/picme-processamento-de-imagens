# Resultados da Classificação Cães e Gatos — AlexNetBinary (Lisandra)

Este diretório reúne todos os arquivos padronizados para comparação de desempenho entre participantes, extraídos estritamente dos artefatos reais da execução realizada no projeto `cats_vs_dogs_Lisandra`.

---

## 📁 Estrutura de Arquivos

| Arquivo | Descrição |
| :--- | :--- |
| **`historico_epocas.csv`** | Tabela contendo 1 linha por época (10 épocas), com perda e acurácia de treino e validação (acurácia normalizada em escala 0 a 1) e taxa de aprendizado. |
| **`configuracao.json`** | Especificação técnica completa: arquitetura `AlexNetBinary`, camadas, posições do dropout (0.5), otimizador Adam, batch size, splits e pré-processamento. |
| **`checkpoint_info.json`** | Metadados do checkpoint `alexnet_cats_dogs_best.pth` (223 MB), comprovando a qual época pertence (Época 5) e o critério de seleção adotado. |
| **`curvas_aprendizado.png`** | Gráfico em alta resolução (300 DPI) com 2 painéis (Perda e Acurácia) comparando treino e validação, com linha vertical destacando o melhor checkpoint. |
| **`curvas_aprendizado.svg`** | Versão vetorial escalável (SVG) do gráfico de curvas de aprendizado. |
| **`treinamento.log`** | Log textual cronológico das 10 épocas executadas e do acionamento do Early Stopping. |
| **`CatsandDogs_classification_saidas.ipynb`** | Notebook Jupyter contendo todas as células e saídas salvas correspondentes ao treinamento. |
| **`alexnet_cats_dogs_best.pth`** | Link simbólico apontando para o arquivo de pesos binários gravado no diretório raiz do projeto. |

---

## 🔍 Origem e Rastreabilidade das Informações

1. **Histórico por Época (`historico_epocas.csv`):**
   * Extraído diretamente do arquivo `training_history_Lisandra.csv`, gerado pela execução do treinamento.
   * As perdas e acurácias representam a **época inteira**, ponderadas exatamente pelo tamanho de cada lote (`loss.item() * inputs.size(0)` e `(predictions == targets).sum()`, normalizados pelo total do conjunto: 19.998 imagens no treino e 2.500 na validação).
   * A acurácia foi convertida para fração decimal de 0 a 1 (dividindo por 100).

2. **Checkpoint `alexnet_cats_dogs_best.pth` (Época e Critério):**
   * **Critério de Seleção:** Menor perda de validação (*lowest validation loss*).
   * **Comprovação da Época 5 (Sem inferência pelo nome):**
     * O código executou um loop com Early Stopping monitorando `val_loss` com paciência de 5 épocas:
       - Época 1: `val_loss = 0.434171`
       - Época 2: `val_loss = 0.336772`
       - Época 3: `val_loss = 0.277607`
       - Época 4: `val_loss = 0.282988`
       - **Época 5: `val_loss = 0.223732` (MÍNIMO GLOBAL DA SÉRIE)**
       - Época 6: `val_loss = 0.323556` (sem melhora: contador 1/5)
       - Época 7: `val_loss = 0.314086` (sem melhora: contador 2/5)
       - Época 8: `val_loss = 0.224588` (sem melhora: contador 3/5, pois 0.224588 > 0.223732)
       - Época 9: `val_loss = 0.513711` (sem melhora: contador 4/5)
       - Época 10: `val_loss = 0.316110` (sem melhora: contador 5/5 -> Parada Antecipada acionada)
     * Após a época 10, o script determinístico executou `model.load_state_dict(best_model_weights)`, restaurando o estado exato da **Época 5**, e o gravou no arquivo `alexnet_cats_dogs_best.pth`.
     * Métricas registradas da Época 5:
       - **Perda de Treino:** 0.187761
       - **Acurácia de Treino:** 0.920692 (92.07%)
       - **Perda de Validação:** 0.223732
       - **Acurácia de Validação:** 0.901600 (90.16%)

---

## ⚠️ Informações Não Registradas / Limitações (Transparência)

Em respeito estrito à diretriz *"Não invente dados nem apresente uma reexecução como se fosse o treino original"*:

* **F1 Score de Treino e Validação:** Não foi implementado nem computado no loop de treinamento original, constando como `None` (vazio) no histórico.
* **Semente Aleatória (`seed`):** Não foi fixada chamada a `torch.manual_seed()` no notebook original.
* **Métricas de Teste Final:** Conforme a instrução expressa (*"Não execute o teste final para preencher essas informações"*), o conjunto de teste de 2.500 imagens não foi inferido para preencher campos do treinamento por época.
* **Learning Rate Scheduler:** O otimizador Adam utilizou taxa de aprendizado estática de `0.0001` sem decaimento agendado.
