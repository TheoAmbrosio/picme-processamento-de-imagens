# Evolução de treino e validação — Lisandra

CSV recebido da autora via Theo: **10 épocas completas**, com perda,
acurácia e erro de treino/validação. Arquivo original preservado; acurácias
convertidas de porcentagem para fração0–1 somente na cópia normalizada.

- [Gráfico SVG](evolucao_epocas.svg), [PNG](evolucao_epocas.png), [PDF](evolucao_epocas.pdf).
- [Histórico normalizado CSV](historico_epocas.csv).
- [CSV original recebido](../../resultados_recebidos/training_history_Lisandra.csv).
- [Fonte, integridade e resumo](disponibilidade.json).

Maior acurácia de validação:91.08%, época8.
Menor perda de validação:0.223732, época5.
Última época: treino96.83%, validação90.32%.

## Associação aos pesos e configuração

O CSV tem10 épocas; o notebook publicado configura5 e só mostra parte da
primeira. Não foi presumido que os pesos enviados sejam da melhor época
por acurácia ou por perda deste CSV, nem que toda a configuração seja idêntica.
Essa associação depende de confirmação da autora. Nenhuma métrica de teste final
foi incluída. Não foram executados treino ou inferência em imagens reais.
