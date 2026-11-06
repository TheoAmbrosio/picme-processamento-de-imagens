# Evolução da Lisandra — execução1 publicada

Histórico completo de10 épocas, com40 previstas e early stopping de paciência5.
O pacote da autora identifica o checkpoint da **época5 por menor val_loss**.
Na época5: validação90,16%, perda0,223732; na época8: maior acurácia91,08%,
mas perda0,224588, superior à da5. São critérios diferentes.

- [Plot SVG](evolucao_epocas.svg), [PNG](evolucao_epocas.png), [PDF](evolucao_epocas.pdf).
- [CSV oficial por época](historico_epocas.csv).
- [Metadados e disponibilidade](disponibilidade.json).
- [Especificação original do checkpoint](../documentacao_publicada/checkpoint_info.json).
- [Pendências de código e pesos](../PENDENCIAS_ATUAIS.md).

Dados numéricos conferidos contra o CSV bruto publicado, normalização0–1
consistente. Não foram executados novos treinos, inferência real ou teste final.
A associação à época5 é documentada pela autora; o novo arquivo de pesos recebido é compatível com a arquitetura publicada.
