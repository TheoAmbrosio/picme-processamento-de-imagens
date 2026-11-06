# Bandeira — CNN V4 (código vigente)

Identificada pela classe CNN em `src/model.py` e pelo loop V4 em `src/train.py`.
O README agora distingue a arquitetura base V1–V3 da V4; o diagrama segue a V4.

- [Método](METODO.md) e [parâmetros](parametros.json).
- [Arquitetura SVG](arquitetura.svg).
- [Gráficos salvos V1–V4](graficos/README.md).
- [Registro original V4](../projeto/ChangesV4.md).
- [Código original](../projeto/src/) e [análise V4](../projeto/Analysis/resultsV4.ipynb).

Os históricos CSV, predições e checkpoints V1–V4 estão publicados no clone atualizado.
Os gráficos numéricos ficam na seção de evolução; figuras originais dos notebooks
também foram preservadas. Nenhum treinamento ou teste final foi executado.

- [Checkpoint V4](../projeto/data/best_model_4.0.pth).
- [Histórico V4](../projeto/data/training_history_4.0.csv).
- [Predições de validação V4](../projeto/data/val_predictions_4.0.csv).
- [Conferência dos artefatos](CONFERENCIA_ATUALIZACAO.json).
- [Conferência completa e ponto de pré-processamento a confirmar](CONFERENCIA_ATUALIZACAO.md).

## Banco de dados do treinamento

[Origem, quantidades, divisão e transformações por modelo](BANCO_DE_DADOS.md).

## Evolução de treino e validação por época

[Gráficos, tabelas CSV e fontes dos históricos](evolucao%20do%20treinamento/README.md).
