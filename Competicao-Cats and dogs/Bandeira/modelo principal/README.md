# Bandeira — CNN V4 Final

Modelo atual: fine-tuning do checkpoint da época 30 da V4, publicado no commit `bd92c13`.
A arquitetura foi preservada: cinco blocos 32→64→128→256→512, média global e uma saída.

- [Método e fases](METODO.md) · [Registro](REGISTRO_MODELO.md).
- [Parâmetros finais](parametros.json) · [Parâmetros da V4 base](parametros_v4_base.json).
- [Arquitetura no formato livro](arquitetura_livro.svg).
- [Evolução por época e painel geral](evolucao%20do%20treinamento/README.md).
- [Banco e transformações](BANCO_DE_DADOS.md).
- [Conferência final](CONFERENCIA_FINAL.json) · [Pontos de atenção](CONFERENCIA_ATUALIZACAO.md).
- [Checkpoint final](../projeto/data/best_model_4.0_Final.pth).
- [Histórico final](../projeto/data/training_history_4.0_Final.csv).
- [Predições salvas](../projeto/data/val_predictions_4.0_Final.csv).
- [Código do fine-tuning](../projeto/src/fine_tuning.py).

Checkpoint escolhido por menor perda: época 10 do fine-tuning, acurácia 94,76%,
BCE 0,13356867 e F1 0,94726248. Maior acurácia da etapa: 94,80% nas épocas 3 e 9.
V1–V4 continuam preservadas como execuções anteriores; a etapa Final deriva apenas da V4.
Sem novo treinamento ou avaliação no conjunto de imagens.
