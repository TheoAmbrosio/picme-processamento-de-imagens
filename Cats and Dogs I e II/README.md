# Cats and Dogs — ResTheo34 Aug8

O modelo principal é `residuals networks/res_theo34_aug8/`.
Todos os materiais consolidados estão em [modelo principal/](modelo%20principal/README.md).
Leia [LINHAGEM_MODELO_PRINCIPAL.md](LINHAGEM_MODELO_PRINCIPAL.md) para identificar
os treinos-base, os checkpoints de origem, as alterações em cada etapa e os dados.
As [regras da competição](INSTRUCOES_COMPETICAO.md) continuam aplicáveis.

## Organização

Na raiz ficam o iniciador do Aug8, regras, registro, linhagem, dados e módulos
compartilhados necessários. Tentativas antigas e scripts históricos estão em
[arquivados/](arquivados/README.md).

- `residuals networks/res_theo34_aug8/`: modelo principal e ampliação8×.
- `residuals networks/res_theo34/`: referência histórica e testes.
- `residuals networks/res_theo34_repeticao/`: repetição e checkpoint da época33.
- `residuals networks/res_theo34_finetuning/`: primeiro ramo de ajuste fino.
- `residuals networks/arquivados/`: demais modelos e iniciadores históricos.
- `resultados_residuals_networks/`: principal e seus antecedentes documentados.
- `resultados_residuals_networks/arquivados/`: outros resultados, sem exclusões.
- `docs/diagramas/arquitetura_res_theo34_aug8.svg`: arquitetura atual.

## Execução

Uma nova execução, sem retomar a já existente:

```bash
bash iniciar_res_theo34_aug8.sh
```

O iniciador confirma CUDA, cria resultados próprios e imprime o log.
Parâmetros efetivos, fontes, histórico, gráficos, último e melhor checkpoint
ficam na pasta da execução. Confira o estado do treino antes de iniciar outro.

Gráfico grande das etapas: `resultados_residuals_networks/res_theo34_aug8/trajetoria_treinamentos.png`
e PDF correspondente. Atualização: execute `docs/diagramas/gerar_res_theo34_aug8.py`
com o Python de `~/Documentos/.venvs/res_theo/`.

Datasets, resultados, ambientes e checkpoints permanecem fora do Git.
O histórico de alterações está em [REGISTRO.md](REGISTRO.md).
