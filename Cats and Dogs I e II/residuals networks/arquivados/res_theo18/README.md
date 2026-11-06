# res_theo18

Regras: [INSTRUCOES_COMPETICAO.md](../../INSTRUCOES_COMPETICAO.md).
Organização, parâmetros e comandos: [guia principal](../../README.md).

Código em `residuals networks/res_theo18/`. Resultados exclusivamente em
`resultados_residuals_networks/res_theo18/<execução>/`, fora do pacote.

A probabilidade de dropout é configurável em `config_res_theo18.py`:
`dropout_probabilidade=0` desativa e `0.3` aplica dropout comum de 30% nos pontos
fixos de `dropout_posicoes`. As somas residuais não recebem hooks diretamente.
A seleção atual contém todas as operações primitivas aceitas, excluindo a primeira convolução (`conv1`) e a camada final (`fc`).
O dropout é desativado em avaliação, e seus parâmetros são salvos no checkpoint.

Execute da raiz `Cats and Dogs I e II`:

```bash
bash iniciar_res_theo18.sh
```

Batch atual: 128. Total: 25 épocas. Backbone
congelado nas primeiras 4 épocas e liberado no início da época 5.
Cada execução salva `parametros.json`, `IDENTIFICACAO.md`, fontes, log e checkpoint
próprios; alterações posteriores de configuração não mudam modelos já salvos.

Configuração atual de dropout: 0.1 (10%), sem aplicação após `conv1` e `fc`.
As demais posições permanecem selecionadas. Alterações valem nas novas execuções.
