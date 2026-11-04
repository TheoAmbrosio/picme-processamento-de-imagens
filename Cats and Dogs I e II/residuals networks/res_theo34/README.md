# res_theo34

Regras: [INSTRUCOES_COMPETICAO.md](../../INSTRUCOES_COMPETICAO.md).
Organização, parâmetros e comandos: [guia principal](../../README.md).

Código em `residuals networks/res_theo34/`. Resultados exclusivamente em
`resultados_residuals_networks/res_theo34/<execução>/`, fora do pacote.

A probabilidade de dropout é configurável em `config_res_theo34.py`:
`dropout_probabilidade=0` desativa e `0.3` aplica dropout comum de 30% nos pontos
fixos de `dropout_posicoes`. As somas residuais não recebem hooks diretamente.
A seleção atual contém todas as operações primitivas aceitas, incluindo fc.
O dropout é desativado em avaliação, e seus parâmetros são salvos no checkpoint.

Execute da raiz `Cats and Dogs I e II`:

```bash
bash iniciar_res_theo34.sh
```

Batch atual: 64. Total: 25 épocas. Backbone
congelado nas primeiras 10 épocas e liberado no início da época 11.
Cada execução salva `parametros.json`, `IDENTIFICACAO.md`, fontes, log e checkpoint
próprios; alterações posteriores de configuração não mudam modelos já salvos.
