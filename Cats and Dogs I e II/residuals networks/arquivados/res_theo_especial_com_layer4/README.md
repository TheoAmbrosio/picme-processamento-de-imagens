# res_theo_especial

Regras: [INSTRUCOES_COMPETICAO.md](../../INSTRUCOES_COMPETICAO.md).
Organização, parâmetros e comandos: [guia principal](../../README.md).

Código em `residuals networks/res_theo_especial/`. Resultados exclusivamente em
`resultados_residuals_networks/res_theo_especial/<execução>/`, fora do pacote.

A probabilidade de dropout é configurável em `config_res_theo_especial.py`:
`dropout_probabilidade=0` desativa e `0.3` aplica dropout comum de 30% nos pontos
fixos de `dropout_posicoes`. As somas residuais não recebem hooks diretamente.
A seleção atual contém todas as operações primitivas aceitas, excluindo a primeira convolução (`conv1`) e a camada final (`fc`).
O dropout é desativado em avaliação, e seus parâmetros são salvos no checkpoint.

Execute da raiz `Cats and Dogs I e II`:

```bash
bash iniciar_res_theo_especial.sh
```

Arquitetura completa da ResNet-18, com dois blocos por estágio e canais
`64 → 32 → 64 → 128 → 256`; a `layer4` é mantida e a saída é `Linear(256,1)`.
A conv1 7x7 usa pesos ImageNet, fica congelada nas épocas 1–5 e é liberada na 6.
As demais convoluções e lineares usam He normal
(`kaiming_normal_`, variância `2/fan_out`),
vieses em zero e BatchNorm com escala 1/deslocamento 0.

Batch atual: 16. Total: 60 épocas.
Os dados originais são usados uma vez e esticados deterministicamente para RGB 224×224.
Cada execução salva `parametros.json`, `IDENTIFICACAO.md`, fontes, log e checkpoint
próprios; alterações posteriores de configuração não mudam modelos já salvos.

Configuração atual de dropout: 0.1 (10%), sem aplicação após `conv1` e `fc`.
As demais posições permanecem selecionadas. Alterações valem nas novas execuções.
