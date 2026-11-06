# Evolução do treinamento — Theo

Perda e acurácia de treino/validação por época, conforme fontes salvas.
## Visão geral até o Aug8

- [Plot geral SVG com divisões entre etapas](visao%20geral/trajetoria_unificada.svg).
- [Plot geral PNG](visao%20geral/trajetoria_unificada.png).
- [Plot geral PDF](visao%20geral/trajetoria_unificada.pdf).
- [Painéis por etapa](visao%20geral/trajetoria_treinamentos.png).
- [Tabela completa, identificando a etapa e a época](visao%20geral/historico_etapas.csv).
- [Linhagem e origem dos pesos](../documentacao/LINHAGEM_MODELO_PRINCIPAL.md).

```text
1. Referência histórica (referência de método, não origem dos pesos)
2. Repetição: nova partida ImageNet → checkpoint da época33
   ├── 3. Primeiro fine-tuning:16 épocas, ramo independente
   └── 4. Aug8:30 épocas, modelo principal
```

As quatro partes estão na apresentação; isso não significa que todos os
pesos passaram por elas em sequência. O eixo do plot geral organiza as etapas
na apresentação, não representa uma única contagem de épocas de pesos herdados.
A continuação37–50 retirada anteriormente continua fora desta seleção.


## 01_referencia_historica

Épocas disponíveis:21. Referência metodológica; interrompida na21. Não forneceu pesos à repetição.

- [Gráfico SVG](<01_referencia_historica/graficos.svg>) · [PNG](<01_referencia_historica/graficos.png>) · [PDF](<01_referencia_historica/graficos.pdf>).
- [Tabela CSV](<01_referencia_historica/historico.csv>) · [Origem dos dados](<01_referencia_historica/fonte.json>).

## 02_repeticao

Épocas disponíveis:36. Nova partida ImageNet;36 épocas observadas. Checkpoint33 origina os dois ramos seguintes.

- [Gráfico SVG](<02_repeticao/graficos.svg>) · [PNG](<02_repeticao/graficos.png>) · [PDF](<02_repeticao/graficos.pdf>).
- [Tabela CSV](<02_repeticao/historico.csv>) · [Origem dos dados](<02_repeticao/fonte.json>).

## 03_primeiro_finetuning

Épocas disponíveis:16. Ramo independente do checkpoint33 da repetição;16 épocas. Não é origem dos pesos do Aug8.

- [Gráfico SVG](<03_primeiro_finetuning/graficos.svg>) · [PNG](<03_primeiro_finetuning/graficos.png>) · [PDF](<03_primeiro_finetuning/graficos.pdf>).
- [Tabela CSV](<03_primeiro_finetuning/historico.csv>) · [Origem dos dados](<03_primeiro_finetuning/fonte.json>).

## Aug8

Épocas disponíveis:30. Novo ramo direto do checkpoint33 da repetição;30 épocas. Modelo principal.

- [Gráfico SVG](<Aug8/graficos.svg>) · [PNG](<Aug8/graficos.png>) · [PDF](<Aug8/graficos.pdf>).
- [Tabela CSV](<Aug8/historico.csv>) · [Origem dos dados](<Aug8/fonte.json>).

