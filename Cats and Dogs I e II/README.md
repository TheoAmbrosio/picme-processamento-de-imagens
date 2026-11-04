# Cats and Dogs I e II

As [regras da competição](INSTRUCOES_COMPETICAO.md) se aplicam a todos os modelos.

## Organização

```text
residuals networks/
  res_theo18/               # ResNet-18 até layer3
  res_theo34/               # ResNet-34 até layer3
resultados_residuals_networks/
  res_theo18/<execução>/
  res_theo34/<execução>/
dataset/                    # Originais compartilhados e preservados
dataset_ampliado/           # Manifesto e JPEGs compartilhados
```

O antigo `res_theo` sem índice foi retirado dos modelos ativos. Seu código e seus
resultados foram preservados no histórico do ResNet-34. O checkpoint antigo mantém
os parâmetros com que foi treinado e pode ser carregado pelo módulo de inferência
do res_theo34, sem aplicar o dropout da configuração atual.

## Iniciar treinamento

Na raiz `Cats and Dogs I e II`, no Ubuntu:

```bash
bash iniciar_res_theo18.sh
bash iniciar_res_theo34.sh
```

Cada comando inicia somente seu próprio modelo, verifica CUDA, cria uma pasta
exclusiva de execução e imprime os caminhos de log/resultados. O PID também fica
em `resultados_residuals_networks/<modelo>/treinamento.pid`. Nenhum comando acima
foi executado automaticamente durante a reorganização.

As configurações atuais estão em `residuals networks/<modelo>/config_<modelo>.py`.
Dropout é configurável por `dropout_probabilidade`: `0` desativa, `0.3` corresponde
a 30%. As posições fixas selecionadas pelo participante ficam em `dropout_posicoes`.

| Modelo | Batch | Épocas | Épocas iniciais congeladas | Dropout |
|---|---:|---:|---:|---:|
| res_theo18 | 128 | 25 | 10 | 0.3 |
| res_theo34 | 64 | 25 | 10 | 0.3 |

Ambos usam Adam, learning rate 0.001, semente 42 e o manifesto balanceado existente.
Somente fc aprende nas épocas 1–10; o backbone é liberado no início da época 11.
As configurações novas não são atribuídas aos resultados de execuções anteriores.

## Identificação e reutilização dos resultados

Cada execução tem um nome com data/hora, batch, épocas, congelamento e dropout:

- `parametros.json`: configuração completa, modelo, ambiente, commit-base e opções efetivas do otimizador.
- `IDENTIFICACAO.md`: resumo legível da execução.
- `codigo/`: snapshot dos fontes de execuções novas.
- `ultima_epoca.pt`: pesos, configuração, manifesto, histórico, otimizador e scaler.
- `historico.csv`, `graficos_treinamento.png` e `painel_treinamento.html`: métricas e gráficos daquela execução.
- `treinamento.log` e `treinamento.pid`: acompanhamento daquela execução.

A migração preservou o checkpoint antigo sem regravá-lo. `codigo_arquivado/`
contém o código retirado na organização, sem afirmar que corresponde ao início do
treino histórico. Os parâmetros históricos foram recuperados do checkpoint.

Para carregar uma execução específica, use o checkpoint dela, não o de outra
pasta. Da raiz do projeto, por exemplo para inferência em uma imagem:

```bash
PYTHONPATH="$PWD/residuals networks:$PWD" /home/al.theo.ambrosio/Documentos/.venvs/res_theo/bin/python -m res_theo18.inferir_res_theo18 "/caminho/da/execucao/ultima_epoca.pt" "/caminho/imagem.jpg"
```

Para o 34, use `res_theo34.inferir_res_theo34`. A configuração é reconstruída do
checkpoint, inclusive dropout; inferência usa avaliação determinística. Não há
retomada automática de treinamento nem execução automática de teste final.

## Dados, ambiente e testes

O ambiente CUDA preparado fica em `~/Documentos/.venvs/res_theo`. Os iniciadores
configuram os caminhos Python necessários; o nome do ambiente é independente dos
nomes dos modelos. Dependências de cada pacote estão em seu `requirements.txt`.

`python criar_dataset_ampliado.py` continua na raiz e preserva os dados existentes.
RGB 224×224, Cat=0 e Dog=1; todas as 16 variantes no treino e uma imagem por original
na avaliação. Treino/validação são separados antes das ampliações. A configuração
atual usa 80%/20%, sem conjunto separado de teste.

```bash
/home/al.theo.ambrosio/Documentos/.venvs/res_theo/bin/python -m pytest "residuals networks/res_theo34/tests" testes_variantes -q
```

Datasets, ambientes, resultados e checkpoints são ignorados pelo Git. O histórico
de alterações está em [REGISTRO.md](REGISTRO.md).
