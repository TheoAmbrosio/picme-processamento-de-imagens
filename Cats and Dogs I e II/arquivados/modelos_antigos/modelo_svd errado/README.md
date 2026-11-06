# Modelo SVD antigo

Documentação histórica preservada. Os dados compartilhados ficam na pasta pai; o modelo atual está em `../res_theo/`.
As [instruções gerais da competição](../INSTRUCOES_COMPETICAO.md) se aplicam aos dois modelos.

# Cats and Dogs I e II

Este projeto classifica imagens de gatos e cachorros usando fatores SVD como entrada de uma rede neural. O caminho mais simples é executar `treinar_modelo.py`; os notebooks continuam disponíveis para exploração interativa.

## 1. Baixar o projeto

No PowerShell do outro PC, com Git e Python instalados:

```powershell
git clone --branch cats-and-dogs https://github.com/TheoAmbrosio/picme-processamento-de-imagens.git
cd 'picme-processamento-de-imagens/Cats and Dogs I e II'
```

Use Python 3.10 ou mais recente. Se `python` não for reconhecido no Windows, tente `py` nos comandos seguintes.

## 2. Instalar as dependências

No mesmo diretório:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

O ambiente virtual evita misturar os pacotes do projeto com os de outros trabalhos. Depois da instalação, use `.\.venv\Scripts\python.exe` para executar o script.

## 3. Configurar o experimento

Abra `treinar_modelo.py` e altere os valores no bloco **Configurações editáveis**:

| Variável | Função |
| --- | --- |
| `k_modelo` | Número de componentes SVD por canal; inteiro de 1 a 299. |
| `escala_cinza` | `True` usa um canal; `False` usa RGB e fatora os três canais. |
| `usar_todos_dados` | `True` usa todas as imagens legíveis; `False` usa apenas `N`. |
| `N` | Total de imagens quando `usar_todos_dados=False`; deve ser par, metade gatos e metade cachorros. |
| `semente_amostragem` | Controla a seleção de imagens e a divisão treino/avaliação. |
| `fracao_treino` | Fração de imagens para treino; o restante vai para avaliação. |
| `taxa_aprendizado` | Taxa de aprendizado do Adam. |
| `epocas` | Quantas passagens de treino serão feitas. |
| `tamanho_batch` | Quantas imagens entram em cada lote de treino. |

Os valores iniciais do script são apenas configurações de partida. Para uma primeira execução curta, reduza `N` e `epocas` e deixe `usar_todos_dados=False`.

## 4. Executar

A perda de otimização agora é `MSE + mean(alpha / (abs(centro - previsao) + epsilon_penalizacao)) + alpha2 * (0.5 - previsoes.mean())**2`.
O termo `alpha2` usa a média do lote atual, sem `detach`, e participa do
gradiente. Ele favorece média 0,5 em cada lote, mas não garante dois extremos:
um lote inteiro com previsões 0,5 também zera esse termo. Lotes com classes
desbalanceadas podem entrar em conflito com essa preferência.

Configuração inicial experimental: taxa 0,001, batch 64, alpha 0,00001,
alpha2 0,001 e epsilon 0,001. Não são valores ótimos comprovados.
O centro é 0,5 na primeira época; nas seguintes, é a média das previsões
observadas durante o treino da época anterior, ponderada pelo número de imagens
de cada lote. Não usa dados da avaliação nem participa do gradiente. O centro
fica fixo em cada época. O log mostra o centro usado e a média do treino atual.
Configure `alpha` no início de `treinar_modelo.py`;
`0` desativa. `epsilon_penalizacao` evita divisão por zero e torna a
fórmula uma aproximação da expressão solicitada. A média é calculada por lote.
O terminal separa MSE de treino, MSE de avaliação, penalização média e perda total
de treino. O gráfico continua mostrando somente MSE.
Em cada época, o terminal também mostra os dois intervalos fixos de largura 0,1
com mais previsões na avaliação, com contagem e porcentagem sobre toda a avaliação.
Exemplo: `intervalos avaliação=[0.4, 0.5): 70 (70.0%); [0.5, 0.6): 30 (30.0%)`.
Os limites inferiores são inclusivos e os superiores exclusivos, exceto 1,0,
incluído no último intervalo. Em empate, vem primeiro o menor intervalo. Se
apenas um intervalo estiver ocupado, só ele será mostrado. Esse resumo descreve
as saídas da rede, não a proporção de acertos.
O campo `MSE avaliação arredondado` usa classes 0 para previsões menores que
0,5 e 1 para previsões maiores ou iguais a 0,5. Com rótulos binários, essa
métrica equivale à fração de classificações erradas (`1 - acurácia`). O MSE
contínuo continua no log; a perda de treinamento usa as saídas contínuas.

A penalização favorece previsões distantes da média anterior, mesmo quando estão erradas;
isso não garante ocupar os dois extremos: pode haver oscilação entre eles.
Seu efeito sobre a classificação precisa ser medido. Exatamente no centro, o
gradiente de `abs` adotado pelo PyTorch é zero, portanto o termo não garante
escapar desse ponto. Perto dele, os gradientes podem ser grandes; o valor inicial
de `alpha` não foi validado por treinamento.

```powershell
.\.venv\Scripts\python.exe treinar_modelo.py
```

Na primeira execução, o script tenta baixar o [Microsoft Cats vs. Dogs no Kaggle](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset). Pode ser necessário autenticar-se no Kaggle. Ele espera encontrar as imagens em `dataset/PetImages/Cat` e `dataset/PetImages/Dog`. Se ambas as pastas já contêm imagens, pula o download.

Em seguida, verifica as imagens legíveis, escolhe a amostra configurada, prepara os vetores SVD e divide os dados em treino e avaliação. O cálculo inicial da SVD pode demorar. Durante o treino, cada época imprime o MSE de treino, o MSE de avaliação e uma estimativa de tempo restante. Ao final, abre o gráfico de MSE; em um ambiente sem interface gráfica, essa janela pode não aparecer.

## Arquivos e cache

Ao concluir o treino, também é exibido um histograma das previsões contínuas
na avaliação final, com intervalos de largura 0,1 e contagem sobre cada barra.
Ele é salvo em `frequencia_resultados.png` na pasta do projeto, inclusive
quando não há interface gráfica. Cada execução concluída substitui esse PNG.
O log de intervalos por época continua disponível.

### Executar neste Linux

Abra o Terminal com `Ctrl + Alt + T` e execute:

```bash
cd '/home/al.theo.ambrosio/Documentos/picme-processamento-de-imagens/Cats and Dogs I e II'
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -u treinar_modelo.py
```

Não é necessário ativar a `.venv`: o comando usa seu Python diretamente.
Para interromper, pressione `Ctrl + C`. O script usa CUDA automaticamente se
disponível; procure `Treinando em cuda` na saída. A preparação SVD usa CPU e
reaproveita o cache. Para consultar a GPU: `nvidia-smi`.

Após a divisão dos dados, `normalizacao.py` estima média e desvio padrão
populacional de cada entrada SVD usando apenas o treino. A transformação
`(x - media) / max(desvio, 1e-6)` é aplicada ao treino e à avaliação com as mesmas
estatísticas. Os rótulos continuam em 0 e 1; os vetores originais no cache são
preservados. As estatísticas são recalculadas a cada execução. O script atual
não salva o modelo nem essas estatísticas; para inferência futura, ambos precisam
ser persistidos juntos.

| Caminho | Conteúdo |
| --- | --- |
| `treinar_modelo.py` | Configuração, rede, treino, avaliação e gráfico. |
| `preparar_dados.py` | Download, recorte, redimensionamento, SVD e leitura do cache. |
| `modelo.ipynb` | Versão interativa do modelo. |
| `testes_k_svd.ipynb` | Experimentos com valores de `k`. |
| `dataset/PetImages/` | Imagens baixadas; não é versionado. |
| `datasets_processados/modelo/300x300/k_<k>/cinza/` ou `rgb/` | Vetores SVD salvos separadamente por `k` e modo de cor; não são versionados. |

Se você rodar novamente com o mesmo `k` e modo de cor, o script reaproveita os vetores salvos cuja imagem original não mudou. Trocar `N` altera a amostra usada no treino, sem apagar o cache de outras imagens.

## Se algo falhar

- `python` não é reconhecido: instale Python e marque a opção de adicioná-lo ao PATH, ou use `py` para criar o ambiente virtual.
- `ModuleNotFoundError`: execute o comando de instalação usando o Python da `.venv` mostrado acima.
- Erro de download do Kaggle: confira conexão, login e acesso ao dataset. Se necessário, baixe e extraia manualmente até obter `dataset/PetImages/Cat` e `Dog`.
- Dataset incompleto: confira o conteúdo de `dataset/PetImages` antes de executar novamente. O script não sobrescreve essa pasta automaticamente.

As imagens e os vetores SVD são gerados no próprio PC e não são incluídos no repositório Git.
