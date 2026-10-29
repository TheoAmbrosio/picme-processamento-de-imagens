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

```powershell
.\.venv\Scripts\python.exe treinar_modelo.py
```

Na primeira execução, o script tenta baixar o [Microsoft Cats vs. Dogs no Kaggle](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset). Pode ser necessário autenticar-se no Kaggle. Ele espera encontrar as imagens em `dataset/PetImages/Cat` e `dataset/PetImages/Dog`. Se ambas as pastas já contêm imagens, pula o download.

Em seguida, verifica as imagens legíveis, escolhe a amostra configurada, prepara os vetores SVD e divide os dados em treino e avaliação. O cálculo inicial da SVD pode demorar. Durante o treino, cada época imprime o MSE de treino, o MSE de avaliação e uma estimativa de tempo restante. Ao final, abre o gráfico de MSE; em um ambiente sem interface gráfica, essa janela pode não aparecer.

## Arquivos e cache

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
