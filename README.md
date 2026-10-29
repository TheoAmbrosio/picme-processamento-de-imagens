# Picme: processamento de imagens

Repositório dos trabalhos de processamento de imagens.

O projeto **Cats and Dogs I e II** está na branch `cats-and-dogs`. Os notebooks e as instruções da competição ficam na pasta `Cats and Dogs I e II` dessa branch.

As imagens do dataset e os fatores SVD gerados localmente não são versionados.

## Rodar Cats and Dogs em outro computador

1. Clone a branch: `git clone --branch cats-and-dogs https://github.com/TheoAmbrosio/picme-processamento-de-imagens.git`.
2. Abra `Cats and Dogs I e II/modelo.ipynb` no Jupyter, tendo essa pasta como diretório de trabalho.
3. Instale as dependências no ambiente do notebook: `%pip install numpy pillow scipy torch matplotlib kagglehub` e reinicie o kernel.
4. Execute as células em ordem. A célula inicial baixa o espelho [Microsoft Cats vs. Dogs no Kaggle](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset) se `dataset/PetImages/Cat` e `Dog` ainda não existirem. Pode ser necessário autenticar-se no Kaggle.
5. Ajuste `k_modelo`, `escala_cinza`, `usar_todos_dados` e `N` na célula de configuração. A preparação cria o cache SVD localmente em `datasets_processados/`; ele é reutilizado em execuções posteriores.

O download depende da disponibilidade e da estrutura do dataset no Kaggle. As imagens, os vetores SVD e as credenciais não são enviados ao GitHub.

## Execução por script

No diretório `Cats and Dogs I e II`, instale as dependências com `python -m pip install -r requirements.txt`. Edite os valores no início de `treinar_modelo.py` e execute `python treinar_modelo.py`. Esse comando baixa o dataset quando necessário, prepara ou reutiliza os vetores SVD, treina e mostra o gráfico de MSE. O módulo `preparar_dados.py` contém o download, o pré-processamento e o cache; `modelo.ipynb` continua disponível para uso interativo.
