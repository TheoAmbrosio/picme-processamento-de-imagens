# Cats and Dogs I e II

O modelo atual é **res_theo**. Código, configurações, testes e instruções estão em
[res_theo/README.md](res_theo/README.md). A abordagem anterior está arquivada em
[modelo_svd errado/](<modelo_svd errado/>).

## Usar em outro PC

Com Git e Python 3.10 ou mais recente:

```powershell
git clone --branch cats-and-dogs https://github.com/TheoAmbrosio/picme-processamento-de-imagens.git
cd 'picme-processamento-de-imagens/Cats and Dogs I e II'
```

Se já clonou, atualize a branch no repositório com `git pull origin cats-and-dogs`.
No PC com GPU, instale PyTorch/Torchvision com CUDA conforme o
[seletor oficial](https://pytorch.org/get-started/locally/) e depois:

```powershell
py -m pip install -r res_theo/requirements.txt
```

Preencha em `res_theo/config_res_theo.py` o otimizador, learning rate, batch size,
épocas e as proporções de validação/teste. Essas escolhas são obrigatórias.

```powershell
py .\criar_dataset_ampliado.py
py -m res_theo.treinar_res_theo
```

O gerador obtém os originais do Kaggle quando necessário. Os caminhos compartilhados
são `dataset/PetImages/Cat` e `Dog`, e `dataset_ampliado/224x224/`. Cada original de
treino recebe 16 variantes RGB de 224×224; validação e teste recebem uma versão
determinística, com separação por conteúdo antes da ampliação. Os rótulos são
`Cat=0` e `Dog=1`. Dados e checkpoints não são enviados ao GitHub.

Para gerar apenas o dataset, as dependências estão em `requirements-dataset.txt`.
O guia do modelo descreve autenticação no Kaggle, retomada, inferência,
avaliação do teste final e as opções de CUDA.

## Organização

- `res_theo/`: modelo atual e instruções de execução.
- `modelo_svd errado/`: código e resultados legíveis do modelo antigo.
- `criar_dataset_ampliado.py`: preparação dos dados compartilhados.
- `REGISTRO.md`: histórico das alterações e verificações.
