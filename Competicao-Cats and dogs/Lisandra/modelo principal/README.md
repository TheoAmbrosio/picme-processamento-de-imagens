# Lisandra — AlexNetBinary com BatchNorm

Referência vigente: pacote resultados_para_competicao e notebook do GitHub,
commit538a5e9. Código e documentação da autora foram preservados.

- [Método](METODO.md) e [configuração completa declarada](documentacao_publicada/configuracao.json).
- [Registro](REGISTRO_MODELO.md).
- [Arquitetura no formato livro SVG](arquitetura_livro.svg), [PNG](arquitetura_livro.png), [PDF](arquitetura_livro.pdf).
- [Diagrama sequencial](arquitetura.svg).
- [Treino e validação:10 épocas](evolucao%20do%20treinamento/README.md).
- [Banco e normalização](BANCO_DE_DADOS.md).
- [Informações do checkpoint declaradas pela autora](documentacao_publicada/checkpoint_info.json).
- [Pendências verificadas](PENDENCIAS_ATUAIS.md), [conferência JSON](CONFERENCIA_ATUALIZACAO.json).
- [Notebook original](../projeto/CatsandDogs_classification.ipynb).

**Novo checkpoint com BatchNorm recebido e compatível.** Carregamento estrito
e forward artificial aprovados; nenhum treinamento ou teste em imagens reais.
[Pesos atuais](checkpoints/versao_com_batchnorm/alexnet_cats_dogs_best.pth) e
[conferência](checkpoints/versao_com_batchnorm/checkpoint_info.json).
README e sintaxe dos notebooks corrigidos localmente, com autorização do Theo. Patch preservado para revisão; repositório remoto da autora não alterado.
