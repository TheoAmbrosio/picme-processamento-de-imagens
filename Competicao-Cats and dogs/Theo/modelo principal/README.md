# Modelo principal — ResTheo34 Aug8

Esta pasta reúne a referência consolidada do Aug8, concluído em05/11/2026.
Os experimentos posteriores arquivados não fazem parte deste modelo.

## Comece por aqui

- [Como foi treinado e resultados](documentacao/FECHAMENTO_AUG8.md).
- [Etapas anteriores e origem dos pesos](documentacao/LINHAGEM_MODELO_PRINCIPAL.md).
- [Registro do modelo](REGISTRO_MODELO.md).
- [Arquitetura SVG no estilo do livro](arquitetura/arquitetura_res_theo34_aug8_livro.svg).
- [Arquitetura SVG com resíduos explícitos](arquitetura/arquitetura_res_theo34_aug8.svg).
- [Gráfico geral SVG com mudanças entre etapas](graficos/trajetoria_unificada.svg).
- [Gráfico geral PNG](graficos/trajetoria_unificada.png).
- [Parâmetros efetivamente usados](treinamento_aug8/parametros.json).
- [Histórico das30 épocas](treinamento_aug8/historico.csv).
- [Gráficos somente do Aug8](treinamento_aug8/graficos_treinamento.png).
- [Amostras das8 ampliações](graficos/amostras_aug8.png).

## Método

ResNet-34 pré-treinada no ImageNet, layer4 removida; permanecem layer1/2/3
com3/4/6 blocos e64/128/256 canais. Média global, Flatten, dropout10% e
classificador Linear256→1. BCEWithLogitsLoss; Cat=0 e Dog=1.

Parte dos pesos da época33 da repetição histórica. Nova fase de30 épocas,
batch64, AdamW com weight_decay0.01. Grupos: conv1/bn1/layer1/layer2 com
LR1e-5; layer3 com5e-5; fc com1e-4. CosineAnnealingLR T_max30,
eta_min1e-7. Toda a rede treinável desde o primeiro batch. AMP e semente42.

Divisão balanceada80/20 antes de ampliar;159984 amostras virtuais de treino
(8 por original),5000 imagens de validação. Ruído, pixels brancos, quadrados
pretos e esticamentos: [detalhes](documentacao/AMPLIACAO.md).
Validação determinística histórica, normalização ImageNet; teste final não executado.

Melhor validação: **98,32% na época29**. Última época:98,22%.
F1 não era calculado nesta etapa; não foi estimado retrospectivamente.

## Pesos e dados originais

Os arquivos grandes continuam no lugar, sem cópias redundantes:

- [Melhor checkpoint](../../../Cats%20and%20Dogs%20I%20e%20II/resultados_residuals_networks/res_theo34_aug8/20261105_163938_589179_bs64_ep30_cong0ep_do10_dadosamp_pre/melhor_modelo.pt).
- [Último checkpoint](../../../Cats%20and%20Dogs%20I%20e%20II/resultados_residuals_networks/res_theo34_aug8/20261105_163938_589179_bs64_ep30_cong0ep_do10_dadosamp_pre/ultima_epoca.pt).
- [Código principal](../../../Cats%20and%20Dogs%20I%20e%20II/residuals%20networks/res_theo34_aug8/).
- [Manifestos e divisão](../../../Cats%20and%20Dogs%20I%20e%20II/dataset_ampliado/224x224/divisao_balanceada/).
- [Imagens originais](../../../Cats%20and%20Dogs%20I%20e%20II/dataset/PetImages/).

`treinamento_aug8/codigo/` preserva os fontes associados à execução;
`historico_das_etapas/` contém os dados e gráficos das etapas da linhagem.
O gráfico geral inclui a referência metodológica e a linhagem do Aug8; não representa todos
como uma única sequência de transferência de pesos. Consulte a linhagem.

Esta pasta é uma cópia organizada dos artefatos existentes. Caminhos absolutos
nos metadados e documentos históricos referem-se às localizações originais.

## Diagrama no formato do livro

- [Arquitetura em blocos SVG](arquitetura_livro.svg).
- [Prévia PNG](arquitetura_livro.png).
- [PDF](arquitetura_livro.pdf).

## Banco de dados do treinamento

[Origem, quantidades, divisão e transformações por modelo](BANCO_DE_DADOS.md).

## Evolução de treino e validação por época

[Gráficos, tabelas CSV e fontes dos históricos](evolucao%20do%20treinamento/README.md).
