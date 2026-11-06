# Fechamento do ResTheo34 Aug8 — 05/11/2026

Esta é a referência final consolidada antes do experimento adicional com barras.
O treinamento Aug8 terminou suas30 épocas, sem execução do teste final.

- Melhor acurácia de validação:98.32%, época29 da fase, acumulada62.
- Nessa época: acurácia treino92.9468%, BCE treino0.16054724,
  BCE validação0.05326914. Não era calculado F1 nesta fase.
- Última época30: acurácia validação98.22%, treino92.84%, BCE validação0.052578.
- Tempo total aproximado:1h06min32s. GPU NVIDIA RTX A4500, CUDA.
- Resultado: resultados_residuals_networks/res_theo34_aug8/
  20261105_163938_589179_bs64_ep30_cong0ep_do10_dadosamp_pre/.
- melhor_modelo.pt: pesos/estados da época29; ultima_epoca.pt: época30.

## Método efetivamente executado

ResNet34 pré-treinada ImageNet até layer3, blocos3/4/6, canais64/128/256,
layer4 removida, média global, Flatten, dropout10%, Linear256→1.
BCEWithLogitsLoss, Cat0/Dog1, RGB224×224 normalizado pelos valores ImageNet.
Partiu dos pesos do melhor checkpoint da repetição, época33 (97.34%).

AdamW novo com decay0.01 em todos os grupos; LRs iniciais1e-5 para
conv1/bn1/layer1/layer2,5e-5 para layer3 e1e-4 para fc. CosineAnnealingLR
T_max30,eta_min1e-7. Batch64,30 épocas,toda a rede liberada, AMP,workers4,
semente42. Mesma divisão balanceada80/20 anterior à ampliação.

159984 amostras virtuais de treino:8 por original. Variantes: original, ruído,
quadrados, horizontal2×+crop, vertical2×+crop, ruído+quadrados e as duas
combinações de esticamento/ruído/quadrados. Ruído gaussiano variância0.08,
2% branco;8 quadrados pretos20–40 com centros na região150×150.
Validação:5000 imagens determinísticas históricas, sem corrupções novas.

## Proveniência e material preservado

LINHAGEM_MODELO_PRINCIPAL.md identifica a referência, repetição
e os dois ramos de fine-tuning. REGISTRO.md documenta pedidos e verificações.
Cada execução preserva parametros.json, codigo/, historico.csv, treinamento.log,
identificação, gráficos e checkpoints. Gráfico unificado SVG/PNG/PDF na pasta
do Aug8; diagramas da arquitetura em docs/diagramas/.

## Experimento posterior

res_theo34_barras é uma nova continuação, a partir do melhor Aug8 época29,
com30 épocas,4 versões com barras e2 sorteios das corrupções anteriores.
Seu resultado não substitui nem reescreve este fechamento.
