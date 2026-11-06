# Modelo principal: res_theo34_aug8

Fechamento consolidado: FECHAMENTO_AUG8.md (98.32% na época29).
Experimentos posteriores (barras, Aug12, misto e continuação cancelada) foram
arquivados em 2026-11-06. O Aug8 e todas as etapas abaixo permanecem disponíveis.

Arquitetura: ResNet34 até layer3, blocos [3,4,6], canais64/128/256, sem layer4,
média global e Linear256→1. Origem ImageNet; rótulos Cat0/Dog1; RGB224×224.
Dropout10% somente antes de fc. A configuração de cada execução é a preservada
em seu parametros.json/checkpoint, nunca a configuração atual de outro pacote.

## Cadeia de referência e execução

| Etapa | Pasta dentro de resultados_residuals_networks | O que foi feito |
|---|---|---|
| Referência histórica | res_theo34/complementados,dropout,com_back | Adam0.001, batch64,50 épocas previstas, backbone congelado5, fc Normal(0,1), dropout0,16×. Interrompida na21; referência metodológica, sem transferência dos pesos para a repetição. |
| Repetição | res_theo34_repeticao/20261105_105951_490569_bs64_ep50_cong5ep_do0_dadosamp_pre | Reiniciou ImageNet nas condições históricas; chegou à36. Melhor checkpoint da33:97.34% validação. Origem dos pesos do Aug8. |
| Principal Aug8 | res_theo34_aug8/20261105_163938_589179_bs64_ep30_cong0ep_do10_dadosamp_pre | Novo ramo da33; mesmo AdamW/grupos/cosseno, dropout10% antes de fc,30 épocas e ampliação8×. |

## Dados e artefatos

Divisão balanceada80/20 antes da ampliação; semente42; validação histórica
determinística Resize256+CenterCrop224. Manifesto compartilhado preservado em
dataset_ampliado/224x224/divisao_balanceada/. Originais em dataset/PetImages/.
Aug8 produz159984 amostras virtuais no treino e usa5000 imagens de validação.
As oito versões e seus parâmetros estão em
residuals networks/res_theo34_aug8/AMPLIACAO.md e no código ampliacao8.py.

Cada pasta contém histórico, log, gráficos, parâmetros e checkpoint. Os caminhos
históricos gravados dentro de JSON/checkpoints antigos foram preservados como
proveniência; os resultados arquivados foram movidos sem regravar os pesos.
Os demais códigos estão em residuals networks/arquivados/; os demais resultados
em resultados_residuals_networks/arquivados/. Nada desses dados foi apagado.

## Visualização da trajetória

- resultados_residuals_networks/res_theo34_aug8/trajetoria_unificada.svg:
  todas as etapas em uma linha cronológica, separadores verticais e legenda
  com alterações e retornos ao checkpoint33 (também PNG/PDF).
- resultados_residuals_networks/res_theo34_aug8/trajetoria_treinamentos.png
- resultados_residuals_networks/res_theo34_aug8/trajetoria_treinamentos.pdf
- resultados_residuals_networks/res_theo34_aug8/resumo_trajetoria.json
- docs/diagramas/arquitetura_res_theo34_aug8.svg (também PNG/PDF)

O gráfico separa as etapas em painéis, com acurácia e BCE de treino/validação.
Pontos pretos indicam a maior acurácia de validação de cada etapa observada.
Não se somam épocas de ramos distintos como se todos fossem sequenciais.
Para atualizar depois de novas épocas, execute o Python do ambiente do projeto
com docs/diagramas/gerar_res_theo34_aug8.py a partir da raiz.
