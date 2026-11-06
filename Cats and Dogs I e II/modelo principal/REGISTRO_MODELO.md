# Registro consolidado do modelo principal

Consolidado em06/11/2026 a partir dos artefatos preservados, sem novo treino.

1. Referência histórica: ResTheo34 ImageNet sem layer4, Adam0.001,
   batch64, backbone congelado5 épocas,16 ampliações, sem dropout efetivo.
   Execução interrompida na21; referência metodológica, sem transferência
   de seus pesos para a repetição.
2. Repetição: nova partida ImageNet nas mesmas condições; executou até36.
   Melhor checkpoint na33:97,34% validação.
3. Aug8 principal: novo ramo diretamente da33, AdamW com grupos
   LR1e-5/5e-5/1e-4, decay0.01, cosseno30 até1e-7, batch64,
   dropout10% antes da fc,8 ampliações,30 épocas completas.
   Melhor época29:98,32% validação; checkpoint preservado.

Os valores detalhados de cada execução estão em `historico_das_etapas/`;
os do principal em `treinamento_aug8/`. Regras e transformações estão em
`documentacao/`. O registro cronológico completo do projeto permanece em
`../REGISTRO.md`, inclusive experimentos posteriores arquivados.

Não houve execução do teste final para esta consolidação. Não foram
atribuídas métricas de outros experimentos ao Aug8.
