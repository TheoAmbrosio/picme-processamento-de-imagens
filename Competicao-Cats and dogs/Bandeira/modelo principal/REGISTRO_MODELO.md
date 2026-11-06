# Bandeira — revisão atualizada em06/11/2026

Commit: c02a77502bdfe5de55660bf0985d0e7789cd0899. Clonado/atualizado com pull --ff-only.
Agora disponíveis pesos, CSVs de histórico e predições de validação V1–V4.
Históricos completos:15/23/30/30 épocas; checkpoints selecionados por menor
val_loss nas épocas15/18/28/30. Versões independentes, do zero, sem
congelamento, conforme declaração atualizada do autor no README.

Conferidos5000 rótulos/predições por versão:2500 de cada classe.
Acurácia salva reproduz exatamente a do melhor checkpoint no histórico;
BCE recalculada dos CSVs coincide numericamente, com diferenças menores
que3e-7. F1 também coincide com os arredondamentos documentados.

Todos os pesos foram lidos em CPU com weights_only=True e verificados
finitos. V4 carregada estritamente na CNN atual, com forward artificial
1×3×384×384 e saída1×1 finita. Isso verifica compatibilidade estrutural,
não é nova avaliação nas imagens de validação ou teste final.

README agora distingue arquitetura base V1–V3 e otimizada V4. Preparação
offline no notebook data-setting faz resize mantendo proporção e crop central
antes dos loaders. Nenhum código do autor modificado, treino ou teste final
executado. Conferência numérica em CONFERENCIA_ATUALIZACAO.json.
