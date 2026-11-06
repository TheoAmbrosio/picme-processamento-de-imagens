# Método da versão vigente

CNN AlexNetBinary do zero: Conv11×11(3→96,s4,p2)+ReLU+Pool3/s2;
Conv5×5(96→256,p2)+ReLU+Pool3/s2; três Conv3×3,p1 com384/384/256
canais e ReLU; último Pool3/s2. Flatten9216 → Linear4096 → ReLU →
Dropout50% → Linear4096 → ReLU → Dropout50% → Linear1 → Sigmoid.
A camada Linear1000 não existe nesta arquitetura.

Adam LR0.0001, BCELoss,5 épocas previstas,batch50. Predição pelo limiar0.5.
Sem scheduler ou carregamento de pesos pré-treinados explícito.
Código escolhe CUDA se disponível; a saída publicada informa CPU.
RGB224×224, pixels[0,1], split80/10/10 aleatório; detalhes em BANCO_DE_DADOS.md.
Não há salvamento de checkpoint ou histórico por época implementado no notebook.

O comentário do DataLoader ainda diz500, mas os argumentos de código e a
saída de400/50/50 lotes correspondem a batch50. Apenas esse valor efetivo
foi documentado; não foi alterado o código da autora.
