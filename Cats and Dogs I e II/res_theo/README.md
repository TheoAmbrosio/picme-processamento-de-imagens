# res_theo

Este modelo segue as [instruções gerais da competição](../INSTRUCOES_COMPETICAO.md),
compartilhadas com o modelo SVD antigo.

Implementação da arquitetura definida pelo participante: ResNet-34 sem `layer4`
(Conv_5), com os BasicBlocks oficiais `[3, 4, 6]`, média global e `Linear(256, 1)`.
`res_theo/res_theo.py` define a classe `ResTheo` e `criar_res_theo()` carrega os pesos
`ResNet34_Weights.IMAGENET1K_V1` do Torchvision.

Todos os parâmetros e buffers de `conv1`, `bn1` e `layer1`–`layer3` são preservados
e conferidos por chave, shape e igualdade exata. `layer4` e o classificador original
são descartados. Apenas `fc.weight` e `fc.bias` novos recebem Normal(0, 1). O início
do treinamento imprime um resumo desse carregamento.

## Configuração e execução no PC com GPU

Use Python 3.10+. Instale a combinação de PyTorch/Torchvision com CUDA indicada no
[seletor oficial](https://pytorch.org/get-started/locally/) para o computador de execução.
O arquivo de dependências não força uma distribuição CPU nem uma versão de CUDA.
Dentro de `Cats and Dogs I e II`:

```powershell
py -m pip install -r res_theo/requirements.txt
```

Em `res_theo/config_res_theo.py`, preencha `CONFIG`: `otimizador` (nome exato em `torch.optim`),
`learning_rate`, `batch_size`, `epocas`, `frac_validacao` e `frac_teste`. A fração de
treino é `1 - frac_validacao - frac_teste`. Essas escolhas não foram fornecidas e
ficam obrigatórias, sem valores metodológicos escolhidos pelo assistente.
`argumentos_otimizador` contém opções adicionais; as opções efetivas, incluindo os
defaults da biblioteca para a classe escolhida pelo participante, são impressas.

Na mesma configuração ficam `num_workers`, `congelar_backbone`, `precisao_mista`,
`cudnn_benchmark`, `channels_last`, `compilar_modelo`, caminhos e semente. Por padrão,
o backbone é treinável; congelá-lo congela também as estatísticas de BatchNorm.
`num_workers` usa até quatro subprocessos conforme a CPU, deixando um núcleo livre;
é alterável no PC de destino e pode ser zero para desativar os subprocessos.

```powershell
py .\criar_dataset_ampliado.py
py -m res_theo.treinar_res_theo
```

O primeiro comando baixa/completa os originais do Kaggle se necessário e prepara
os arquivos e manifestos. O segundo usa somente treino e validação. A ordem das
16 variantes de treino é embaralhada a cada época. Todas elas são utilizadas,
conforme solicitado; nenhuma nova ampliação é aplicada por cima dos JPEGs.

O dispositivo CUDA é selecionado automaticamente quando disponível. AMP usa
`torch.amp.autocast` e `GradScaler` em CUDA; em CPU fica desabilitada. `pin_memory`
e transferências `non_blocking` são usados para CUDA, e `persistent_workers` somente
quando há subprocessos. A BCE é acumulada no dispositivo, com uma leitura escalar
por conjunto ao fim da época. Não há busca automática, scheduler ou mudanças na perda.

`cudnn_benchmark=False`, `channels_last=False` e `compilar_modelo=False` são os padrões.
Benchmark do cuDNN pode escolher algoritmos diferentes entre execuções. Não foi
medido ganho de `channels_last` ou `torch.compile` no PC CUDA; permanecem opcionais
e desligados. Sementes não garantem determinismo entre ambientes.

## Contrato dos dados

- Originais preservados: `dataset/PetImages/Cat` e `Dog`.
- Rótulos explícitos e verificados no CSV: **Cat = 0; Dog = 1**.
- Saídas: `dataset_ampliado/224x224/PetImages/Cat` e `Dog`, JPEGs RGB de 224×224.
- O SHA-256 do conteúdo original define o split **antes** de gerar variantes. Cópias
  byte a byte ficam no mesmo conjunto; frações são aproximadas e registradas.
- Treino: 2 geometrias × 2 espelhamentos × 4 rotações = 16 imagens por original válido.
- Validação/teste: uma imagem por original, com menor lado 256, bilinear e recorte
  central 224, seguindo a geometria do preset ImageNet V1.
- Normalização é feita **uma vez**, no carregamento: pixels em `[0, 1]`, média
  `[0.485, 0.456, 0.406]` e desvio `[0.229, 0.224, 0.225]`, obtidos do preset oficial.
  Não há normalização salva nos JPEGs nem novo redimensionamento no DataLoader.

O leitor confere todas as combinações, caminhos, rótulos e hashes, e rejeita conteúdo
presente em mais de um split. O teste final não é usado para ajustar ou escolher pesos.
O checkpoint da última época inclui configurações, mapeamento, identidade do manifesto,
estado do otimizador e scaler. O log reporta BCE, a função solicitada.

Arquivos de uma execução antiga com ampliações em avaliação não são aceitos como
um dataset novo. Eles são preservados. Para preparar a divisão correta em outra pasta:

```powershell
py .\criar_dataset_ampliado.py --saida dataset_ampliado/224x224/divisao_v2
```

Nesse caso, ajuste `CONFIG.manifesto` para
`dataset_ampliado/224x224/divisao_v2/manifesto_variantes.csv`, a partir da pasta do
projeto. Com dataset e manifesto já compatíveis, a retomada pula imagens concluídas;
a remoção de Conv_5 não exige recriar as imagens. A revisão da divisão antiga ocorre
por causa da separação treino/validação/teste solicitada, e não da arquitetura.

## Probabilidade de cachorro e teste final

`forward(x)` retorna logits `[B, 1]`; `BCEWithLogitsLoss` recebe rótulos float `[B, 1]`.
Não há sigmoid antes da perda. Para imagens já preparadas/normalizadas:

```python
modelo.eval()
probabilidade = modelo.predict_proba(x)  # sigmoid(logits), [B, 1], P(cachorro)
classe = (probabilidade >= 0.5).long()   # apenas quando uma classe for necessária
```

Para uma imagem nova, o módulo `res_theo.inferir_res_theo` faz a preparação determinística e carrega
o checkpoint sem baixar ImageNet novamente:

```powershell
py -m res_theo.inferir_res_theo .\res_theo\resultados\ultima_epoca.pt "C:\caminho\imagem.jpg"
```

O teste final é um comando separado, para executar quando a escolha do modelo estiver
encerrada. Ele verifica a identidade do manifesto e informa a BCE sem ajustar pesos:

```powershell
py -m res_theo.treinar_res_theo --avaliar-teste .\res_theo\resultados\ultima_epoca.pt
```

## Verificações pequenas

```powershell
py -m pip install pytest
py -m pytest res_theo/tests/test_res_theo.py -q
```

Os testes usam tensores/imagens artificiais e comparam a transferência aos pesos
oficiais (download/cache de aproximadamente 83 MB na primeira execução). Não geram
o dataset real nem fazem treinamento real. CUDA/AMP/compilação dependem de verificação
no hardware de execução. A definição e o preset foram conferidos na
[documentação oficial da ResNet-34](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet34.html)
e o loop segue a [API oficial de AMP](https://docs.pytorch.org/docs/stable/amp.html).
