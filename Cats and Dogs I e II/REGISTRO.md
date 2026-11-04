# Registro de trabalho — Cats and Dogs I e II

## 04/11/2026 — Organização em pastas e publicação solicitada

- **Pedido:** publicar as alterações em `cats-and-dogs`; durante a preparação, o usuário pediu separar o modelo novo em `res_theo` e chamar a pasta do antigo de `modelo_svd errado`.
- **Organização:** código, configuração, testes, dependências e guia atuais em `res_theo/`; modelo anterior e seus resultados legíveis em `modelo_svd errado/`. Originais e dataset ampliado continuam compartilhados na raiz do projeto, fora do Git. O gerador continua na raiz e sua importação de configuração não exige torch.
- **Comandos:** execução dos módulos por `python -m res_theo.treinar_res_theo` e `python -m res_theo.inferir_res_theo`; checkpoints em `res_theo/resultados/`, ignorados no Git. README principal direciona ao guia atualizado e preserva a documentação antiga na pasta histórica.
- **Verificação:** testes artificiais, conferência de imports e diff, sem treino ou ampliação do dataset real. O envio incluirá a reorganização, o pipeline e as instruções; não incluirá dados, ambiente virtual nem checkpoints.

## 04/11/2026 — Implementação de res_theo e pipeline binário

- **Decisões do participante:** ResNet-34 oficial sem Conv_5/layer4, grupos [3, 4, 6], RGB 224×224, média global e Linear(256, 1); logits com BCEWithLogitsLoss, sigmoid apenas para probabilidades, Cat=0 e Dog=1. Uso de pesos ImageNet permitido pelo grupo, confirmado nesta sessão. Solicitou nome `res_theo` e todas as 16 variantes por original de treino.
- **Arquitetura e pesos:** classe `ResTheo` em `res_theo.py`, reutilização dos módulos oficiais até layer3 e verificação explícita de parâmetros/buffers. Somente fc.weight/fc.bias recebem Normal(0, 1). Congelamento opcional também mantém BatchNorm do backbone em avaliação. Nenhuma camada extra ou alteração da perda.
- **Treinamento e inferência:** arquivos separados, checkpoint identificado como res_theo, probabilidade em [B,1], seleção automática de CUDA, AMP/GradScaler condicional, workers/pinning/transferências assíncronas aplicáveis e acumulação da perda sem leitura escalar por batch. cuDNN benchmark, channels_last e compile são opções; as duas últimas permanecem desativadas por falta de medição no hardware de destino. Teste final usa comando separado.
- **Escolhas ainda não fornecidas:** otimizador, learning rate, batch size, épocas e proporções de treino/validação/teste. São campos obrigatórios em `config_res_theo.py`; não foram escolhidos pelo assistente. Defaults efetivos da classe de otimizador escolhida pelo participante são impressos pelo programa.
- **Dados:** gerador continua nos caminhos existentes, usa o hash do original para dividir treino/validação/teste antes de ampliar, preserva todas as 16 versões no treino e produz uma única versão determinística em validação/teste. CSVs explicitam rótulo, split, origem e hash. JPEGs sem normalização; normalização ImageNet aplicada uma vez no carregamento. A geometria de avaliação/inferência corresponde ao preset oficial.
- **Contrato e retomada:** configuração de dataset com esquema 2 impede misturar splits/configurações incompatíveis e bloqueia saídas antigas sem esse contrato, preservando-as. O leitor rejeita hashes em splits distintos, rótulos conflitantes, caminhos inválidos e variantes incompletas. Evidência: testes artificiais de vazamento e retomada. Essa exigência decorre da nova separação dos dados, não da remoção de Conv_5.
- **Validação local:** 14 testes com tensores/imagens artificiais aprovados em CPU, incluindo batches 1/2, shapes intermediários, probabilidades, perda/backward, igualdade dos pesos oficiais e buffers antes do treino, inicialização restrita à cabeça, congelamento de BN, normalização/geometria, checkpoint/inferência, geração mínima e retomada. Conferência de sintaxe e diff. Pesos oficiais foram baixados/cacheados para essa verificação.
- **Limites:** nenhum treino real, geração do dataset real ou download do dataset foi executado nesta implementação. CUDA/AMP, channels_last e compile não foram exercitados em GPU; não há medição de desempenho nem resultado de classificação. Arquivos e dependências novos documentados em `RES_THEO.md`. Sem commit ou push.

## 04/11/2026 — Download automático e variantes de 224×224

- **Pedido:** permitir executar `criar_dataset_ampliado.py` em outro PC, baixando os originais do Kaggle antes da transformação; trocar 300×300 por 224×224. O usuário proibiu executar o trabalho pesado neste PC.
- **Implementação:** download/cache via `kagglehub` quando faltam JPGs originais, cópia apenas dos arquivos ausentes para `dataset/PetImages`, filtragem de arquivos de imagem e dimensões finais de 224×224. Mantidas as 16 variantes por imagem legível.
- **Retomada:** saída e manifestos em `dataset_ampliado/224x224`, evitando reutilizar variantes antigas de 300×300. `--apenas-verificar` não baixa originais.
- **Dependências e uso:** acrescentado `requirements-dataset.txt` e comandos no README. A referência de contagem agora indica os 12.500 arquivos JPG por classe do download, distinguindo-os das imagens legíveis.
- **Verificação e limites:** revisão do código e do diff; nenhum download, processamento ou treinamento executado nesta alteração. A integração completa deverá ser executada no outro PC. Sem commit ou push.

## 03/11/2026 — Localização dos dados e proposta de ampliação

- **Pedido:** mover o dataset original para a raiz de `Cats and Dogs I e II` e preparar um prompt para outra IA implementar um script de criação de dataset ampliado. O script será colocado nessa mesma pasta; sua implementação ainda não foi solicitada aqui.
- **Mudança realizada:** `Primeiro modelo que deu errado/dataset` foi movido para `dataset`, mantendo `PetImages/Cat` e `PetImages/Dog`. Esse é o caminho esperado por `treinar_modelo.py`. Os dados continuam fora do Git por `.gitignore`.
- **Ideia proposta:** para cada imagem válida, gerar duas geometrias (maior recorte quadrado central redimensionado para 300×300 e imagem inteira deformada para 300×300), com versão original e espelhada horizontalmente, cada uma nas rotações de 0°, 90°, 180° e 270°. São 16 saídas por entrada, preservando o rótulo. A efetividade para robustez dependerá de avaliação experimental, com separação treino/teste antes da ampliação para impedir vazamento de variantes da mesma imagem.
- **Contagem observada:** 12.499 arquivos em `Cat` e 10.224 em `Dog` após a mudança. Se todos forem legíveis, isso produzirá 199.984 imagens de gato e 163.584 de cachorro. A meta de 199.984 por classe exige 12.499 originais válidas em cada classe; não foi inferida nem fabricada a diferença em `Dog`.
- **Próximo passo:** outra IA receberá o prompt e implementará o script. Nenhuma ampliação foi executada, nem houve commit ou push.

Este arquivo acompanha as tentativas e os avanços do projeto. Os resultados abaixo descrevem o que foi implementado ou verificado; não equivalem a uma avaliação final do classificador.

## 28/10/2026 — Reconstrução do projeto

- **Objetivo:** recuperar o trabalho feito na conversa anterior e deixá-lo executável em notebooks.
- **Motivo expresso:** a conversa compartilhada não fornecia o código diretamente; o texto colado pelo usuário serviria para reconstruí-lo.
- **Mudanças:** foram criados `modelo.ipynb`, `testes_k_svd.ipynb` e instruções da competição.
- **Resultado observado:** o histórico do Git registra a adição dos notebooks e das instruções no commit `511d12b`. O desempenho do modelo não foi medido nesse marco.

## 29/10/2026 — Padronização de imagens e representação SVD

- **Objetivo:** tornar as imagens comparáveis para a SVD e escolher entre escala de cinza e RGB.
- **Motivo expresso:** o usuário queria preservar a proporção ao recortar o centro, redimensionar para 300 × 300 e, no modo RGB, fatorar separadamente os três canais. Ele também queria ajustar `k` no início do código.
- **Mudanças:** os notebooks passaram a recortar e redimensionar as imagens antes da SVD; o modo RGB produz fatores para cada canal. A rede usa as dimensões derivadas de `k`, camadas ocultas com largura reduzida a `floor(2/3)` da anterior, saída sigmoide e MSE.
- **Resultado observado:** as mudanças foram publicadas no commit `2b44996`. Esse registro não contém métricas de classificação.

## 29/10/2026 — Cache, amostragem e tempo de execução

- **Objetivo:** evitar refazer a SVD e tornar os experimentos controláveis.
- **Motivo expresso:** o pré-cálculo era demorado, ocorreu um erro `ArpackError`, e o usuário queria selecionar `N` imagens com metade de cada classe e acompanhar o tempo restante do treino.
- **Mudanças:** foi acrescentada uma alternativa de SVD densa quando o ARPACK falha; os vetores foram separados por `k` e modo de cor em `datasets_processados/`; o modelo recebeu `usar_todos_dados`, `N` e estimativa de tempo restante por época.
- **Resultado observado:** 10.667 vetores já calculados para `k=20` em cinza foram preservados no novo local. O cache e as imagens ficam fora do Git. As alterações foram publicadas no commit `42c6422`; o treinamento completo não foi executado pelo assistente.

## 29/10/2026 — Execução em outro computador

- **Objetivo:** permitir que o outro PC obtenha os arquivos e prepare as imagens localmente.
- **Motivo expresso:** o usuário considerou melhor baixar as imagens no próprio PC de destino e gerar ali os fatores SVD, sem colocar as imagens no GitHub. Depois pediu um único script para configurar, treinar e plotar.
- **Mudanças:** o notebook ganhou uma célula de download pelo Kaggle; depois foram criados `preparar_dados.py`, `treinar_modelo.py`, `requirements.txt` e um README com instruções de execução.
- **Resultado observado:** os scripts e o guia foram publicados no commit `4e7a2ee`. A estrutura e o diff foram verificados, mas o download e o treino não foram executados no outro PC pelo assistente.

## 29/10/2026 — Configuração e primeira execução neste Linux

- **Objetivo e motivo expresso:** o usuário pediu clonar `theo-codex`, aplicar suas instruções e skills e executar `treinar_modelo.py` da branch `cats-and-dogs` neste computador.
- **Passos:** o clone inicial encontrou bloqueio de rede e ausência de credenciais. Como `gh` não estava instalado e `sudo` não estava disponível, foi usado GitHub CLI portátil oficial; o usuário autorizou o login no navegador. Foram clonados os dois repositórios. O instalador original `Apply.ps1` foi executado com PowerShell portátil oficial, após diagnóstico e prévia sem conflitos.
- **Resultado da configuração:** instruções globais, quatro skills pessoais e conhecimento geral foram aplicados. O diagnóstico confirmou os arquivos instalados. As skills ECC não estavam disponíveis nesta sessão.
- **Dependências:** criado ambiente `.venv` com Python 3.13; instalado `requirements.txt`; `pip check` passou. PyTorch instalado: `2.15.0+cu132`.
- **Execução autorizada inicialmente:** baixado o dataset do Kaggle, sem login adicional. Foram encontradas 24.998 imagens válidas, divididas em 22.498 de treino e 2.500 de avaliação. Duas imagens ilegíveis foram ignoradas pelo código. Preparados os vetores SVD para `k=20`, escala de cinza, usando todas as imagens, semente de amostragem 42, fração de treino 0,90, Adam com taxa 0,1, batch 50 e 1.000 épocas configuradas.
- **Correção do diagnóstico da GPU:** o teste dentro do sandbox não enxergava os dispositivos NVIDIA e levou inicialmente à conclusão incorreta de indisponibilidade da GPU. Fora do sandbox, CUDA foi confirmado na NVIDIA RTX A4500; o próprio treinamento imprimiu `Treinando em cuda | entrada=12000`. A SVD foi calculada em CPU.
- **Resultado observado:** o log `treinamento.log` registra seis épocas completas; na sexta, MSE treino 0,499556 e avaliação 0,504000. Esses números não demonstram boa classificação e não representam treinamento concluído.

## 29/10/2026 — Interrupção e normalização dos vetores

- **Mudança de direção e motivo expresso:** após perguntar sobre normalização, o usuário pediu parar o treinamento, adicionar a normalização e ensinar a execução manual, sem executar novamente. Também pediu manter este registro com a skill `registro-de-trabalho`.
- **Interrupção:** enviada interrupção ao processo em execução; a sessão terminou com código 130. Nenhum novo treinamento foi iniciado.
- **Mudanças:** criado `normalizacao.py`, com cálculo incremental em lotes de média e desvio padrão populacional por entrada, acumulando em float64. `treinar_modelo.py` ajusta as estatísticas somente no treino e usa os mesmos valores na avaliação. O desvio tem piso `1e-6`; os rótulos e o cache SVD original são preservados. A escolha técnica foi do assistente para evitar vazamento e diferenças de escala, não uma hipótese atribuída ao usuário.
- **Validação:** teste sintético confirmou estatísticas em lotes contra cálculo direto, média zero e desvio unitário em entradas não constantes, tratamento de coluna constante, manutenção dos rótulos, aplicação das estatísticas de treino na avaliação e rejeição de treino vazio. A sintaxe dos dois arquivos foi verificada sem executar o script de treinamento.
- **Documentação:** README atualizado com comando Linux, interrupção por `Ctrl + C`, uso automático de CUDA e descrição da padronização. O modelo e as estatísticas ainda não são persistidos pelo script; essa limitação está documentada. Nenhum commit ou push foi feito.

## Próximos passos

- Avaliar manualmente a penalização configurável abaixo; nenhum treinamento com esse termo foi feito pelo assistente.

- O usuário executará manualmente `treinar_modelo.py` seguindo o comando Linux do `README.md`; o assistente não deve iniciar outro treinamento sem nova solicitação explícita.
- Medir o efeito da padronização em uma nova execução; ainda não há resultados do modelo com os dados normalizados.
- Confirmar que o gráfico aparece ao fim do treino; o primeiro treinamento foi interrompido antes dessa etapa.
- Registrar as configurações usadas e as métricas observadas antes de comparar experimentos.

## 29/10/2026 — Relato de estagnação e penalização perto de 0,5

- **Contexto e motivo expresso:** o usuário relatou que o MSE de treino não diminuiu em 50 épocas e pediu adicionar `alpha / |0.5 - valor|`, com `alpha` configurável, para penalizar o travamento em 0,5. O relato não veio acompanhado das previsões nem de um log dessas 50 épocas; a causa da estagnação não foi confirmada.
- **Discussão anterior:** foram levantadas hipóteses de saturação das sigmoides e taxa de aprendizado elevada; sugeridos experimentos separados com taxa 0,001 e batch maior para comparar tempo por época. Essas hipóteses não foram tratadas como diagnóstico comprovado.
- **Configurações encontradas antes da edição:** `k_modelo=10`, RGB, `usar_todos_dados=False`, `N=1000`, taxa 0,001, batch 256 e 1.000 épocas. Essas edições já estavam no arquivo e foram preservadas.
- **Implementação:** interpretado `valor` como previsão individual da rede. Adicionado `alpha=0.001` configurável e `epsilon_penalizacao=1e-6`; perda de otimização `MSE + mean(alpha / (abs(previsao - 0.5) + epsilon_penalizacao))`. O epsilon é uma adaptação numérica explícita da fórmula singular solicitada. `alpha=0` desativa o termo. Validação impede alpha negativo ou não finito e epsilon não positivo ou não finito.
- **Métricas:** MSE de treino e avaliação continuam sem penalização para permitir comparação; o terminal mostra também a penalização média e a perda total de treino. O gráfico permanece de MSE.
- **Validação observada:** teste isolado da função, extraída por AST sem executar ou importar o programa de treinamento, confirmou fórmula e média, direção dos gradientes em ambos os lados de 0,5, finitude em 0,5 e desativação com alpha zero. Sintaxe verificada. Nenhum treinamento foi iniciado.
- **Limitações:** o termo favorece confiança mesmo em previsões erradas; exatamente em 0,5, o gradiente de `abs` usado pelo PyTorch é zero. Perto desse valor, o gradiente pode ser grande. O valor inicial de alpha e qualquer melhora no aprendizado não foram validados em experimento. A instrução de não iniciar treinamento continua vigente.
- **Próximo passo:** o usuário configura alpha e executa manualmente, comparando MSE e previsões com a execução sem penalização. Sem commit ou push.

## 29/10/2026 — Intervalos das previsões no terminal

- **Objetivo e motivo expresso:** o usuário pediu identificar os dois intervalos de largura 0,1 em que as respostas se concentram. Esclareceu que queria essa informação no log do terminal, não em um gráfico.
- **Mudanças:** a avaliação de cada época acumula as previsões em dez intervalos fixos entre 0 e 1, reutilizando a mesma passagem que calcula o MSE. O log mostra os dois intervalos ocupados mais frequentes, com quantidade e porcentagem sobre toda a avaliação. A escolha de usar a avaliação foi explicitada pelo assistente. Empates priorizam o menor limite inferior; um único intervalo ocupado produz uma única entrada. 1,0 pertence ao último intervalo.
- **Validação:** teste sintético isolado, sem importar nem executar o programa de treinamento, verificou limites, inclusão de 1,0, ordenação, porcentagens, empates, faixa única e preservação do MSE. Nenhum gráfico novo foi adicionado e nenhum treinamento foi iniciado.
- **Resultado e limite:** o código produzirá o resumo nas próximas execuções iniciadas pelo usuário; ainda não foram medidas as concentrações do modelo real após essa alteração. Esses intervalos descrevem previsões e não acurácia. Uma execução já em andamento não incorpora a edição.
- **Próximo passo:** o usuário executa manualmente e observa o campo `intervalos avaliação` por época. Sem commit ou push.

## 29/10/2026 — MSE de avaliação com saídas arredondadas

- **Motivo expresso:** após relatar concentração das respostas perto de zero ou do meio, o usuário esclareceu que deseja observar o MSE de validação com as previsões arredondadas. Não pediu arredondar a saída usada para retropropagação.
- **Mudanças:** adicionado ao log por época `MSE avaliação arredondado`, calculado com classe 0 quando previsão < 0,5 e classe 1 quando previsão >= 0,5. Mantidos o MSE contínuo, os intervalos e a perda diferenciável de treinamento. A nova métrica também é guardada no histórico em memória; o gráfico existente permanece inalterado.
- **Validação:** avaliação isolada em cinco exemplos artificiais, com lotes desiguais, confirmou dois erros e MSE arredondado de 0,4; também foram conferidos o limiar exato 0,5 e a preservação do MSE contínuo. Nenhum treinamento foi executado.
- **Interpretação:** para os rótulos 0 e 1 deste projeto, MSE arredondado é a taxa de erro, igual a 1 menos a acurácia. Ainda não há medição dessa métrica no modelo real após a alteração.
- **Próximo passo:** o usuário reinicia manualmente o script para incorporar a mudança e compara as duas métricas. Sem commit ou push.

## 29/10/2026 — Histograma final das previsões

- **Motivo expresso:** o usuário pediu também plotar a frequência dos resultados ao final.
- **Mudanças:** adicionada `plotar_frequencias`, que avalia o modelo ao término do treino e conta as previsões contínuas em dez intervalos de largura 0,1, incluindo 1 no último. O gráfico mostra quantidades por barra e é aberto junto com o gráfico de MSE. Uma cópia é salva em `frequencia_resultados.png`, sobrescrita a cada execução concluída. Os dados usados são os de avaliação, coerentes com os intervalos do log.
- **Validação:** função testada isoladamente com modelo identidade e cinco previsões artificiais, sem treinamento. Confirmadas contagens, limites 0/0,1/0,5/1, processamento em lotes e geração de PNG em `/tmp`. A sintaxe foi verificada.
- **Limites e próximo passo:** ainda não foi gerado histograma de um modelo real com essa versão. O usuário precisa iniciar uma nova execução e concluí-la. Configurações existentes foram preservadas; nenhum treinamento, commit ou push foi executado pelo assistente.

## 29/10/2026 — Penalização centrada na média da época anterior

- **Motivo expresso:** o usuário observa concentração das respostas em uma média e quer substituir o centro fixo 0,5 da penalização pela média do resultado anterior, buscando concentração nos dois extremos.
- **Interpretação aplicada:** resultado anterior significa a época anterior. O centro é 0,5 na primeira época e depois recebe a média de todas as previsões observadas nos lotes de treino da época anterior, ponderada pelo número de imagens, não uma média simples das médias dos lotes. São previsões coletadas ao longo da atualização dos pesos; não uma nova passagem com o modelo final da época.
- **Mudanças:** função `penalizacao_media_anterior(previsoes, centro)` calcula `mean(alpha / (abs(previsoes - centro) + epsilon_penalizacao))`. O centro permanece fixo durante a época, sem gradiente e sem uso de avaliação. A soma usa previsões destacadas do grafo e float64, sem armazenar todos os resultados. O log mostra o centro utilizado e a média atual. As configurações editadas pelo usuário foram preservadas.
- **Validação:** testes sintéticos isolados verificaram fórmula com centros diferentes, gradientes nos dois lados, finitude no centro, alpha zero e média ponderada em lotes de tamanhos diferentes. A sintaxe e o diff foram verificados; nenhum treinamento foi executado.
- **Limites:** afastar da média anterior não garante acertar as classes nem ocupar os dois extremos simultaneamente; pode causar oscilação entre extremos. O gradiente do valor absoluto segue zero exatamente no centro e pode ser grande perto dele. Não há resultado experimental dessa versão.
- **Próximo passo:** o usuário reinicia manualmente, compara métricas e histograma. Sem commit ou push.

## 29/10/2026 — Equilíbrio da média com alpha2 e configuração experimental

- **Pedidos e mudança de direção:** o usuário pediu somar `alpha2 * (0.5 - media_anterior)^2`, escolher parâmetros e, depois, fazer alpha2 afetar o gradiente. A versão inicialmente adicionada com média anterior foi testada e mostrou gradiente inalterado, por ser constante. Essa versão foi substituída antes de qualquer treinamento pelo assistente.
- **Implementação final:** `penalizacao_equilibrio(previsoes)` calcula `alpha2 * (0.5 - previsoes.mean())^2`, usando a média do lote atual com grafo de autograd preservado. A primeira penalização continua centrada na média da época anterior, calculada exclusivamente no treino. O log separa MSE, repulsão e penalização de equilíbrio; a perda total inclui os dois termos.
- **Parâmetros escolhidos pelo assistente:** taxa de aprendizado 0,001, batch 64, alpha 0,00001, alpha2 0,001 e epsilon 0,001. Mantidos k=20, cinza, N=1000, fração 0,90 e 100 épocas. São pontos de partida experimentais, não valores ideais medidos. Lote foi reduzido considerando o erro de memória relatado anteriormente; alpha foi reduzido e epsilon aumentado para moderar a singularidade.
- **Validação observada:** teste sintético isolado confirmou gradientes de alpha2 contra a expressão analítica `2*alpha2*(media-0.5)/n`, com direção correta acima e abaixo de 0,5, perda e gradiente zero em média 0,5 e desativação com alpha2 zero. Sintaxe e diff verificados. Nenhum treinamento foi iniciado.
- **Limitações:** média 0,5 não implica distribuição em dois extremos e também é atingida com todas as previsões em 0,5. O equilíbrio é imposto por lote e pode conflitar com lotes desbalanceados. Ainda não há métricas reais com a configuração final.
- **Próximo passo:** execução manual pelo usuário, comparando MSE contínuo, MSE arredondado e histograma. Sem commit ou push.


## 29/10/2026 — Aplicação do terceiro colocado da busca

- **Pedido:** aplicar em `treinar_modelo.py` a configuração marcada como convergida no top 3 e fornecer comando de execução manual.
- **Fonte:** terceiro registro de `resultados_busca_k_cor_v5/ranking.csv`; k=20, cinza, N=1000, fração de treino 0,90, semente 42, lr=0,0003, batch=256, alpha=0,001, alpha2=0,01, epsilon=0,001, oito camadas ocultas e fator 0,5. Mantido limite de 100 épocas da busca.
- **Mudanças:** ajustados lr, batch, alpha, alpha2 e fator; demais valores já coincidiam. Preservadas as ativações ReLU internas e sigmoide final do arquivo atual.
- **Evidência da busca:** melhor época 18, MSE de validação 0,2401744843; parada na época 23 por MSE estabilizado. O script manual não foi alterado para reproduzir essa parada automática.
- **Validação:** sintaxe conferida e valores comparados ao ranking. Nenhum treinamento iniciado, commit ou push realizado. Configurar os mesmos hiperparâmetros não garante resultados idênticos.
- **Próximo passo:** execução manual pelo usuário.


## 29/10/2026 — Restauração do ponto de entrada

- **Relato:** comando do treinamento retornava imediatamente ao terminal, sem saída.
- **Diagnóstico:** arquivo local terminava em `previsoes = mod`, dentro da função do histograma, sem `main()` e sem chamada de entrada. O trecho restante era sintaticamente válido, mas apenas definia funções. A origem do truncamento não foi determinada.
- **Correção:** restaurados histograma e salvamento PNG, preparação dos dados, divisão treino/avaliação, padronização, construção da rede e chamada de treinamento dentro de `main()`, com guarda `if __name__ == "__main__"`. Incluída mensagem imediata de início. Preservadas configurações atuais, incluindo todos os dados, k=20 e batch=128.
- **Validação:** análise AST confirmou sintaxe, definição de main e chamada protegida no ponto de entrada; diff conferido. Nenhum treinamento executado.
- **Limitação observada:** disco com apenas cerca de 122 MB livres no diagnóstico; não foi apagado nenhum arquivo. Isso pode impedir novas gravações de cache.
- **Próximo passo:** usuário executa manualmente o comando. Sem commit ou push.


## 29/10/2026 — Explicação matemática em Delta

- **Pedido:** usar a skill theo-delta para uma página HTML comum com rolagem, sem slides, explicando matematicamente o projeto.
- **Produção:** criada `docs/explicacao-matematica.dlt` como artigo em português, cobrindo SVD, dimensão dos fatores, padronização exclusiva do treino, arquitetura ReLU/sigmoide, MSE, ambas as penalizações e gradientes, interpretação das métricas, batches e limites experimentais. A configuração descrita foi lida do código atual.
- **Ferramentas:** preparado Delta 0.4.0 e dependências em diretórios temporários; não foi feita instalação global. Compilado `docs/explicacao-matematica.html`, com matemática pré-renderizada e recursos incorporados.
- **Validação:** lint e build do Delta; inspeção do HTML sem erros KaTeX e com oito seções de artigo; captura de tela local pelo Chrome confirmou formato com rolagem. Corrigidas numeração duplicada e uma advertência de acento em modo matemático. Nenhum treinamento foi executado.
- **Limites:** documento explica implementação e hipóteses; não declara eficácia das penalizações nem verificação humana de prova. Sem publicação, commit ou push.


## 29/10/2026 — Preparação para commit e envio solicitado

- **Pedido:** usuário autorizou commit de todas as alterações e upload ao GitHub; confirmou que imagens e dados processados não precisam ser enviados.
- **Escopo:** código, README, registro, fonte Delta e HTML, gráfico final, log e resultados legíveis da busca (rankings, configurações, históricos e gráficos). Dataset, fatores processados, ambiente virtual, checkpoints `.pt` e temporários da busca ficam fora do Git e são preservados localmente. Checkpoints somam aproximadamente 356 GiB.
- **Espaço:** a partição do projeto estava cheia. Ferramentas temporárias criadas pelo assistente foram removidas de /tmp; como estavam em outra partição, foi necessário mover o cache de downloads do pip para `/tmp/theo-pip-cache-preservado`, preservando seu conteúdo e liberando aproximadamente 2,9 GiB em /home. Nenhum dataset ou modelo foi removido.
- **Configuração atual preservada:** o usuário já havia alterado k para 5 e alpha/alpha2 para zero; esses valores serão versionados como estão. O artigo Delta registra a configuração histórica explicitada na sua elaboração.
- **Validação:** sintaxe dos scripts Python e diff conferidos, busca por padrões comuns de credenciais sem correspondências. Sem execução de treinamento. Commit e envio serão confirmados após a operação.
