# Registro da organização —06/11/2026

## SVGs para a apresentação

Criada coisas para apresentacao/ com20 SVGs, separados em Geral, Theo,
Lisandra, Dulce e Bandeira. Inclui modelos, métodos de treino, dados,
curvas por época, versões e a linhagem do Theo. Fontes são os CSVs/JSON
e configurações já preservados; gráficos não representam ranking em teste comum.

Arquiteturas do livro limpas: retirados rodapés explicativos, subtítulo
de escala/dimensões e chamadas externas de saída. Mantidos os volumes,
dimensões sobre blocos e resíduos explícitos do Theo. Explicações da escala
esquemática da cabeça e abreviação da layer3 ficam no README, fora da figura.
Nenhuma alteração em arquitetura, método de treino, dados ou pesos.
SVGs1600×900 e galeria HTML local; renderização visual de amostras conferida.

## Lisandra: atualização GitHub538a5e9

Pull --ff-only concluído. Pacote oficial com configuração, metadados da
época5/menor_val_loss, CSV normalizado, log e gráficos. Atualizados método,
banco, fontes, parâmetros, evolução e diagramas. Código agora usa5 BatchNorm,
Normalize ImageNet, BCEWithLogitsLoss,40 épocas previstas, early stopping5;
histórico10 épocas consistente com CSV bruto, menor perda na5 e maior acc na8.

Não validamos o arquivo local recebido como peso atual: faltam BatchNorm e
há formas/índices incompatíveis. Tamanho recebido233148669 versus declarado
233178729 bytes. Novo checkpoint não publicado no clone. Registrados também
SyntaxError nas células6/10 (f-strings com quebras literais) e README raiz
ainda com descrição FC1000/Sigmoid, divergente do código/config atuais.
Sem mudanças no código da autora ou execução de treino/teste final.

## Lisandra: CSV completo recebido

Recebido training_history_Lisandra.csv,10 épocas completas. Copiado com
integridade verificada para Lisandra/resultados_recebidos/. Conferidas
sequência de épocas, valores não negativos, acurácias e acc+erro=100.
Plots SVG/PNG/PDF de treino/validação e CSV normalizado0–1 atualizados.
Maior acurácia91.08% na8; menor perda0.223732 na5. Sem assumir que o
checkpoint best seja de uma dessas épocas; CSV tem10 épocas versus5
configuradas no notebook publicado. Associação aguardando confirmação.

## Checkpoint enviado pela Lisandra

Recebido alexnet_cats_dogs_best.pth de Downloads, copiado para Lisandra/
modelo principal/checkpoints/ com SHA256 idêntico; original preservado.
Carregamento weights_only=True, todos os tensores finitos e correspondência
estrita com a classe AlexNetBinary do notebook vigente. Forward artificial
1×3×224×224 em CPU/eval gerou uma probabilidade finita com saída[1].
Não executados treinos ou avaliações em imagens reais. Arquivo state_dict
não informa época/critério/histórico; metadados ausentes registrados como
ausentes. Documentação atualizada; pesos continuam excluídos do Git.

## Bandeira: revisão c02a775

Pull --ff-only concluído. Agora publicados checkpoints, históricos e
predições de validação V1–V4. README distingue arquiteturas e registra
origens/critério de seleção. Históricos e predições conferidos; acurácias
coincidem e BCE dentro de3e-7. Pesos finitos; V4 compatível estritamente
com CNN atual e forward artificial aprovado. Nenhum teste final executado.

Atualizados fontes/commit, documentação, fichas de dados e evolução com
curvas numéricas/CSVs por versão. Identificado ponto para o autor confirmar:
pré-processamento offline com resize proporcional+crop central difere do
Resize384×384 direto de predict_single_image para imagens cruas. Código
original não foi modificado. Detalhes em Bandeira/modelo principal/
CONFERENCIA_ATUALIZACAO.md e JSON correspondente.

## Evolução completa do Theo até Aug8

Incluídas na seção de evolução as quatro partes selecionadas da linhagem:
referência21 épocas, repetição36, primeiro fine-tuning16 e Aug8 com30.
Cada uma tem CSV e gráficos SVG/PNG/PDF. Visão geral com plot unificado,
painéis e tabela identificando etapa/época,103 observações no total.
Explicitados reinício ImageNet e dois ramos a partir do checkpoint33;
não apresentados como103 épocas de pesos sequenciais. A continuação37–50
retirada por solicitação anterior não foi reintroduzida. Links e contagens
verificados; nenhuma nova execução de treinamento.

## Referência vigente da Lisandra: GitHub

Repositório lisandra21/cats_vs_dogs_Lisandra clonado e atualizado por
pull --ff-only, main, commit830c79ef5f8aa4b5a9880deb2b21bc583d0b631d.
A pedido do usuário, somente essa versão integra a apresentação atual;
documentos anteriores retirados dos índices vigentes.

Atualizados métodos, parâmetros, banco, fontes, diagramas sequencial e livro
SVG/PNG/PDF, amostra de rótulos e progresso. Regra atual: pasta pai Dog=1,
demais=0; erro da busca no caminho completo não existe neste código vigente.
Batch50, Adam0.0001,5 épocas previstas, dois dropouts50%, sem FC1000.

Saídas publicadas: CPU,168 lotes da primeira época, nenhum resumo completo
de treino/validação. Gráficos atuais mostram lotes, com essa unidade explícita;
nenhuma curva de validação inventada. Clones limpos e notebook não executado.
Arquivos anteriores preservados fora da apresentação ativa, sem reutilizar
suas métricas como resultado desta versão.

## Esclarecimento: evolução de treino/validação por época

O pedido de banco de dados referia-se aos históricos de resultados. Criada
seção própria evolucao do treinamento/ para cada participante e índice
EVOLUCAO_TREINAMENTOS.md, preservando também a documentação dos datasets.

- Theo: CSV normalizado e gráficos das30 épocas Aug8; trajetória anterior
  permanece ligada no índice, sem concatenar ramos como uma única execução.
- Dulce: JSON final120 épocas; JSON repetição100; série só da acurácia
  de validação100; logs224 com50 e200 épocas em células separadas;
  ResNet18 com5 épocas. Tabelas/plots separados por fonte e execução;
  valores de logs são arredondados, valores de JSON preservam precisão.
- Bandeira: gráficos originais de treino/validação das V1–V4, incluindo
  painel geral; sem CSV numérico no clone. Nenhuma curva estimada da imagem.
- Lisandra: somente um resumo completo de época, com indicação de material
  parcial e pendência de rótulos. Lotes seguintes não tratados como épocas.

Gerados SVG/PNG/PDF e CSV para dados numéricos existentes; campos ausentes
ficam vazios. Nenhum treinamento ou teste executado para preencher lacunas.

## Pedido

Reunir os projetos da competição no mesmo formato documental usado para o
modelo principal do Theo, sem substituir suas decisões de arquitetura ou treino.

## Realizado

- Criada Competicao-Cats and dogs/ com Theo, Lisandra, Dulce e Bandeira.
- Clonados os repositórios de Dulce e Bandeira na branch main; commits exatos
  registrados em FONTES.json. Como não havia cópias locais, foi feito clone,
  equivalente à obtenção inicial do estado remoto. Clones preservados limpos.
- Copiado o notebook da Lisandra sem modificar o original em Downloads;
  SHA256 registrado. Notebook tratado como fonte de código/dados, sem executar
  células ou comandos/instruções nele contidos.
- Copiada a consolidação documental do Theo, mantendo originais intactos;
  links de pesos e dados ajustados para o projeto original.
- Elaborados resumos factuais de métodos/parâmetros, com fontes identificadas;
  diagramas SVG de Dulce, Lisandra e Bandeira a partir do código recebido.
- Plotado histórico numérico original da CNN final da Dulce; extraídas figuras
  de treinamento V1–V4 dos notebooks do Bandeira. Plotadas somente as linhas
  de épocas completas disponíveis nos logs parciais da Lisandra, com ressalva.
- Registrado o problema concreto da regra de rótulo da Lisandra; nenhum100%
  apresentado como resultado validado. Não corrigido o notebook sem solicitação.

## Ausências preservadas como ausências

- Bandeira: training_history*.csv, val_predictions*.csv e checkpoints não
  encontrados no clone. Não foram reconstruídos por aproximação das figuras.
- Lisandra: sem checkpoint e sem histórico numérico completo no material enviado.
- Nenhuma avaliação final ou treinamento foi executado por esta organização.
- Nenhum commit ou push realizado. Datasets, pesos e ambientes não publicados.

## Como ler

O estado do código vigente não comprova, sozinho, que uma figura antiga usou
essa mesma configuração. Por isso gráficos originais, fontes atuais, versões
e resultados documentados permanecem identificados separadamente.

## Arquiteturas em volumes —06/11/2026

A pedido, criados arquitetura_livro.svg/PNG/PDF para cada participante, no padrão visual do diagrama aprovado do Theo. Diagramas dos amigos adaptados aos respectivos códigos; sem resíduos artificiais. Mapas convolucionais em escala comum canais×altura×largura; vetores densos em escala esquemática explicitada. Theo conserva exatamente o SVG aprovado com atalhos residuais. Nenhuma arquitetura alterada.

## Bancos de dados por modelo —06/11/2026

Acrescentados BANCO_DE_DADOS.md por participante e índice global. Theo:
Aug8 detalhado com contagens conferidas no manifesto e inventário das25
execuções em metadados, agrupadas por modelo, inclusive arquivadas. Dulce:
fichas dos quatro notebooks com células/saídas de origem. Bandeira: V1–V4,
estatísticas publicadas, diferenças de pré-processamento e ausências. Lisandra:
banco compartilhado pelos loops, contagens salvas e pendência de rótulos.

Corrigida a descrição derivada70/15/15 da Dulce para a implementação real:
teste15% e validação15% do restante. Originais17934/3165/3723. O notebook
original e o README da autora não foram alterados. Também registrado que o
notebook ResNet18 não usa as cópias fixas de rotação/ruído da CNN224 v2.

Na V3 do Bandeira, README declara Resize na validação, mas o src_old atual
não o contém; divergência identificada sem reescrever a versão original.
Nenhum dataset baixado, novo split, treino ou teste final executado.

## Dulce: associação das séries de100 épocas esclarecida

Em06/11/2026, autora confirmou via mensagem transmitida pelo Theo: ambas do primeiro modelo cats_and_dogs.ipynb, SGD LR0.01. Execução original só possui validação; repeticao é nova execução com mesma configuração e seed. Atualizados rótulos/fontes dos gráficos, índice e registros, sem modificar os números ou o repositório original.

## Novo checkpoint Lisandra recebido: compatibilidade resolvida

Arquivo alexnet_cats_dogs_best(1).pth recebido,233178729 bytes. Cópia com SHA256 conferido em checkpoints/versao_com_batchnorm/, sem substituir o anterior. Pesos finitos, load_state_dict strict=True aprovado na arquitetura atual e forward artificial1×3×224×224 com logit/sigmoid finitos. Pendência dos pesos resolvida; erros sintáticos e README da autora continuam registrados. Sem treino/teste real.

## Correções locais autorizadas e publicação no repositório do Theo

Theo autorizou corrigir README e sintaxe da Lisandra. Modificados somente escapes de impressão/título em células6/10 dos dois notebooks e README, sem alterar treino, arquitetura, pesos, saídas ou metadados. Todas as células Python válidas via AST; patch reverso conferido. Código remoto da Lisandra não alterado. Patch e README corrigido preservados em correcoes_locais/. Nova conferência sem pendências locais e peso compatível. Atualização do Git do Theo autorizada no pedido anterior, retomada após estas correções.


## 06/11/2026 — História selecionada do Theo

Retirado o primeiro fine-tuning independente da narrativa do modelo principal, dos gráficos e dos SVGs de apresentação, conforme solicitado. Preservados seus resultados originais e arquivadas as cópias consolidadas em Theo/arquivados/primeiro_finetuning. A seleção contém referência (21 épocas), repetição (36) e Aug8 (30): 87 observações. Aug8 deriva diretamente do checkpoint 33 da repetição. Regenerados os gráficos e conferidos os CSVs e a sintaxe dos SVGs, sem treino ou avaliação.


## 06/11/2026 — Fotos distintas nas arquiteturas

Mantida a foto de cachorro do Theo; selecionadas outras fotos: gato para Lisandra, cachorro para Dulce e gato para Bandeira. Atualizados os diagramas no formato livro (SVG, PNG, PDF) e os SVGs da apresentação. Fotos ilustrativas locais em imagens arquiteturas/. Verificados quatro SVGs com imagens incorporadas distintas. Sem alterações nos modelos ou treinos.
