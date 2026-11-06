# Competição Cats and Dogs

[Coisas para apresentação: SVGs por participante e visões gerais](coisas%20para%20apresentacao/README.md).

Organização documental dos projetos recebidos em06/11/2026.
Não foram executados treinamentos, testes finais ou alterações metodológicas.

| Participante | Modelo documentado | Material consolidado |
|---|---|---|
| Theo | ResTheo34 Aug8, com antecedentes selecionados | [Abrir](Theo/modelo%20principal/README.md) |
| Lisandra | AlexNetBinary vigente no GitHub | [Abrir](Lisandra/modelo%20principal/README.md) |
| Dulce | CNN224 v2, declarada final no README da autora | [Abrir](Dulce/modelo%20principal/README.md) |
| Bandeira | CNN V4 vigente em src/model.py | [Abrir](Bandeira/modelo%20principal/README.md) |

Cada participante tem `modelo principal/`, com arquitetura SVG, método,
parâmetros e gráficos disponíveis. As fontes originais ficam separadas dos
documentos gerados. O código e os resultados originais do Theo continuam em
`../Cats and Dogs I e II/`, preservados.

## Disponibilidade e limites do material

## Evolução do treino e da validação por época

[Abrir históricos, tabelas CSV e gráficos de cada participante](EVOLUCAO_TREINAMENTOS.md).
Esta é a seção dos resultados ao longo das épocas; a seção de bancos de dados
abaixo descreve as imagens e a preparação delas.

### Históricos disponíveis

- **Theo:** histórico e gráficos das etapas da linhagem; melhor Aug898,32%
  na época29. Checkpoint ligado ao projeto original, sem duplicar pesos.
- **Dulce:** histórico completo de120 épocas da CNN224 v2; validação96,9036%
  na época112. O experimento ResNet18 também existe no clone, mas não foi
  escolhido como substituto da CNN final pela organização.
- **Bandeira:** fonte V4 e análises V1–V4. Pesos, históricos CSV e predições
  de validação V1–V4 disponíveis no clone atualizado. Gráficos numéricos
  SVG/PNG/PDF por época na seção de evolução; V4 compatível com os pesos.
- **Lisandra:** pacote publicado com10 épocas e checkpoint declarado da5
  por menor val_loss (90,16% de validação); maior acurácia91,08% na8.
  Modelo atual usa BatchNorm, BCEWithLogitsLoss e Normalize ImageNet.
  Novo peso com BatchNorm recebido, compatibilidade estrita e teste artificial aprovados.
  [Pendências verificadas](Lisandra/modelo%20principal/PENDENCIAS_ATUAIS.md).

**Não há ranking comum:** os participantes usaram divisões, normalizações,
datasets e protocolos diferentes. Métricas acima são descrições de fontes
distintas. Saídas de testes já presentes nos notebooks não orientaram decisões
ou alterações nesta organização.

## Fontes e atualização

## Banco de dados do treinamento

[Consultar origem, divisão, quantidades e ampliações de todos os modelos](BANCOS_DE_DADOS.md).
As fichas incluem variantes anteriores e distinguem saídas observadas de
contagens apenas esperadas; nenhuma imagem foi baixada para esta documentação.

### Repositórios e fontes

Repositórios clonados na branch principal publicada:

- [Dulce](https://github.com/dulce-costa/Picme2.2Cats_Dogs).
- [Bandeira](https://github.com/Mateus-Band/Dogs-vs-Cats-IMPATech).
- [Lisandra](https://github.com/lisandra21/cats_vs_dogs_Lisandra), versão vigente e commit em [FONTES.json](FONTES.json).

[FONTES.json](FONTES.json) registra commits e origens.
[REGISTRO_ORGANIZACAO.md](REGISTRO_ORGANIZACAO.md) registra o que foi realizado.
Os clones têm seus próprios `.git` e não foram alterados. Para atualizá-los:

```bash
git -C "Dulce/projeto" pull --ff-only
git -C "Bandeira/projeto" pull --ff-only
git -C "Lisandra/projeto" pull --ff-only
```

Esses comandos partem desta pasta. `organizar.py` gera a consolidação sem
executar código dos participantes; utiliza matplotlib no ambiente Python
existente do Theo. Uma nova execução refaz os documentos a partir das fontes.
O notebook vigente da Lisandra é lido diretamente do clone do GitHub.

Esta pasta reúne os materiais organizados localmente.
Código e documentação são versionados no repositório do Theo;
artefatos de execução permanecem locais, conforme [PUBLICACAO.md](PUBLICACAO.md).

## Arquiteturas no formato do livro

- [Theo: SVG](Theo/modelo%20principal/arquitetura_livro.svg).
- [Lisandra: SVG](Lisandra/modelo%20principal/arquitetura_livro.svg).
- [Dulce: SVG](Dulce/modelo%20principal/arquitetura_livro.svg).
- [Bandeira: SVG](Bandeira/modelo%20principal/arquitetura_livro.svg).
