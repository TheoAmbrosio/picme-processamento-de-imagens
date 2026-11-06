# Situação após correções locais autorizadas

- Novo checkpoint com BatchNorm compatível; carregamento estrito e forward artificial aprovados.
- Sintaxe corrigida nos dois notebooks locais: strings de impressão e títulos com escapes apropriados.
- README local alinhado à arquitetura, normalização, perda e método publicados.

Não há pendências técnicas das verificações acima na cópia local. As correções
não mudaram arquitetura, hiperparâmetros, saídas salvas ou pesos.
Foram autorizadas pelo Theo e estão preservadas em
[patch para revisão](correcoes_locais/sintaxe_readme.patch).
O repositório remoto da Lisandra não foi alterado por nós.

Época5 e seleção por menor val_loss continuam conforme os metadados da autora;
state_dict não armazena esses campos. Não executamos treino nem teste real.
