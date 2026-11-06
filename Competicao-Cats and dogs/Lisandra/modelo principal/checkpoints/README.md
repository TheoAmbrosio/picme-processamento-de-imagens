# Pesos recebidos da Lisandra

`alexnet_cats_dogs_best.pth`, enviado pela autora via Theo em06/11/2026.
Arquivo original em Downloads preservado; cópia local conferida por SHA256.

- Formato: state_dict puro, sem época, histórico, otimizador ou critério.
- Compatível estritamente com AlexNetBinary vigente, cabeça9216→4096→4096→1.
- Pesos finitos; forward artificial1×3×224×224 em eval aprovado.
- Sem teste em imagens reais, cálculo de acurácia ou novo treinamento.

Época e critério de seleção ainda precisam ser informados pela autora;
o sufixo best do nome não comprova esses dados. O histórico visível no
notebook continua parcial e não foi atribuído automaticamente a esses pesos.

Detalhes em checkpoint_info.json. Pesos excluídos do Git por .gitignore.
