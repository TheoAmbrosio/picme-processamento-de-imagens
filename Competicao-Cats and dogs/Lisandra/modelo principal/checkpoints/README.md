# Pesos recebidos — incompatíveis com a publicação atual

`alexnet_cats_dogs_best.pth`, enviado pela autora via Theo em06/11/2026.
Arquivo original em Downloads preservado; cópia local conferida por SHA256.

- Formato: state_dict puro, sem época, histórico, otimizador ou critério.
- Compatibilidade anterior era com o commit830c79e, sem BatchNorm.
- **Não é compatível com o modelo atual538a5e9**, que acrescenta BatchNorm.
- Pesos finitos; forward artificial1×3×224×224 em eval aprovado.
- Sem teste em imagens reais, cálculo de acurácia ou novo treinamento.

O novo pacote declara checkpoint da época5 por menor val_loss, mas informa
233178729 bytes, enquanto este arquivo tem233148669. É necessário receber
o arquivo novo correspondente. Não atribuir essa época/critério ao peso
local anterior. Histórico atual completo está na seção de evolução.

Detalhes em checkpoint_info.json. Pesos excluídos do Git por .gitignore.
