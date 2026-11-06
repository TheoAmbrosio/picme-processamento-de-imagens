# Lisandra — banco da versão publicada

Fontes: configuracao.json e notebook do repositório, commit538a5e9.
Kaggle bhavikjikadara/dog-and-cat-classification-dataset;24998 imagens.
Treino19998, validação2500, teste2500; split80/10/10 com shuffle,
sem random_state/stratify ou seed manual definidos. Sem manifesto nominal
dos arquivos nem contagem por classe publicada.

Rótulos pela pasta pai imediata: Dog=1, Cat=0. RGB224×224 por Resize direto,
ToTensor e **Normalize ImageNet**: mean[0.485,0.456,0.406],
std[0.229,0.224,0.225]. Sem data augmentation declarado. Batch50;
400/50/50 lotes previstos. Não atribuir a este modelo a ausência de Normalize
de versões anteriores. Sem tratamento explícito de duplicatas/ilegíveis.

Os dados de teste existem no código, mas não foram executados aqui. Dados
originais e splits não baixados ou refeitos por esta organização.
