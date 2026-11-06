# Ampliações existentes para o dataset misto

Levantamento dos pipelines atuais e arquivados; não define uma distribuição nova.

| Família | Implementação existente |
|---|---|
| Geometria antiga | Esticamento224×224 ou recorte; normal/espelhada; rotações0/90/180/270;16 combinações |
| Cinza | Preto e branco convertido para3 canais RGB |
| Ruído | Gaussiano,seguido de clamp[0,1]; variâncias históricas0.15/0.1/0.08 conforme fase |
| Pixels brancos |2% de posições sorteadas,normalmente em combinação com ruído |
| Cutout inicial |4 retângulos pretos;altura/largura20–40px |
| Quadrados pretos |8 quadrados de lado20–40; distribuição livre ou centros em região150×150 |
| Quadrados coloridos |8 quadrados20–40,corRGB aleatória por quadrado |
| Esticamento horizontal |224×448,seguido de recorte central224×224 |
| Esticamento vertical |448×224,seguido de recorte central224×224 |
| Combinações Aug8 |Ruído+quadrados; horizontal/vertical+ruído+quadrados |
| Barras |1–4 pretas,largura20–50px; horizontais/verticais/diagonais45/135; atravessam a imagem |
| Faixa Aug12 |Uma faixa horizontal0–50px sobre uma variante sorteada do Aug8 |
| Blur Aug12 |Média10×10,com padding refletido; original,ruído ou quadrados |

Blurs5×5 e20×20 foram apenas prévias visuais,não versões treinadas.
Dropout não é data augmentation: pertence ao modelo.

## Mistura intensificada atual

Implementação em dataset_misto.py; manifestos virtuais preparados por
preparar_dataset_misto.py. Uma versão por tipo listado em VARIANTES:
original; esticamento2× em direção sorteada+crop central; cinza; gaussiano;
pixels brancos; gaussiano+pixels brancos;retângulos pretos; quadrados pretos
livres; quadrados pretos centrais; quadrados coloridos; ruído+quadrados;
barras;faixa horizontal;blur da original;blur do ruído;blur dos quadrados.

16 versões por origem,sem rotações/espelhamentos. Quadrados8→10 (+25%,teto),
variância recente0.08→0.10 (+25%) e blur médio12×12. Demais tamanhos e
intensidades preservados,inclusive barras1–4,largura20–50,faixa0–50 e2% branco.
O ruído anterior de outras intensidades permanece registrado no histórico;
o novo misto usa0.10. A validação histórica não recebe novas transformações.
