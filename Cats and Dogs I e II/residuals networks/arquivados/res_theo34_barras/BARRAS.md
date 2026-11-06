# Experimento adicional com barras

Origem: melhor Aug8 da época29,98.32% validação. Nova fase de30 épocas,
mesmo AdamW/decay0.01,gruposLR1e-5/5e-5/1e-4,cosseno30 até1e-7,
dropout10% antes de fc,batch64. Otimizador e scheduler novos.

Cada original de treino gera6 versões virtuais:4 leituras com barras e2
sorteios independentes entre ruído,quadrados e ruído+quadrados (com reposição).
Não entram versões originais nem esticadas. Total119988 amostras de treino.
As5000 imagens de validação histórica permanecem intactas.

Barras:1–4 por leitura,largura perpendicular20–50px,pretas,atravessam a imagem
inteira. Sorteia-se horizontal,vertical ou diagonal; diagonais45°/135°.
Ponto de passagem sorteado na região central150×150,como nas oclusões anteriores.
As quatro versões podem variar a cada época/leitura. Ruídos/quadrados anteriores
mantêm variância0.08,2% branco e8 quadrados20–40,conforme código preservado.
