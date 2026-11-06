# Bandeira — dados do fine-tuning V4 Final

Mesma origem, split estratificado80/20 e seed42 da V4:20.000 treino e5.000
validação esperados para25.000 JPEGs. As predições publicadas contêm5.000 exemplos.
Não há multiplicação fixa da base: augmentation online a cada leitura.

Treino: RandomResizedCrop384(scale0,9–1; ratio padrão3/4–4/3), flip0,5,
rotação±5°, ColorJitter brilho/contraste0,05, ToTensor e normalização do treino
em norm_stats_3.0.json. Validação: Resize((384,384)), ToTensor, mesma normalização.
Os arquivos podem já ter sido preparados offline; consulte V4.md.

Fonte: ../projeto/src/dataset.py e src/fine_tuning.py, commit bd92c13.
