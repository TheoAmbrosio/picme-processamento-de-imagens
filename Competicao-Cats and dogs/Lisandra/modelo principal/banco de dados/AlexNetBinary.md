# Lisandra — dados da versão vigente no GitHub

Fonte: `projeto/CatsandDogs_classification.ipynb`, células2–4.
Kaggle `bhavikjikadara/dog-and-cat-classification-dataset`, via kagglehub.
Saída publicada:24998 imagens;19998 treino,2500 validação,2500 teste.
Split80/10/10 com shuffle=True, sem random_state ou stratify explícitos.
Nenhum manifesto nominal dos splits ou contagem por classe foi publicado.

Rótulo usa a pasta pai imediata:1 se parent_dir.lower()=='dog',0 caso
contrário. Para pastas Cat/Dog, Cat=0 e Dog=1. A regra atual foi conferida
no código; não houve nova leitura de todas as imagens nesta organização.

PIL RGB → Resize((224,224)), esticando a proporção → ToTensor[0,1].
Sem Normalize por canal nem augmentation explícito. Mesma transformação
nos três splits. Sem tratamento explícito de duplicatas ou imagens ilegíveis.

Batch efetivo50, apesar de comentário ainda citar500. Saída:400 lotes de
treino,50 validação e50 teste. Último lote do treino tem48 imagens pelas
contagens publicadas. Treino shuffle=True, demais shuffle=False.

A célula4 publica uma prancha de10 imagens com rótulos, extraída para
../amostra_rotulos.png. Ela é uma amostra, não verificação exaustiva do banco.
Não foram baixados dados nem executadas células do notebook aqui.
