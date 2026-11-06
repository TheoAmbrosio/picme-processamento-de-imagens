# O que está publicado e o que fica local

São versionados código de organização, documentação, registros, diagramas
de arquitetura e identificação das fontes. Pesos, datasets, ambientes e
resultados de execuções não são enviados ao repositório do Theo.

As pastas de gráficos de resultados, históricos numéricos, logs e checkpoints
citadas nos índices continuam na cópia local. Links para esses arquivos
podem não abrir na visualização do GitHub, pois os arquivos estão excluídos.

Os repositórios dos participantes têm seus próprios históricos e não são
adicionados como repositórios Git embutidos nem submódulos. Para obter as
fontes numa nova cópia, execute nesta pasta, somente se as pastas ainda
não existirem:

```bash
git clone https://github.com/dulce-costa/Picme2.2Cats_Dogs.git Dulce/projeto
git clone https://github.com/Mateus-Band/Dogs-vs-Cats-IMPATech.git Bandeira/projeto
git clone https://github.com/lisandra21/cats_vs_dogs_Lisandra.git Lisandra/projeto
```

Os clones podem conter artefatos que os próprios autores publicaram. Isso
não os inclui automaticamente no repositório do Theo. Commits usados na
consolidação estão em FONTES.json. Documentação não substitui os pesos
necessários para executar inferência.
