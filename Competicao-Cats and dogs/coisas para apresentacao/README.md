# Coisas para apresentação

SVGs prontos para navegador e slides, em páginas1600×900.

- [Geral](Geral/README.md): panorama dos modelos, métodos, dados e versões.
- [Theo](Theo/README.md): arquitetura, treino, curvas e linhagem até Aug8.
- [Lisandra](Lisandra/README.md): arquitetura com BatchNorm, treino e10 épocas.
- [Dulce](Dulce/README.md): CNN224 v2, treino, curvas e experimentos.
- [Bandeira](Bandeira/README.md): CNN V4 Final, fine-tuning, curvas e versões.
- [Galeria de todos os SVGs](galeria.html).
- [Relatório Pets publicado: métodos, ranking e gráficos](<Resumo publico Pets/README.md>).
- [Avaliação completa local: CSVs e fotos](<Resultados Pets/README.md>).

Arquiteturas sem rodapés explicativos; blocos, dimensões e resíduos preservados.
Detalhes ficam nos documentos principais para apoiar a fala de vocês.

As validações são distintas; não é um ranking em teste comum. Fases do Theo
incluem uma referência de método; o Aug8 parte diretamente do checkpoint33 da repetição.
No desenho, a cabeça densa tem escala esquemática e layer3 do Theo é abreviada.

Fontes: CSV/JSON dos treinamentos e configurações preservadas. Gerador:
../gerar_apresentacao.py. Os SVGs de treinamento usam históricos existentes. A avaliação Pets está separada e documentada, sem novo treinamento.
