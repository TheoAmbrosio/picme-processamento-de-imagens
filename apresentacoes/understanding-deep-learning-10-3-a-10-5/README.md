# Apresentação: Understanding Deep Learning, seções 10.3–10.5

Apresentação de Theo, em Delta, sobre convolução 2D (10.3), downsampling e upsampling (10.4), e aplicações em classificação, detecção e segmentação (10.5), do livro *Understanding Deep Learning*, de Simon J. D. Prince. O notebook 10.5 aplica convolução ao MNIST como exemplo de classificação.

## Arquivos

- `apresentacao.dlt`: 24 slides editáveis, com roteiro completo.
- `apresentacao.html`: versão compilada para visualização.
- `exploracoes.ipynb`: notebook próprio para cálculos e figuras da apresentação.
- `Scripts/kernel_em_acao.py`: animação Manim com a pixel art enviada por Theo e cinco filtros 3×3. O script foi atualizado para usar essa arte; o MP4 atual ainda mostra a arte anterior. Veja `Scripts/README.md` para renderizar novamente.
- `assets/figura_10_10.svg` a `assets/figura_10_13.svg`: figuras originais do livro para convolução RGB, downsampling, upsampling e convolução transposta.
- `assets/figura_10_14.png` a `assets/figura_10_20.*`: figuras do livro para convolução 1×1, classificação, detecção e segmentação.
- `10_3_2D_Convolution.ipynb`, `10_4_Downsampling_and_Upsampling.ipynb` e `10_5_Convolution_For_MNIST.ipynb`: notebooks originais do livro, incluídos como material de estudo, sem alterações de conteúdo.

Os três notebooks foram obtidos em 2026-11-01 de [udlbook/udlbook, Notebooks/Chap10](https://github.com/udlbook/udlbook/tree/main/Notebooks/Chap10). A apresentação e o notebook `exploracoes.ipynb` são arquivos próprios. O `aula2.html` do professor serviu apenas como referência visual e não foi copiado.

## Demonstração interativa sugerida

O [visualizador 2D de CNN de Adam W. Harley](https://adamharley.com/nn_vis/cnn/2d.html) permite desenhar um dígito, observar filtro, entrada ponderada, cálculo e saída, e ativar camadas de convolução e redução. Há também uma [versão 3D](https://adamharley.com/nn_vis/cnn/3d.html). Os slides incluem duas capturas da demonstração com crédito ao autor. Referência: A. W. Harley, “An Interactive Node-Link Visualization of Convolutional Neural Networks”, ISVC, 2015.

As imagens `assets/harley_8.png` e `assets/harley_1.png` são capturas da demonstração do site com os dígitos 8 e 1 desenhados para esta apresentação. Os slides mostram as respostas das camadas a esses dois exemplos e identificam o visualizador e seu autor. Os desenhos e textos de análise são próprios; os mapas de ativação e a interface são do visualizador de Harley.

## Figuras do livro

As figuras 10.10–10.13 foram extraídas, sem edição, do [pacote SVG do capítulo 10 disponibilizado na página oficial do livro](https://udlbook.github.io/udlbook/). Correspondem, respectivamente, aos arquivos `ConvImage.svg`, `ConvDown.svg`, `ConvUp.svg` e `ConvTranspose.svg`. Crédito: Simon J. D. Prince, *Understanding Deep Learning*, MIT Press, 2023. O pacote informa licença CC BY-NC-ND. Os slides preservam as figuras e citam o autor.

As figuras 10.14–10.20 foram convertidas do [pacote PDF oficial do capítulo 10](https://github.com/udlbook/udlbook/raw/main/PDFFigures/UDLChap10PDF.zip), respectivamente de `Conv1DChannels.pdf`, `ConvImageNet.pdf`, `ConvAlex.pdf`, `ConvVGG.pdf`, `ConvYOLO.pdf`, `ConvSemSeg.pdf` e `ConvSemSegResults.pdf`. Foram apenas rasterizadas e recortadas nas margens brancas para caber nos slides. Cada slide informa o número da figura e os autores originais indicados no livro.

A imagem de Naruto foi fornecida por Theo como referência do Pinterest. `Scripts/assets/naruto_referencia_recortada.png` é um recorte sem a grade, preparado para a animação; o artista da imagem original não foi identificado.

A animação Manim está no slide “Kernel em ação”. A versão 1080p/30 fps tem cerca de 984 KB e 13 segundos; o HTML a carrega de `Scripts/KernelEmAcao_1080p30.mp4`, sem embuti-la. Ao compartilhar, envie a pasta inteira ou mantenha o MP4 nesse caminho relativo. Se o navegador impedir a reprodução local, abra o MP4 diretamente ou sirva a pasta com `python -m http.server 8000` e acesse `http://localhost:8000/apresentacao.html`.

## Roteiro de fala (cerca de 19 minutos)

1. Abertura e percurso (slides 1–2): 1 minuto. Diga que a apresentação privilegia a intuição visual e as aplicações; a implementação em código será trabalhada ao longo da semana.
2. Cálculo local, RGB, vídeo e canais versus resolução (slides 3–6): 4 minutos. Mostre a janela transparente do kernel e compare os cinco resultados finais; o vídeo dura cerca de 14 segundos.
3. Downsampling, upsampling, transposta e convolução 1×1 (slides 7–11): 5 minutos. Use as figuras 10.11–10.13 para explicar dimensões e perda de informação.
4. Aplicações da seção 10.5 (slides 12–18): 5 minutos. Use 10.15–10.17 para classificação, 10.18 para detecção e 10.19–10.20 para segmentação; cerca de 40 segundos por figura.
5. Visualizador de Harley, dígitos 8 e 1, limites da interpretação (slides 19–22): 3 minutos. Se houver internet, abra o visualizador; as capturas cobrem a demonstração offline.
6. Síntese e referências (slides 23–24): 1 minuto.

Reserve até 1 minuto para perguntas ou para demonstrar ao vivo o visualizador. O total fica entre 19 e 20 minutos. Os textos dos slides são pontos de apoio, não uma leitura literal.

## Compilar

Nesta pasta, execute `delta lint apresentacao.dlt` e `delta build apresentacao.dlt -o apresentacao.html`. Abra o HTML gerado no navegador.

## Materiais de estudo

`exploracoes.ipynb` está disponível para cálculos próprios. Os três notebooks do livro são referências separadas e não representam resultados reproduzidos nesta apresentação.
