# Cinco filtros sobre uma pixel art

`kernel_em_acao.py` usa `assets/naruto_referencia_recortada.png`, preparado a partir da imagem enviada por Theo. O script reduz a arte para uma malha 64×96 e aplica cinco filtros 3×3. Dois kernels percorrem a entrada com os pesos em uma sobreposição semitransparente e em um painel ampliado; o quadro final compara bordas verticais, horizontais, duas direções diagonais e média 3×3 (blur).

Os filtros de borda são aplicados à luminância calculada a partir do RGB; o blur calcula a média separadamente em R, G e B. Os mapas de borda mostram a magnitude da resposta, em cores diferentes, com a mesma escala de visualização. Trata-se de correlação cruzada 2D com passo 1 e extensão da borda, operação frequentemente chamada de convolução em CNNs. Os filtros diagonais respondem a transições nessas orientações; não são classificadores de linhas por si só.

## Renderizar no Windows

No PowerShell, entre na pasta:

```powershell
Set-Location -LiteralPath 'C:\Users\theov\OneDrive\Documentos\Picme processamento de imagens\apresentacoes\understanding-deep-learning-10-3-a-10-5\Scripts'
```

Se o comando `manim` estiver disponível:

```powershell
manim -ql --fps 8 -r 640,360 kernel_em_acao.py KernelEmAcao
```

Se não estiver no PATH, use o `uv` já instalado:

```powershell
uv python install 3.12
uvx --python 3.12 manim -ql --fps 8 -r 640,360 kernel_em_acao.py KernelEmAcao
```

Para renderizar em 1080p e 30 fps:

```powershell
uvx --python 3.12 manim -qh --fps 30 -r 1920,1080 kernel_em_acao.py KernelEmAcao
```

Se `manim` já estiver no PATH, troque `uvx --python 3.12 manim` por `manim`. O MP4 aparece em `media/videos/kernel_em_acao/`, em uma subpasta correspondente à resolução e ao FPS. Confira o caminho exato na última linha da saída do Manim.

A versão já renderizada `KernelEmAcao_1080p30.mp4` ainda mostra a arte anterior. Para atualizar o slide após renderizar com o novo script, copie o MP4 gerado em `media/videos/kernel_em_acao/1080p30/KernelEmAcao.mp4` para `KernelEmAcao_1080p30.mp4` nesta pasta.

```powershell
Copy-Item -LiteralPath 'media/videos/kernel_em_acao/1080p30/KernelEmAcao.mp4' -Destination 'KernelEmAcao_1080p30.mp4' -Force
```

A apresentação Delta reproduz `Scripts/KernelEmAcao_1080p30.mp4`.
