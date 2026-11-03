"""Cinco filtros 3×3 sobre pixel art adaptada da imagem enviada pelo usuário.

Dois filtros percorrem a entrada; o quadro final compara cinco resultados.
Os filtros de borda atuam na luminância da imagem RGB. O blur atua em
cada canal RGB. A operação implementada é correlação cruzada, como nas CNNs.
"""

from __future__ import annotations

from pathlib import Path
from PIL import Image

import numpy as np

from manim import (
    BLUE_C,
    DOWN,
    FadeIn,
    FadeOut,
    Group,
    ImageMobject,
    LEFT,
    linear,
    MoveAlongPath,
    RIGHT,
    Scene,
    Square,
    Text,
    UP,
    VGroup,
    VMobject,
    WHITE,
    YELLOW,
)
from manim.mobject.types.image_mobject import RESAMPLING_ALGORITHMS


ALTURA = 96
LARGURA = 64
ARTE = Path(__file__).parent / "assets" / "naruto_referencia_recortada.png"

FILTROS = (
    (
        "Bordas verticais",
        np.array(((-1, 0, 1), (-2, 0, 2), (-1, 0, 1)), dtype=float),
        (72, 212, 238),
    ),
    (
        "Bordas horizontais",
        np.array(((-1, -2, -1), (0, 0, 0), (1, 2, 1)), dtype=float),
        (255, 178, 86),
    ),
    (
        "Diagonal ↘",
        np.array(((-2, -1, 0), (-1, 0, 1), (0, 1, 2)), dtype=float),
        (226, 126, 207),
    ),
    (
        "Diagonal ↗",
        np.array(((0, 1, 2), (-1, 0, 1), (-2, -1, 0)), dtype=float),
        (154, 137, 255),
    ),
    (
        "Blur 3×3",
        np.ones((3, 3), dtype=float) / 9,
        (210, 221, 235),
    ),
)


def pixel_art_naruto() -> np.ndarray:
    """Carrega o recorte da imagem enviada e a amostra como pixel art RGB."""
    personagem = Image.open(ARTE).convert("RGBA")
    personagem.thumbnail((LARGURA, ALTURA), Image.Resampling.NEAREST)
    fundo = Image.new("RGB", (LARGURA, ALTURA), (18, 25, 43))
    x = (LARGURA - personagem.width) // 2
    y = (ALTURA - personagem.height) // 2
    fundo.paste(personagem, (x, y), personagem)
    return np.asarray(fundo)


def correlacionar(matriz: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Correlação cruzada 2D, passo 1, com extensão de borda."""
    altura, largura = matriz.shape
    entrada = np.pad(matriz, 1, mode="edge")
    saida = np.empty((altura, largura), dtype=float)
    for y in range(altura):
        for x in range(largura):
            saida[y, x] = np.sum(entrada[y : y + 3, x : x + 3] * kernel)
    return saida


def aplicar_filtro(rgb: np.ndarray, indice: int) -> np.ndarray:
    """Aplica filtro de borda à luminância ou blur a cada canal RGB."""
    _, kernel, tinta = FILTROS[indice]
    if indice == 4:
        canais = [correlacionar(rgb[:, :, c].astype(float), kernel) for c in range(3)]
        return np.clip(np.stack(canais, axis=2), 0, 255).astype(np.uint8)

    luminancia = np.tensordot(
        rgb.astype(float), np.array((0.2126, 0.7152, 0.0722)), axes=([2], [0])
    )
    resposta = correlacionar(luminancia, kernel)
    intensidade = np.clip(np.abs(resposta) * 255 / 850, 0, 255)
    cor = np.asarray(tinta, dtype=float)
    return np.clip(intensidade[:, :, None] * cor / 255, 0, 255).astype(np.uint8)


def imagem_pixelada(pixels: np.ndarray, altura: float) -> ImageMobject:
    imagem = ImageMobject(pixels)
    imagem.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    imagem.set_height(altura)
    return imagem


def sobreposicao_kernel(kernel: np.ndarray, passo: float, cor) -> VGroup:
    """Matriz semitransparente com os nove pesos sobre a entrada."""
    grupo = VGroup()
    for y in range(3):
        for x in range(3):
            celula = Square(
                side_length=passo,
                stroke_color=cor,
                stroke_width=1.4,
                fill_color="#101827",
                fill_opacity=0.52,
            ).move_to(((x - 1) * passo, (1 - y) * passo, 0))
            peso = kernel[y, x]
            texto = f"{peso:g}"
            rotulo = Text(texto, font_size=14, color=WHITE)
            if rotulo.width > passo * 0.83:
                rotulo.scale_to_fit_width(passo * 0.83)
            rotulo.move_to(celula.get_center())
            grupo.add(celula, rotulo)
    return grupo


def percurso_imagem(imagem: ImageMobject, passo: float, altura: int, largura: int) -> VMobject:
    """Zigue-zague por várias regiões do retrato, mantendo a janela dentro."""
    centro = imagem.get_center()
    canto_x = centro[0] - imagem.width / 2
    canto_y = centro[1] + imagem.height / 2
    pontos = []
    linhas = [round((altura - 1) * fracao) for fracao in (0.10, 0.26, 0.42, 0.58, 0.74, 0.90)]
    for linha, y in enumerate(linhas):
        colunas = range(2, largura - 2) if linha % 2 == 0 else range(largura - 3, 1, -1)
        for x in colunas:
            pontos.append((canto_x + (x + 0.5) * passo,
                           canto_y - (y + 0.5) * passo, 0))
    caminho = VMobject()
    caminho.set_points_as_corners(pontos)
    return caminho


class KernelEmAcao(Scene):
    """Dois filtros percorrem o Naruto; o final compara cinco saídas."""

    def construct(self) -> None:
        self.camera.background_color = "#101827"
        original = pixel_art_naruto()
        titulo = Text("Cinco filtros sobre uma pixel art", font_size=34, color=WHITE)
        titulo.to_edge(UP, buff=0.3)
        self.play(FadeIn(titulo), run_time=0.35)

        entrada = imagem_pixelada(original, 5.0).move_to(LEFT * 3.5)
        rotulo_entrada = Text("Entrada RGB • Naruto", font_size=21, color=WHITE)
        rotulo_entrada.next_to(entrada, UP, buff=0.22)
        saida = imagem_pixelada(aplicar_filtro(original, 0), 5.0).move_to(RIGHT * 3.5)
        rotulo_saida = Text(FILTROS[0][0], font_size=23, color=YELLOW)
        rotulo_saida.next_to(saida, UP, buff=0.22)
        self.play(FadeIn(entrada), FadeIn(rotulo_entrada), run_time=0.5)

        passo = entrada.height / original.shape[0]
        caminho = percurso_imagem(entrada, passo, *original.shape[:2])
        for indice in (0, 1):
            nome, kernel, _ = FILTROS[indice]
            cor = BLUE_C if indice == 0 else YELLOW
            sobreposicao = sobreposicao_kernel(kernel, passo, cor)
            sobreposicao.move_to(caminho.get_start())
            painel = sobreposicao_kernel(kernel, 0.46, cor)
            painel.move_to((0, 0, 0))
            rotulo_painel = Text("pesos", font_size=18, color=WHITE)
            rotulo_painel.next_to(painel, UP, buff=0.12)
            if indice == 1:
                novo_rotulo = Text(nome, font_size=23, color=YELLOW)
                novo_rotulo.move_to(rotulo_saida)
                self.play(
                    FadeOut(saida),
                    FadeOut(rotulo_saida),
                    FadeIn(novo_rotulo),
                    run_time=0.35,
                )
                rotulo_saida = novo_rotulo
                saida = imagem_pixelada(aplicar_filtro(original, indice), 5.0)
                saida.move_to(RIGHT * 3.5)
            self.play(
                FadeIn(sobreposicao),
                FadeIn(painel),
                FadeIn(rotulo_painel),
                run_time=0.2,
            )
            self.play(MoveAlongPath(sobreposicao, caminho, rate_func=linear),
                      run_time=2.6)
            self.play(
                FadeIn(saida),
                FadeOut(sobreposicao),
                FadeOut(painel),
                FadeOut(rotulo_painel),
                run_time=0.45,
            )
            self.wait(0.4)

        self.play(
            FadeOut(entrada),
            FadeOut(rotulo_entrada),
            FadeOut(saida),
            FadeOut(rotulo_saida),
            FadeOut(titulo),
            run_time=0.45,
        )
        self.mostrar_comparacao(original)

    def mostrar_comparacao(self, original: np.ndarray) -> None:
        titulo = Text("O mesmo retrato, cinco filtros", font_size=34, color=WHITE)
        titulo.to_edge(UP, buff=0.24)
        self.play(FadeIn(titulo), run_time=0.3)

        conjuntos = [("Original", original)] + [
            (nome, aplicar_filtro(original, i))
            for i, (nome, _, _) in enumerate(FILTROS)
        ]
        miniaturas = Group()
        for indice, (nome, pixels) in enumerate(conjuntos):
            coluna = indice % 3
            linha = indice // 3
            imagem = imagem_pixelada(pixels, 2.18)
            imagem.move_to(((-1 + coluna) * 4.1, 1.32 - linha * 3.2, 0))
            legenda = Text(nome, font_size=19, color=WHITE)
            legenda.next_to(imagem, DOWN, buff=0.12)
            miniaturas.add(imagem, legenda)

        self.play(FadeIn(miniaturas), run_time=0.7)
        nota = Text(
            "Bordas = magnitude da resposta na luminância; blur = média RGB.",
            font_size=17,
            color=WHITE,
        ).to_edge(DOWN, buff=0.16)
        self.play(FadeIn(nota), run_time=0.25)
        self.wait(3)
