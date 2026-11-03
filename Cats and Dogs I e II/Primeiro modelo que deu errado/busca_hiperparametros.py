"""
Busca de hiperparâmetros para Cats vs Dogs.

- Testa diferentes valores de k e modos de cor.
- Testa arquiteturas, taxas de aprendizado, alpha e alpha2.
- Escolhe automaticamente o maior batch que cabe na memória.
- Interrompe modelos que apresentam estabilização do MSE.
- Salva checkpoints, históricos, gráficos e ranking.
- Permite retomar experimentos interrompidos.

Execute:
    python busca_hiperparametros.py
"""

from __future__ import annotations

import csv
import gc
import hashlib
import itertools
import json
import math
import os
import random
import signal
import time

from copy import deepcopy
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from torch.utils.data import DataLoader, random_split

import treinar_modelo as base
from preparar_dados import ConjuntoImagensSVD, garantir_dataset
from normalizacao import DadosPadronizados, ajustar_padronizacao


# ============================================================
# CONFIGURAÇÕES
# ============================================================

HORAS_MAXIMAS = 5.0

EPOCAS_POR_TESTE = 100
PACIENCIA = 25

# ------------------------------------------------------------
# CONVERGÊNCIA
# ------------------------------------------------------------

PARAR_SE_CONVERGIR = True

EPOCA_MINIMA_CONVERGENCIA = 5
JANELA_CONVERGENCIA = 5

TOLERANCIA_RELATIVA_CONVERGENCIA = 5e-3
TOLERANCIA_ABSOLUTA_CONVERGENCIA = 2e-3

EXIGIR_TREINO_ESTAVEL = True

EXIGIR_GRADIENTE_PEQUENO = False
TOLERANCIA_GRADIENTE = 1e-3


# ------------------------------------------------------------
# DATASET E SVD
# ------------------------------------------------------------

VALORES_K = [5, 10, 15, 20]

# True = escala de cinza
# False = RGB
MODOS_COR = [True]


# ------------------------------------------------------------
# HIPERPARÂMETROS
# ------------------------------------------------------------

TAXAS_APRENDIZADO = [
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
]

ALPHAS = [0.0, 0.0001, 0.0003, 0.001, 0.003, 0.01]
ALPHAS2 = [0.0, 0.0001, 0.0003, 0.001, 0.003, 0.01]

FATORES_REDUCAO = [
    1/3,
    1/2,
    2/3,
]

CAMADAS_OCULTAS = [
    3,
    5,
    8,
]


# ------------------------------------------------------------
# BATCH AUTOMÁTICO
# ------------------------------------------------------------

BATCH_INICIAL = 16
BATCH_MAXIMO = 256


# ------------------------------------------------------------
# OUTRAS CONFIGURAÇÕES
# ------------------------------------------------------------

SEMENTE = base.semente_amostragem
EPSILON = base.epsilon_penalizacao

PASTA_RESULTADOS = (
    Path(__file__).resolve().parent
    / "resultados_busca_k_cor_v5"
)

PARAR = False


# ============================================================
# CONTROLE DE EXECUÇÃO
# ============================================================

def pedir_parada(_sig, _frame):
    global PARAR

    PARAR = True

    print(
        "\nParada solicitada. "
        "Salvando ao final da época atual...",
        flush=True,
    )


def salvar_json_atomico(caminho, objeto):
    caminho = Path(caminho)

    temporario = caminho.with_suffix(
        caminho.suffix + ".tmp"
    )

    with open(temporario, "w", encoding="utf-8") as arquivo:
        json.dump(
            objeto,
            arquivo,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

        arquivo.flush()
        os.fsync(arquivo.fileno())

    os.replace(temporario, caminho)


def salvar_torch_atomico(caminho, objeto):
    caminho = Path(caminho)

    temporario = caminho.with_suffix(
        caminho.suffix + ".tmp"
    )

    torch.save(objeto, temporario)

    os.replace(temporario, caminho)


# ============================================================
# IDENTIFICAÇÃO DOS EXPERIMENTOS
# ============================================================

def id_teste(lr, alpha, alpha2, k, cinza, fator, camadas):

    def formatar(valor):
        if valor == 0:
            return "0"

        return (
            f"{valor:.0e}"
            .replace("+", "")
            .replace("-", "m")
        )

    fator_texto = (
        f"{fator:.6f}".replace(".", "p")
    )

    cor = "cinza" if cinza else "rgb"

    return (
        f"k_{k}__{cor}"
        f"__lr_{formatar(lr)}"
        f"__a_{formatar(alpha)}"
        f"__a2_{formatar(alpha2)}"
        f"__f_{fator_texto}"
        f"__h_{camadas}"
    )


# ============================================================
# GERAÇÃO DAS COMBINAÇÕES
# ============================================================

def gerar_grade(dimensoes):

    rng = random.Random(SEMENTE)

    grupos = {}

    for (k, cinza), dimensao in dimensoes.items():

        candidatos = []
        prioridades = []

        for fator, camadas in itertools.product(
            FATORES_REDUCAO,
            CAMADAS_OCULTAS,
        ):

            try:
                base.larguras_rede(
                    dimensao,
                    fator,
                    camadas,
                )

            except ValueError:
                continue

            arquitetura = (
                k,
                cinza,
                fator,
                camadas,
            )

            # Baseline sem penalizações.
            if (
                1e-3 in TAXAS_APRENDIZADO
                and 0.0 in ALPHAS
                and 0.0 in ALPHAS2
            ):
                prioridades.append(
                    (1e-3, 0.0, 0.0, *arquitetura)
                )

            for lr, a, a2 in itertools.product(
                TAXAS_APRENDIZADO,
                ALPHAS,
                ALPHAS2,
            ):

                candidatos.append(
                    (lr, a, a2, *arquitetura)
                )

        rng.shuffle(prioridades)
        rng.shuffle(candidatos)

        grupos[(k, cinza)] = list(
            dict.fromkeys(
                prioridades + candidatos
            )
        )

    chaves = list(grupos)

    rng.shuffle(chaves)

    # Alterna grupos, mas executa 12 testes consecutivos
    # com o mesmo k e modo de cor.

    indice = 0

    while True:

        encontrou = False

        for chave in chaves:

            bloco = grupos[chave][
                indice:indice + 12
            ]

            if bloco:
                encontrou = True
                yield from bloco

        if not encontrou:
            break

        indice += 12


# ============================================================
# MEMÓRIA E BATCH AUTOMÁTICO
# ============================================================

def falta_memoria(erro):

    texto = str(erro).lower()

    return (
        isinstance(
            erro,
            torch.cuda.OutOfMemoryError,
        )
        or (
            isinstance(erro, RuntimeError)
            and (
                "out of memory" in texto
                or "can't allocate memory" in texto
                or "not enough memory" in texto
            )
        )
    )


def descobrir_batch(modelo, conjunto_treino, dispositivo):

    limite = min(
        BATCH_MAXIMO,
        len(conjunto_treino),
    )

    batch = min(BATCH_INICIAL, limite)

    melhor = None

    while batch <= limite:

        copia = None
        otimizador = None

        x = None
        y = None
        previsoes = None

        try:

            copia = deepcopy(modelo).to(dispositivo)

            otimizador = torch.optim.Adam(
                copia.parameters(),
                lr=1e-3,
            )

            amostra = [
                conjunto_treino[i]
                for i in range(batch)
            ]

            x = torch.stack([
                item[0] for item in amostra
            ]).to(dispositivo)

            y = torch.stack([
                item[1] for item in amostra
            ]).to(dispositivo)

            previsoes = copia(x)

            mse = (
                (previsoes - y).square().mean()
            )

            mse.backward()

            otimizador.step()

            if dispositivo.type == "cuda":
                torch.cuda.synchronize()

            melhor = batch

            print(
                f"Batch {batch}: OK",
                flush=True,
            )

        except (RuntimeError, MemoryError) as erro:

            if (
                not falta_memoria(erro)
                and not isinstance(erro, MemoryError)
            ):
                raise

            print(
                f"Batch {batch}: memória insuficiente",
                flush=True,
            )

            break

        finally:

            del copia
            del otimizador
            del x
            del y
            del previsoes

            gc.collect()

            if dispositivo.type == "cuda":
                torch.cuda.empty_cache()

        batch *= 2

    if melhor is None:
        raise RuntimeError(
            "Nenhum batch coube na memória. "
            "Reduza BATCH_INICIAL."
        )

    return melhor


# ============================================================
# AVALIAÇÃO
# ============================================================

def diagnosticar(modelo, carregador, dispositivo):

    modelo.eval()

    mse_total = 0.0
    erro_total = 0.0
    total = 0

    histograma = [0] * 10

    with torch.no_grad():

        for x, y in carregador:

            x = x.to(dispositivo)
            y = y.to(dispositivo)

            previsoes = modelo(x)

            mse_total += (
                (previsoes - y)
                .square()
                .sum()
                .item()
            )

            classes = (
                previsoes >= 0.5
            ).to(y.dtype)

            erro_total += (
                (classes - y)
                .square()
                .sum()
                .item()
            )

            indices = torch.clamp(
                (previsoes.flatten() * 10).long(),
                0,
                9,
            )

            frequencias = torch.bincount(
                indices,
                minlength=10,
            ).cpu().tolist()

            histograma = [
                a + b
                for a, b in zip(
                    histograma,
                    frequencias,
                )
            ]

            total += y.numel()

    return (
        mse_total / total,
        erro_total / total,
        histograma,
    )


# ============================================================
# DETECÇÃO DE CONVERGÊNCIA
# ============================================================

def diagnostico_convergencia(historico):

    """
    Detecta estabilização do MSE em uma janela.

    Compara a amplitude dos valores recentes e verifica
    se houve uma mudança abrupta na entrada da janela.

    Não garante matematicamente um mínimo local.
    """

    n = len(historico)
    janela = JANELA_CONVERGENCIA

    minimo = max(
        EPOCA_MINIMA_CONVERGENCIA,
        janela + 1,
    )

    if n < minimo:
        return False, None

    diagnostico = {}

    for chave in (
        "mse_treino",
        "mse_validacao",
    ):

        valores = [
            float(registro[chave])
            for registro in historico[-janela:]
        ]

        anterior = float(
            historico[-janela - 1][chave]
        )

        if not all(
            math.isfinite(v)
            for v in valores + [anterior]
        ):
            return False, None

        media = sum(valores) / janela

        referencia = max(
            abs(media),
            abs(anterior),
            1e-12,
        )

        amplitude = (
            max(valores) - min(valores)
        )

        variacao_borda = abs(
            valores[0] - anterior
        )

        limite = (
            TOLERANCIA_ABSOLUTA_CONVERGENCIA
            +
            TOLERANCIA_RELATIVA_CONVERGENCIA
            * referencia
        )

        estavel = (
            amplitude <= limite
            and variacao_borda <= limite
        )

        diagnostico[chave] = {
            "media": media,
            "amplitude": amplitude,
            "variacao_borda": variacao_borda,
            "limite": limite,
            "estavel": estavel,
        }

    convergiu = (
        diagnostico["mse_validacao"]["estavel"]
    )

    if EXIGIR_TREINO_ESTAVEL:

        convergiu = (
            convergiu
            and diagnostico["mse_treino"]["estavel"]
        )

    return convergiu, diagnostico


# ============================================================
# NORMA DO GRADIENTE
# ============================================================

def norma_gradiente_total(
    modelo,
    carregador,
    dispositivo,
    alpha,
    alpha2,
    centro,
):

    modelo.eval()

    modelo.zero_grad(set_to_none=True)

    total = len(carregador.dataset)

    for x, y in carregador:

        x = x.to(dispositivo)
        y = y.to(dispositivo)

        previsoes = modelo(x)

        perda = (
            (previsoes - y)
            .square()
            .mean()
        )

        if alpha:

            penalizacao = (
                1.0
                /
                (
                    (previsoes - centro).abs()
                    + EPSILON
                )
            ).mean()

            perda = (
                perda + alpha * penalizacao
            )

        if alpha2:

            equilibrio = (
                0.5 - previsoes.mean()
            ).square()

            perda = (
                perda + alpha2 * equilibrio
            )

        peso = len(x) / total

        (perda * peso).backward()

    soma = 0.0

    for parametro in modelo.parameters():

        if parametro.grad is not None:

            soma += (
                parametro.grad.detach()
                .square()
                .sum()
                .item()
            )

    modelo.zero_grad(set_to_none=True)

    return math.sqrt(soma)


# ============================================================
# GRÁFICOS
# ============================================================

def salvar_figuras(pasta, historico):

    if not historico:
        return

    epocas = [
        r["epoca"]
        for r in historico
    ]

    mse_treino = [
        r["mse_treino"]
        for r in historico
    ]

    mse_validacao = [
        r["mse_validacao"]
        for r in historico
    ]

    fig, ax = plt.subplots(
        figsize=(8, 4.5)
    )

    ax.plot(
        epocas,
        mse_treino,
        label="MSE treino",
    )

    ax.plot(
        epocas,
        mse_validacao,
        label="MSE validação",
    )

    ax.set_xlabel("Época")
    ax.set_ylabel("MSE")

    ax.set_title(
        "Evolução do MSE"
    )

    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        pasta / "curva_mse.png",
        dpi=140,
    )

    plt.close(fig)

    # Histograma das previsões.

    ultimo = historico[-1]

    frequencias = ultimo["hist_validacao"]

    fig, ax = plt.subplots(
        figsize=(8, 4.5)
    )

    posicoes = [
        (i + 0.5) / 10
        for i in range(10)
    ]

    barras = ax.bar(
        posicoes,
        frequencias,
        width=0.092,
        edgecolor="black",
    )

    ax.bar_label(
        barras,
        padding=2,
        fontsize=8,
    )

    ax.set_xticks([
        i / 10 for i in range(11)
    ])

    ax.set_xlim(0, 1)

    ax.set_ylim(
        0,
        max(max(frequencias), 1) * 1.2,
    )

    ax.set_xlabel(
        "Probabilidade prevista"
    )

    ax.set_ylabel(
        "Quantidade"
    )

    ax.set_title(
        f"Histograma — época {ultimo['epoca']}"
    )

    fig.tight_layout()

    fig.savefig(
        pasta / "histograma_validacao.png",
        dpi=140,
    )

    plt.close(fig)


# ============================================================
# RANKING
# ============================================================

COLUNAS_RANKING = [
    "id",
    "status",
    "k",
    "escala_cinza",
    "entrada_dim",
    "parametros",
    "batch",
    "lr",
    "alpha",
    "alpha2",
    "fator",
    "camadas",
    "epoca",
    "melhor_epoca",
    "melhor_mse_validacao",
    "melhor_mse_arredondado",
    "segundos",
    "convergiu",
    "norma_gradiente",
    "motivo_parada",
]


def atualizar_ranking():

    resultados = []

    for caminho in (
        PASTA_RESULTADOS / "testes"
    ).glob("*/estado.json"):

        estado = json.loads(
            caminho.read_text(
                encoding="utf-8"
            )
        )

        if (
            estado.get("epoca", 0) < 1
            or estado.get(
                "melhor_mse_validacao"
            ) is None
        ):
            continue

        resultados.append({
            coluna: estado.get(coluna)
            for coluna in COLUNAS_RANKING
        })

    resultados.sort(
        key=lambda x: (
            x["melhor_mse_validacao"],
            x["id"],
        )
    )

    destino = (
        PASTA_RESULTADOS / "ranking.csv"
    )

    temporario = destino.with_suffix(
        ".csv.tmp"
    )

    with open(
        temporario,
        "w",
        newline="",
        encoding="utf-8",
    ) as arquivo:

        escritor = csv.DictWriter(
            arquivo,
            fieldnames=COLUNAS_RANKING,
        )

        escritor.writeheader()
        escritor.writerows(resultados)

    os.replace(
        temporario,
        destino,
    )

    if resultados:

        salvar_json_atomico(
            PASTA_RESULTADOS / "melhor_resultado.json",
            resultados[0],
        )


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main():

    global PARAR

    signal.signal(
        signal.SIGINT,
        pedir_parada,
    )

    signal.signal(
        signal.SIGTERM,
        pedir_parada,
    )

    prazo = (
        time.monotonic()
        + HORAS_MAXIMAS * 3600
    )

    PASTA_RESULTADOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    pasta_testes = (
        PASTA_RESULTADOS / "testes"
    )

    pasta_testes.mkdir(
        exist_ok=True
    )

    configuracao = {
        "valores_k": VALORES_K,
        "modos_cor": MODOS_COR,
        "lrs": TAXAS_APRENDIZADO,
        "alphas": ALPHAS,
        "alphas2": ALPHAS2,
        "fatores": FATORES_REDUCAO,
        "camadas": CAMADAS_OCULTAS,
        "epocas": EPOCAS_POR_TESTE,
        "paciencia": PACIENCIA,
        "batch_inicial": BATCH_INICIAL,
        "batch_maximo": BATCH_MAXIMO,
        "usar_todos_dados": base.usar_todos_dados,
        "N": base.N,
        "fracao_treino": base.fracao_treino,
        "semente": SEMENTE,
        "epsilon": EPSILON,
        "versao": 5,
        "convergencia": {
            "ativar": PARAR_SE_CONVERGIR,
            "epoca_minima": EPOCA_MINIMA_CONVERGENCIA,
            "janela": JANELA_CONVERGENCIA,
            "tolerancia_relativa":
                TOLERANCIA_RELATIVA_CONVERGENCIA,
            "tolerancia_absoluta":
                TOLERANCIA_ABSOLUTA_CONVERGENCIA,
            "exigir_treino_estavel":
                EXIGIR_TREINO_ESTAVEL,
            "exigir_gradiente":
                EXIGIR_GRADIENTE_PEQUENO,
            "tolerancia_gradiente":
                TOLERANCIA_GRADIENTE,
        },
    }

    assinatura = hashlib.sha256(
        json.dumps(
            configuracao,
            sort_keys=True,
        ).encode()
    ).hexdigest()

    caminho_config = (
        PASTA_RESULTADOS / "configuracao.json"
    )

    if caminho_config.exists():

        anterior = json.loads(
            caminho_config.read_text(
                encoding="utf-8"
            )
        )

        if anterior["assinatura"] != assinatura:

            raise RuntimeError(
                "Configuração alterada. "
                "Use outra PASTA_RESULTADOS."
            )

    else:

        salvar_json_atomico(
            caminho_config,
            {
                "assinatura": assinatura,
                "configuracao": configuracao,
            },
        )

    diretorio = (
        Path(__file__).resolve().parent
        / "dataset"
        / "PetImages"
    )

    garantir_dataset(diretorio)

    dispositivo = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Dispositivo: {dispositivo}",
        flush=True,
    )

    # --------------------------------------------------------
    # DIMENSÕES DAS ENTRADAS
    # --------------------------------------------------------

    dimensoes = {}

    for k, cinza in itertools.product(
        VALORES_K,
        MODOS_COR,
    ):

        dados_temporarios = ConjuntoImagensSVD(
            diretorio,
            k,
            cinza,
            base.usar_todos_dados,
            base.N,
            SEMENTE,
        )

        dimensoes[(k, cinza)] = (
            dados_temporarios.entrada_dim
        )

        del dados_temporarios

    # --------------------------------------------------------
    # EXPERIMENTOS
    # --------------------------------------------------------

    chave_ativa = None
    dados_ativos = None
    treino = None
    validacao = None

    grade = gerar_grade(dimensoes)

    for (
        lr,
        alpha,
        alpha2,
        k,
        cinza,
        fator,
        camadas,
    ) in grade:

        if PARAR or time.monotonic() >= prazo:
            break

        nome = id_teste(
            lr,
            alpha,
            alpha2,
            k,
            cinza,
            fator,
            camadas,
        )

        pasta = pasta_testes / nome

        estado_path = (
            pasta / "estado.json"
        )

        checkpoint_path = (
            pasta / "checkpoint_atual.pt"
        )

        historico_path = (
            pasta / "historico.json"
        )

        # Não repetir testes já finalizados.

        if estado_path.exists():

            estado = json.loads(
                estado_path.read_text(
                    encoding="utf-8"
                )
            )

            if estado["status"] in (
                "concluido",
                "early_stopping",
                "convergiu",
            ):
                continue

        else:
            estado = None

        # ----------------------------------------------------
        # PREPARAÇÃO DOS DADOS
        # ----------------------------------------------------

        if (k, cinza) != chave_ativa:

            print(
                f"\nPreparando k={k}, "
                f"cor={'cinza' if cinza else 'RGB'}",
                flush=True,
            )

            dados_ativos = ConjuntoImagensSVD(
                diretorio,
                k,
                cinza,
                base.usar_todos_dados,
                base.N,
                SEMENTE,
            )

            n_treino = int(
                base.fracao_treino
                * len(dados_ativos)
            )

            n_validacao = (
                len(dados_ativos) - n_treino
            )

            if min(n_treino, n_validacao) < 1:

                raise ValueError(
                    "Treino ou validação vazios."
                )

            subconjunto_treino, subconjunto_val = (
                random_split(
                    dados_ativos,
                    [
                        n_treino,
                        n_validacao,
                    ],
                    generator=torch.Generator().manual_seed(
                        SEMENTE
                    ),
                )
            )

            # Calcula ou reutiliza o cache SVD.

            for i in range(len(dados_ativos)):

                if (
                    PARAR
                    or time.monotonic() >= prazo
                ):
                    break

                dados_ativos[i]

                if (i + 1) % 200 == 0:

                    print(
                        f"SVD: {i + 1}/{len(dados_ativos)}",
                        flush=True,
                    )

            if PARAR or time.monotonic() >= prazo:
                break

            media, desvio = ajustar_padronizacao(
                subconjunto_treino,
                min(64, n_treino),
            )

            treino = DadosPadronizados(
                subconjunto_treino,
                media,
                desvio,
            )

            validacao = DadosPadronizados(
                subconjunto_val,
                media,
                desvio,
            )

            chave_ativa = (k, cinza)

        # ----------------------------------------------------
        # MODELO
        # ----------------------------------------------------

        pasta.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.manual_seed(SEMENTE)

        if dispositivo.type == "cuda":
            torch.cuda.manual_seed_all(SEMENTE)

        modelo = base.RedeNeural(
            dados_ativos.entrada_dim,
            fator=fator,
            n_ocultas=camadas,
        ).to(dispositivo)

        quantidade_parametros = sum(
            p.numel()
            for p in modelo.parameters()
        )

        if estado is None:

            batch = descobrir_batch(
                modelo,
                treino,
                dispositivo,
            )

            estado = {
                "id": nome,
                "k": k,
                "escala_cinza": cinza,
                "entrada_dim":
                    dados_ativos.entrada_dim,
                "parametros":
                    quantidade_parametros,
                "batch": batch,
                "lr": lr,
                "alpha": alpha,
                "alpha2": alpha2,
                "fator": fator,
                "camadas": camadas,
                "epoca": 0,
                "status": "em_andamento",
                "melhor_mse_validacao": None,
                "melhor_epoca": 0,
                "melhor_mse_arredondado": None,
                "segundos": 0.0,
                "convergiu": False,
                "norma_gradiente": None,
                "motivo_parada": None,
            }

        else:

            batch = estado["batch"]

            if not checkpoint_path.exists():

                raise RuntimeError(
                    f"Checkpoint ausente: {nome}"
                )

        otimizador = torch.optim.Adam(
            modelo.parameters(),
            lr=lr,
        )

        gerador = (
            torch.Generator()
            .manual_seed(SEMENTE + 177)
        )

        treino_loader = DataLoader(
            treino,
            batch_size=batch,
            shuffle=True,
            generator=gerador,
        )

        validacao_loader = DataLoader(
            validacao,
            batch_size=batch,
            shuffle=False,
        )

        centro = 0.5
        historico = []

        # ----------------------------------------------------
        # RETOMAR CHECKPOINT
        # ----------------------------------------------------

        if checkpoint_path.exists():

            checkpoint = torch.load(
                checkpoint_path,
                map_location=dispositivo,
                weights_only=False,
            )

            modelo.load_state_dict(
                checkpoint["modelo"]
            )

            otimizador.load_state_dict(
                checkpoint["otimizador"]
            )

            gerador.set_state(
                checkpoint["gerador"]
            )

            centro = checkpoint["centro"]
            estado = checkpoint["estado"]
            historico = checkpoint["historico"]

        else:

            salvar_json_atomico(
                estado_path,
                estado,
            )

        print(
            f"\n[{datetime.now().isoformat(timespec='seconds')}]"
            f"\nModelo: {nome}"
            f"\nParâmetros: {quantidade_parametros:,}"
            f"\nBatch: {batch}"
            f"\nPróxima época: {estado['epoca'] + 1}",
            flush=True,
        )

        inicio_teste = time.monotonic()

        # ----------------------------------------------------
        # TREINAMENTO
        # ----------------------------------------------------

        for epoca in range(
            estado["epoca"] + 1,
            EPOCAS_POR_TESTE + 1,
        ):

            if PARAR or time.monotonic() >= prazo:
                break

            modelo.train()

            soma_mse = 0.0
            soma_p1 = 0.0
            soma_p2 = 0.0
            soma_previsoes = 0.0
            quantidade = 0

            for x, y in treino_loader:

                x = x.to(dispositivo)
                y = y.to(dispositivo)

                otimizador.zero_grad(
                    set_to_none=True
                )

                previsoes = modelo(x)

                mse = (
                    (previsoes - y)
                    .square()
                    .mean()
                )

                if alpha:

                    p1 = alpha * (
                        1.0
                        /
                        (
                            (previsoes - centro).abs()
                            + EPSILON
                        )
                    ).mean()

                else:

                    p1 = previsoes.new_zeros(())

                if alpha2:

                    p2 = alpha2 * (
                        0.5 - previsoes.mean()
                    ).square()

                else:

                    p2 = previsoes.new_zeros(())

                perda = mse + p1 + p2

                perda.backward()

                otimizador.step()

                n = y.numel()

                soma_mse += mse.item() * n
                soma_p1 += p1.item() * n
                soma_p2 += p2.item() * n

                soma_previsoes += (
                    previsoes.detach()
                    .sum()
                    .item()
                )

                quantidade += n

            # ------------------------------------------------
            # VALIDAÇÃO
            # ------------------------------------------------

            (
                mse_validacao,
                erro_arredondado,
                histograma,
            ) = diagnosticar(
                modelo,
                validacao_loader,
                dispositivo,
            )

            mse_treino = (
                soma_mse / quantidade
            )

            media_previsoes = (
                soma_previsoes / quantidade
            )

            registro = {
                "epoca": epoca,
                "mse_treino": mse_treino,
                "mse_validacao": mse_validacao,
                "mse_validacao_arredondado":
                    erro_arredondado,
                "penalizacao_alpha":
                    soma_p1 / quantidade,
                "penalizacao_alpha2":
                    soma_p2 / quantidade,
                "media_previsoes":
                    media_previsoes,
                "centro_anterior": centro,
                "hist_validacao": histograma,
            }

            centro = media_previsoes

            historico.append(registro)

            # ------------------------------------------------
            # MELHOR MODELO
            # ------------------------------------------------

            melhor_anterior = estado[
                "melhor_mse_validacao"
            ]

            melhorou = (
                math.isfinite(mse_validacao)
                and (
                    melhor_anterior is None
                    or mse_validacao
                    < melhor_anterior - 1e-10
                )
            )

            if melhorou:

                estado[
                    "melhor_mse_validacao"
                ] = mse_validacao

                estado["melhor_epoca"] = epoca

                estado[
                    "melhor_mse_arredondado"
                ] = erro_arredondado

                salvar_torch_atomico(
                    pasta / "melhor_modelo.pt",
                    {
                        "modelo": modelo.state_dict(),
                        "entrada_dim":
                            dados_ativos.entrada_dim,
                        "k": k,
                        "escala_cinza": cinza,
                        "batch": batch,
                        "lr": lr,
                        "alpha": alpha,
                        "alpha2": alpha2,
                        "fator": fator,
                        "camadas": camadas,
                        "larguras": modelo.larguras,
                        "epoca": epoca,
                        "mse_validacao":
                            mse_validacao,
                        "configuracao":
                            configuracao,
                    },
                )

            # ------------------------------------------------
            # CONVERGÊNCIA
            # ------------------------------------------------

            convergiu = False
            diagnostico = None
            norma_gradiente = None

            if PARAR_SE_CONVERGIR:

                (
                    convergiu,
                    diagnostico,
                ) = diagnostico_convergencia(
                    historico
                )

            if (
                convergiu
                and EXIGIR_GRADIENTE_PEQUENO
            ):

                norma_gradiente = (
                    norma_gradiente_total(
                        modelo,
                        treino_loader,
                        dispositivo,
                        alpha,
                        alpha2,
                        centro,
                    )
                )

                convergiu = (
                    math.isfinite(norma_gradiente)
                    and norma_gradiente
                    <= TOLERANCIA_GRADIENTE
                )

            registro["convergencia"] = diagnostico

            registro[
                "norma_gradiente"
            ] = norma_gradiente

            estado["convergiu"] = bool(
                convergiu
            )

            estado[
                "norma_gradiente"
            ] = norma_gradiente

            estado["epoca"] = epoca

            estado["segundos"] += (
                time.monotonic() - inicio_teste
            )

            inicio_teste = time.monotonic()

            sem_melhora = (
                epoca - estado["melhor_epoca"]
            )

            # ------------------------------------------------
            # CRITÉRIOS DE PARADA
            # ------------------------------------------------

            if convergiu:

                estado["status"] = "convergiu"

                estado["motivo_parada"] = (
                    "MSE estabilizado"
                )

            elif (
                PACIENCIA
                and sem_melhora >= PACIENCIA
            ):

                estado["status"] = (
                    "early_stopping"
                )

                estado["motivo_parada"] = (
                    "Validação sem melhora"
                )

            elif epoca >= EPOCAS_POR_TESTE:

                estado["status"] = "concluido"

                estado["motivo_parada"] = (
                    "Limite de épocas"
                )

            else:

                estado["status"] = (
                    "em_andamento"
                )

                estado["motivo_parada"] = None

            finalizar = (
                estado["status"] != "em_andamento"
            )

            # ------------------------------------------------
            # SALVAMENTO
            # ------------------------------------------------

            salvar_torch_atomico(
                checkpoint_path,
                {
                    "modelo": modelo.state_dict(),
                    "otimizador":
                        otimizador.state_dict(),
                    "gerador": gerador.get_state(),
                    "centro": centro,
                    "estado": estado.copy(),
                    "historico": historico,
                },
            )

            salvar_json_atomico(
                historico_path,
                historico,
            )

            salvar_json_atomico(
                estado_path,
                estado,
            )

            salvar_figuras(
                pasta,
                historico,
            )

            atualizar_ranking()

            # ------------------------------------------------
            # LOG NO TERMINAL
            # ------------------------------------------------

            tempo_restante = max(
                0,
                prazo - time.monotonic(),
            ) / 60

            print(
                f"{nome}"
                f" | época={epoca}/{EPOCAS_POR_TESTE}"
                f" | treino={mse_treino:.6f}"
                f" | val={mse_validacao:.6f}"
                f" | arredondado={erro_arredondado:.4f}"
                f" | melhor={estado['melhor_mse_validacao']:.6f}"
                f" | convergiu={convergiu}"
                f" | status={estado['status']}"
                f" | restante={tempo_restante:.1f} min",
                flush=True,
            )

            if finalizar:
                break

    print(
        "\nBusca encerrada.",
        flush=True,
    )

    print(
        f"Resultados: {PASTA_RESULTADOS}",
        flush=True,
    )

    print(
        "Execute novamente para continuar "
        "os experimentos pendentes.",
        flush=True,
    )


if __name__ == "__main__":
    main()