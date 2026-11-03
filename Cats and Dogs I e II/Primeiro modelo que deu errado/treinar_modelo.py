"""Configure abaixo e execute: python treinar_modelo.py"""

from pathlib import Path
import math
import time

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from preparar_dados import ConjuntoImagensSVD, TAMANHO, garantir_dataset
from normalizacao import DadosPadronizados, ajustar_padronizacao


# Configurações editáveis
k_modelo = 5
escala_cinza = True
usar_todos_dados = False
N = 1000  # total; usado somente quando usar_todos_dados=False
semente_amostragem = 42
fracao_treino = 0.90
taxa_aprendizado = 0.0003
epocas = 100
tamanho_batch = 128
alpha = 0.0  # configuração do terceiro colocado na busca; 0 desativa
alpha2 = 0.0  # peso do desvio da média atual do lote; com gradiente
epsilon_penalizacao = 0.001  # limita a singularidade perto do centro
fator_reducao = 0.5  # proporcao da largura anterior
numero_camadas_ocultas = 8


def penalizacao_media_anterior(previsoes: torch.Tensor, centro: float) -> torch.Tensor:
    """O centro vem só do treino anterior e é constante na época atual."""
    return alpha * (1.0 / ((previsoes - centro).abs() + epsilon_penalizacao)).mean()


def penalizacao_equilibrio(previsoes: torch.Tensor) -> torch.Tensor:
    """A média atual permanece no grafo para influenciar os pesos da rede."""
    return alpha2 * (0.5 - previsoes.mean()).square()


def larguras_rede(entrada_dim: int, fator: float, n_ocultas: int) -> list[int]:
    """Gera larguras validas; nao permite camadas vazias ou nulas."""
    if not 0 < fator <= 1 or not math.isfinite(fator):
        raise ValueError("fator deve estar em (0, 1].")
    if not isinstance(n_ocultas, int) or n_ocultas < 0:
        raise ValueError("n_ocultas deve ser inteiro nao negativo.")
    larguras = [entrada_dim]
    for _ in range(n_ocultas):
        seguinte = int(larguras[-1] * fator)
        if seguinte < 1:
            raise ValueError(f"Arquitetura inviavel: {larguras[-1]} * {fator} < 1; "
                             "reduza o numero de camadas ou aumente o fator.")
        larguras.append(seguinte)
    return larguras


class RedeNeural(nn.Module):
    def __init__(self, entrada_dim: int, fator: float | None = None,
                 n_ocultas: int | None = None):
        super().__init__()
        fator = fator_reducao if fator is None else fator
        n_ocultas = numero_camadas_ocultas if n_ocultas is None else n_ocultas
        larguras = larguras_rede(entrada_dim, fator, n_ocultas)
        camadas = []
        for atual, seguinte in zip(larguras[:-1], larguras[1:]):
            camadas.extend((nn.Linear(atual, seguinte), nn.ReLU()))
        # Saida sigmoide: previsoes no intervalo [0, 1].
        camadas.extend((nn.Linear(larguras[-1], 1), nn.Sigmoid()))
        self.rede = nn.Sequential(*camadas)
        self.larguras = larguras

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rede(x)


def resumir_intervalos(contagens: list[int]) -> str:
    """Mostra até dois intervalos ocupados; em empate, prioriza o menor limite."""
    total = sum(contagens)
    maiores = sorted(range(10), key=lambda i: (-contagens[i], i))
    partes = []
    for i in maiores[:2]:
        if contagens[i] == 0:
            continue
        fechamento = "]" if i == 9 else ")"
        partes.append(f"[{i / 10:.1f}, {(i + 1) / 10:.1f}{fechamento}: "
                      f"{contagens[i]} ({100 * contagens[i] / total:.1f}%)")
    return "; ".join(partes) if partes else "sem previsões"


def avaliar_no_conjunto(modelo: nn.Module, carregador: DataLoader, criterio: nn.Module,
                       dispositivo: torch.device) -> tuple[float, float, str]:
    modelo.eval()
    perda_total = 0.0
    perda_arredondada_total = 0.0
    total = 0
    contagens = torch.zeros(10, dtype=torch.long, device=dispositivo)
    with torch.no_grad():
        for x, y in carregador:
            x, y = x.to(dispositivo), y.to(dispositivo)
            previsoes = modelo(x)
            perda_total += criterio(previsoes, y).item() * len(x)
            classes = (previsoes >= 0.5).to(y.dtype)
            perda_arredondada_total += criterio(classes, y).item() * len(x)
            limites = torch.arange(1, 10, device=dispositivo, dtype=previsoes.dtype) / 10
            indices = torch.bucketize(previsoes.flatten().contiguous(), limites, right=True)
            contagens += torch.bincount(indices, minlength=10)
            total += len(x)
    return (perda_total / total, perda_arredondada_total / total,
            resumir_intervalos(contagens.cpu().tolist()))


def treinar(modelo: nn.Module, treino: DataLoader, avaliacao: DataLoader,
           dispositivo: torch.device) -> dict[str, list[float]]:
    criterio = nn.MSELoss()
    otimizador = torch.optim.Adam(modelo.parameters(), lr=taxa_aprendizado)
    historico = {"treino": [], "avaliacao": [], "avaliacao_arredondada": []}
    inicio = time.perf_counter()
    centro_penalizacao = 0.5
    for epoca in range(epocas):
        modelo.train()
        perda_total = 0.0
        penalizacao_total = 0.0
        equilibrio_total = 0.0
        soma_previsoes = torch.zeros((), dtype=torch.float64, device=dispositivo)
        total = 0
        for x, y in treino:
            x, y = x.to(dispositivo), y.to(dispositivo)
            otimizador.zero_grad()
            previsoes = modelo(x)
            mse = criterio(previsoes, y)
            penalizacao = penalizacao_media_anterior(previsoes, centro_penalizacao)
            soma_previsoes += previsoes.detach().to(torch.float64).sum()
            equilibrio = penalizacao_equilibrio(previsoes)
            perda = mse + penalizacao + equilibrio
            perda.backward()
            otimizador.step()
            perda_total += mse.item() * len(x)
            penalizacao_total += penalizacao.item() * len(x)
            equilibrio_total += equilibrio.item() * len(x)
            total += len(x)
        mse_treino = perda_total / total
        penalizacao_media = penalizacao_total / total
        equilibrio_medio = equilibrio_total / total
        media_previsoes = soma_previsoes.item() / total
        mse_avaliacao, mse_arredondado, intervalos = avaliar_no_conjunto(
            modelo, avaliacao, criterio, dispositivo)
        historico["treino"].append(mse_treino)
        historico["avaliacao"].append(mse_avaliacao)
        historico["avaliacao_arredondada"].append(mse_arredondado)
        restante = int((time.perf_counter() - inicio) / (epoca + 1) * (epocas - epoca - 1))
        horas, resto = divmod(restante, 3600)
        minutos, segundos = divmod(resto, 60)
        print(f"Época {epoca + 1}/{epocas} | MSE treino={mse_treino:.6f} | "
              f"MSE avaliação={mse_avaliacao:.6f} | "
              f"MSE avaliação arredondado={mse_arredondado:.6f} | "
              f"penalização={penalizacao_media:.6f} | "
              f"penalização equilíbrio={equilibrio_medio:.6f} | "
              f"centro penalização={centro_penalizacao:.6f} | "
              f"média previsões treino={media_previsoes:.6f} | "
              f"perda treino={mse_treino + penalizacao_media + equilibrio_medio:.6f} | "
              f"intervalos avaliação={intervalos} | "
              f"restante={horas:02d}:{minutos:02d}:{segundos:02d}",
              flush=True)
        centro_penalizacao = media_previsoes
    return historico


def plotar(historico: dict[str, list[float]]) -> None:
    plt.figure()
    plt.plot(range(1, len(historico["treino"]) + 1), historico["treino"], label="Treino")
    plt.plot(range(1, len(historico["avaliacao"]) + 1), historico["avaliacao"], label="Avaliação")
    plt.xlabel("Época")
    plt.ylabel("MSE")
    plt.title("Evolução do MSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plotar_frequencias(modelo: nn.Module, carregador: DataLoader,
                      dispositivo: torch.device, destino: Path) -> None:
    """Cria o histograma final da avaliação sem guardar todos os vetores na GPU."""
    modelo.eval()
    contagens = torch.zeros(10, dtype=torch.long, device=dispositivo)
    with torch.no_grad():
        for x, _ in carregador:
            previsoes = modelo(x.to(dispositivo)).flatten().contiguous()
            limites = torch.arange(1, 10, device=dispositivo, dtype=previsoes.dtype) / 10
            indices = torch.bucketize(previsoes, limites, right=True)
            contagens += torch.bincount(indices, minlength=10)
    frequencias = contagens.cpu().tolist()
    fig, ax = plt.subplots()
    barras = ax.bar([(i + 0.5) / 10 for i in range(10)], frequencias,
                    width=0.1, edgecolor="black")
    ax.bar_label(barras, padding=3)
    ax.set_xticks([i / 10 for i in range(11)])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(max(frequencias), 1) * 1.2)
    ax.set_xlabel("Previsão da rede (antes do arredondamento)")
    ax.set_ylabel("Quantidade de imagens")
    ax.set_title("Frequência das previsões — avaliação final")
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    print(f"Histograma final salvo em: {destino}", flush=True)


def main() -> None:
    print("Iniciando preparação do treinamento...", flush=True)
    for nome, valor in (("alpha", alpha), ("alpha2", alpha2)):
        if not math.isfinite(valor) or valor < 0:
            raise ValueError(f"{nome} deve ser finito e não negativo.")
    if not math.isfinite(epsilon_penalizacao) or epsilon_penalizacao <= 0:
        raise ValueError("epsilon_penalizacao deve ser finito e positivo.")
    if not isinstance(epocas, int) or epocas < 1:
        raise ValueError("epocas deve ser inteiro positivo.")
    if not isinstance(tamanho_batch, int) or tamanho_batch < 1:
        raise ValueError("tamanho_batch deve ser inteiro positivo.")
    if not 0 < fracao_treino < 1:
        raise ValueError("fracao_treino deve estar entre 0 e 1.")
    if not math.isfinite(taxa_aprendizado) or taxa_aprendizado <= 0:
        raise ValueError("taxa_aprendizado deve ser finita e positiva.")

    pasta_projeto = Path(__file__).resolve().parent
    diretorio = pasta_projeto / "dataset" / "PetImages"
    garantir_dataset(diretorio)
    dados = ConjuntoImagensSVD(diretorio, k_modelo, escala_cinza,
                              usar_todos_dados, N, semente_amostragem)
    n_treino = int(fracao_treino * len(dados))
    n_avaliacao = len(dados) - n_treino
    if min(n_treino, n_avaliacao) < 1:
        raise ValueError("A divisão precisa deixar imagens para treino e avaliação.")
    treino, avaliacao = random_split(
        dados, [n_treino, n_avaliacao],
        generator=torch.Generator().manual_seed(semente_amostragem),
    )
    print(f"Total={len(dados)} | treino={n_treino} | avaliação={n_avaliacao}", flush=True)
    print("Preparando os vetores SVD; arquivos em cache serão reutilizados...", flush=True)
    for indice in range(len(dados)):
        dados[indice]
        if (indice + 1) % 100 == 0 or indice + 1 == len(dados):
            print(f"SVD: {indice + 1}/{len(dados)}", flush=True)

    print("Calculando a padronização somente no treino...", flush=True)
    media, desvio = ajustar_padronizacao(treino, tamanho_batch)
    treino = DadosPadronizados(treino, media, desvio)
    avaliacao = DadosPadronizados(avaliacao, media, desvio)
    print("Padronização aplicada ao treino e à avaliação.", flush=True)

    carregador_treino = DataLoader(treino, batch_size=tamanho_batch, shuffle=True)
    carregador_avaliacao = DataLoader(avaliacao, batch_size=tamanho_batch)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = RedeNeural(dados.entrada_dim).to(dispositivo)
    print(f"Treinando em {dispositivo} | entrada={dados.entrada_dim}", flush=True)
    historico = treinar(modelo, carregador_treino, carregador_avaliacao, dispositivo)
    plotar_frequencias(modelo, carregador_avaliacao, dispositivo,
                      pasta_projeto / "frequencia_resultados.png")
    plotar(historico)


if __name__ == "__main__":
    main()

