"""Configure abaixo e execute: python treinar_modelo.py"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from preparar_dados import ConjuntoImagensSVD, TAMANHO, garantir_dataset


# Configurações editáveis
k_modelo = 20
escala_cinza = True
usar_todos_dados = False
N = 1000  # total; usado somente quando usar_todos_dados=False
semente_amostragem = 42
fracao_treino = 0.90
taxa_aprendizado = 0.1
epocas = 100
tamanho_batch = 50


class RedeNeural(nn.Module):
    def __init__(self, entrada_dim: int):
        super().__init__()
        camadas = []
        largura = entrada_dim
        for _ in range(10):
            seguinte = (2 * largura) // 3
            if seguinte < 1:
                raise ValueError("A regra de 2/3 produziu uma camada vazia.")
            camadas.extend((nn.Linear(largura, seguinte), nn.Sigmoid()))
            largura = seguinte
        camadas.extend((nn.Linear(largura, 1), nn.Sigmoid()))
        self.rede = nn.Sequential(*camadas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rede(x)


def mse_no_conjunto(modelo: nn.Module, carregador: DataLoader, criterio: nn.Module,
                    dispositivo: torch.device) -> float:
    modelo.eval()
    perda_total = 0.0
    total = 0
    with torch.no_grad():
        for x, y in carregador:
            x, y = x.to(dispositivo), y.to(dispositivo)
            perda_total += criterio(modelo(x), y).item() * len(x)
            total += len(x)
    return perda_total / total


def treinar(modelo: nn.Module, treino: DataLoader, avaliacao: DataLoader,
           dispositivo: torch.device) -> dict[str, list[float]]:
    criterio = nn.MSELoss()
    otimizador = torch.optim.Adam(modelo.parameters(), lr=taxa_aprendizado)
    historico = {"treino": [], "avaliacao": []}
    inicio = time.perf_counter()
    for epoca in range(epocas):
        modelo.train()
        perda_total = 0.0
        total = 0
        for x, y in treino:
            x, y = x.to(dispositivo), y.to(dispositivo)
            otimizador.zero_grad()
            perda = criterio(modelo(x), y)
            perda.backward()
            otimizador.step()
            perda_total += perda.item() * len(x)
            total += len(x)
        mse_treino = perda_total / total
        mse_avaliacao = mse_no_conjunto(modelo, avaliacao, criterio, dispositivo)
        historico["treino"].append(mse_treino)
        historico["avaliacao"].append(mse_avaliacao)
        restante = int((time.perf_counter() - inicio) / (epoca + 1) * (epocas - epoca - 1))
        horas, resto = divmod(restante, 3600)
        minutos, segundos = divmod(resto, 60)
        print(f"Época {epoca + 1}/{epocas} | MSE treino={mse_treino:.6f} | "
              f"MSE avaliação={mse_avaliacao:.6f} | restante={horas:02d}:{minutos:02d}:{segundos:02d}",
              flush=True)
    return historico


def plotar(historico: dict[str, list[float]]) -> None:
    plt.plot(range(1, len(historico["treino"]) + 1), historico["treino"], label="Treino")
    plt.plot(range(1, len(historico["avaliacao"]) + 1), historico["avaliacao"], label="Avaliação")
    plt.xlabel("Época")
    plt.ylabel("MSE")
    plt.title("Evolução do MSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main() -> None:
    if not isinstance(epocas, int) or epocas < 1:
        raise ValueError("epocas deve ser inteiro positivo.")
    if not isinstance(tamanho_batch, int) or tamanho_batch < 1:
        raise ValueError("tamanho_batch deve ser inteiro positivo.")
    if not 0 < fracao_treino < 1:
        raise ValueError("fracao_treino deve estar entre 0 e 1.")
    if taxa_aprendizado <= 0:
        raise ValueError("taxa_aprendizado deve ser positiva.")

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
    print(f"Total={len(dados)} | treino={n_treino} | avaliação={n_avaliacao}")
    print("Preparando os vetores SVD; arquivos em cache serão reutilizados...")
    for indice in range(len(dados)):
        dados[indice]
        if (indice + 1) % 100 == 0 or indice + 1 == len(dados):
            print(f"SVD: {indice + 1}/{len(dados)}", flush=True)

    carregador_treino = DataLoader(treino, batch_size=tamanho_batch, shuffle=True)
    carregador_avaliacao = DataLoader(avaliacao, batch_size=tamanho_batch)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = RedeNeural(dados.entrada_dim).to(dispositivo)
    print(f"Treinando em {dispositivo} | entrada={dados.entrada_dim}")
    historico = treinar(modelo, carregador_treino, carregador_avaliacao, dispositivo)
    plotar(historico)


if __name__ == "__main__":
    main()
