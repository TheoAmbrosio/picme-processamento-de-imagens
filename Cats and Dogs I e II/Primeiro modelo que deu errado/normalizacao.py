"""Padronização dos vetores SVD com estatísticas exclusivas do treino."""

import torch
from torch.utils.data import DataLoader, Dataset


def ajustar_padronizacao(treino: Dataset, tamanho_batch: int):
    """Calcula média e desvio populacional por entrada, sem carregar tudo na RAM."""
    total = 0
    media = None
    m2 = None
    for x, _ in DataLoader(treino, batch_size=tamanho_batch, shuffle=False):
        x = x.to(torch.float64)
        variancia_lote, media_lote = torch.var_mean(x, dim=0, correction=0)
        quantidade = len(x)
        if media is None:
            media = media_lote
            m2 = variancia_lote * quantidade
        else:
            diferenca = media_lote - media
            novo_total = total + quantidade
            m2 += (variancia_lote * quantidade
                   + diferenca.square() * total * quantidade / novo_total)
            media += diferenca * quantidade / novo_total
        total += quantidade
    if total == 0:
        raise ValueError("O conjunto de treino não pode estar vazio.")
    desvio = (m2 / total).clamp_min(0).sqrt().clamp_min(1e-6)
    return media.float(), desvio.float()


class DadosPadronizados(Dataset):
    """Aplica as mesmas estatísticas ao treino, avaliação ou novas imagens."""

    def __init__(self, conjunto: Dataset, media: torch.Tensor, desvio: torch.Tensor):
        self.conjunto = conjunto
        self.media = media
        self.desvio = desvio

    def __len__(self):
        return len(self.conjunto)

    def __getitem__(self, indice):
        x, y = self.conjunto[indice]
        return (x - self.media) / self.desvio, y
