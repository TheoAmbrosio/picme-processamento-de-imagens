"""Lê somente os splits explícitos; não sorteia nem amplia durante a avaliação."""
import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .res_theo34 import PESOS

ROTULOS = {"Cat": 0, "Dog": 1}
SPLITS = ("treino", "validacao", "teste")
PRESET = PESOS.transforms()
NORMALIZAR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=PRESET.mean, std=PRESET.std),
])


def normalizar_imagem(imagem: Image.Image) -> torch.Tensor:
    # Os JPEGs já têm 224x224; aplicar só escala [0, 1] e normalização ImageNet uma vez.
    return NORMALIZAR(imagem)


def carregar_divisao(manifesto: Path, splits_obrigatorios=SPLITS, aceitar_rotulos_duplicados=False) -> dict:
    conjuntos = {split: [] for split in SPLITS}
    hashes = {}
    originais = {}
    caminhos = set()
    raiz = manifesto.resolve().parent
    with manifesto.open(encoding="utf-8", newline="") as arquivo:
        leitor = csv.DictReader(arquivo)
        obrigatorias = {"split", "classe", "rotulo", "sha256", "caminho", "arquivo_origem",
                       "geometria", "espelho", "rotacao"}
        if not obrigatorias.issubset(leitor.fieldnames or []):
            raise ValueError("Manifesto antigo/incompleto; use criar_dataset_ampliado.py atualizado.")
        for linha in leitor:
            split, classe, sha = linha["split"], linha["classe"], linha["sha256"]
            if split not in conjuntos or classe not in ROTULOS or len(sha) != 64:
                raise ValueError("Manifesto inválido ou antigo; refaça com criar_dataset_ampliado.py.")
            if int(linha["rotulo"]) != ROTULOS[classe]:
                raise ValueError("Rótulos incompatíveis: esperado Cat=0 e Dog=1.")
            if sha in hashes and hashes[sha][0] != split:
                raise ValueError(f"Vazamento: mesmo conteúdo em {hashes[sha][0]} e {split}.")
            if sha in hashes and hashes[sha][1] != classe and not aceitar_rotulos_duplicados:
                raise ValueError("Rótulos conflitantes para arquivos de conteúdo idêntico.")
            hashes[sha] = (split, classe)
            caminho = (raiz / linha["caminho"]).resolve()
            if not caminho.is_relative_to(raiz) or caminho in caminhos:
                raise ValueError(f"Caminho duplicado ou fora do dataset: {caminho}")
            caminhos.add(caminho)
            if not caminho.is_file():
                raise ValueError(f"Imagem ausente: {caminho}")
            conjuntos[split].append((caminho, ROTULOS[classe]))
            chave = (classe, linha["arquivo_origem"])
            grupo = originais.setdefault(chave, {"split": split, "sha": sha, "contagem": 0,
                                                 "combinacoes": set()})
            if grupo["split"] != split or grupo["sha"] != sha:
                raise ValueError(f"Vazamento ou origem inconsistente: {chave}")
            grupo["contagem"] += 1
            grupo["combinacoes"].add((linha["geometria"], linha["espelho"], int(linha["rotacao"])))
    for chave, grupo in originais.items():
        if grupo["split"] == "treino":
            esperado = {(g, e, r) for g in ("recorte", "esticada")
                        for e in ("normal", "espelhada") for r in (0, 90, 180, 270)}
            if grupo["contagem"] != 16 or grupo["combinacoes"] != esperado:
                raise ValueError(f"Treino sem as 16 variantes completas: {chave}")
        elif grupo["contagem"] != 1 or grupo["combinacoes"] != {("avaliacao", "normal", 0)}:
            raise ValueError(f"Avaliação deve ter uma versão determinística: {chave}")
    if any(not conjuntos[split] for split in splits_obrigatorios):
        raise ValueError("Um split está vazio; confira as frações e o manifesto.")
    return conjuntos


class DatasetManifesto(Dataset):
    def __init__(self, amostras):
        self.amostras = amostras
        self.class_to_idx = ROTULOS.copy()

    def __len__(self):
        return len(self.amostras)

    def __getitem__(self, indice):
        caminho, rotulo = self.amostras[indice]
        with Image.open(caminho) as imagem:
            if imagem.mode != "RGB" or imagem.size != (224, 224):
                raise ValueError(f"Imagem fora do contrato RGB 224x224: {caminho}")
            x = normalizar_imagem(imagem)
        return x, torch.tensor([rotulo], dtype=torch.float32)  # rótulo [1] -> batch [B,1]
