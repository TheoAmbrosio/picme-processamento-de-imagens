"""Lê somente os splits explícitos; não sorteia nem amplia durante a avaliação."""
import csv
import hashlib
import math
import random
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

from .res_theo_especial import PESOS

ROTULOS = {"Cat": 0, "Dog": 1}
SPLITS = ("treino", "validacao", "teste")
PRESET = PESOS.transforms()
NORMALIZAR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=PRESET.mean, std=PRESET.std),
])
PREPARAR_ORIGINAL = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR,
                      antialias=True),
    NORMALIZAR,
])
RAIZ_PROJETO = Path(__file__).resolve().parents[2]
VARIANTES_ESPECIAIS = ("original", "cinza", "ruido_sal", "quadrados_coloridos", "cutout")


def normalizar_imagem(imagem: Image.Image) -> torch.Tensor:
    # Os JPEGs já têm 224x224; aplicar só escala [0, 1] e normalização ImageNet uma vez.
    return NORMALIZAR(imagem)


def carregar_divisao(manifesto: Path, splits_obrigatorios=SPLITS,
                     aceitar_rotulos_duplicados=False, usar_dados_ampliados=True,
                     raiz_originais=None, esquema_ampliacao=None) -> dict:
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
            if usar_dados_ampliados and esquema_ampliacao is None and not caminho.is_file():
                raise ValueError(f"Imagem ausente: {caminho}")
            conjuntos[split].append((caminho, ROTULOS[classe]))
            chave = (classe, linha["arquivo_origem"])
            grupo = originais.setdefault(chave, {"split": split, "sha": sha, "contagem": 0,
                                                 "combinacoes": set(), "classe": classe,
                                                 "arquivo_origem": linha["arquivo_origem"]})
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
    if not usar_dados_ampliados or esquema_ampliacao == "especial_5_variantes":
        base = (Path(raiz_originais).resolve() if raiz_originais is not None
                else (RAIZ_PROJETO / "dataset/PetImages").resolve())
        conjuntos = {split: [] for split in SPLITS}
        for grupo in originais.values():
            caminho = (base / grupo["classe"] / grupo["arquivo_origem"]).resolve()
            if not caminho.is_relative_to(base) or not caminho.is_file():
                raise ValueError(f"Imagem original ausente ou fora do dataset: {caminho}")
            conjuntos[grupo["split"]].append((caminho, ROTULOS[grupo["classe"]]))
    if esquema_ampliacao == "especial_5_variantes":
        conjuntos["treino"] = [(*amostra, variante) for amostra in conjuntos["treino"]
                               for variante in VARIANTES_ESPECIAIS]
        conjuntos["validacao"] = [(*amostra, "original")
                                  for amostra in conjuntos["validacao"]]
        conjuntos["teste"] = [(*amostra, "original") for amostra in conjuntos["teste"]]
    if any(not conjuntos[split] for split in splits_obrigatorios):
        raise ValueError("Um split está vazio; confira as frações e o manifesto.")
    return conjuntos


class DatasetManifesto(Dataset):
    def __init__(self, amostras, usar_dados_ampliados=True, esquema_ampliacao=None,
                 semente=42, ruido_gaussiano_variancia=0.1,
                 pixels_brancos_fracao=0.02, cutout_retangulos=8,
                 cutout_lado_minimo=20, cutout_lado_maximo=40):
        self.amostras = amostras
        self.usar_dados_ampliados = usar_dados_ampliados
        self.esquema_ampliacao = esquema_ampliacao
        self.semente = semente
        self.ruido_gaussiano_variancia = ruido_gaussiano_variancia
        self.pixels_brancos_fracao = pixels_brancos_fracao
        self.cutout_retangulos = cutout_retangulos
        self.cutout_lado_minimo = cutout_lado_minimo
        self.cutout_lado_maximo = cutout_lado_maximo
        self.class_to_idx = ROTULOS.copy()

    def __len__(self):
        return len(self.amostras)

    def __getitem__(self, indice):
        amostra = self.amostras[indice]
        caminho, rotulo = amostra[:2]
        variante = amostra[2] if len(amostra) == 3 else None
        with Image.open(caminho) as imagem:
            imagem = ImageOps.exif_transpose(imagem)
            if self.esquema_ampliacao == "especial_5_variantes":
                x = self._preparar_especial(imagem.convert("RGB"), caminho, variante)
            elif self.usar_dados_ampliados:
                if imagem.mode != "RGB" or imagem.size != (224, 224):
                    raise ValueError(f"Imagem fora do contrato RGB 224x224: {caminho}")
                x = normalizar_imagem(imagem)
            else:
                x = PREPARAR_ORIGINAL(imagem.convert("RGB"))
        return x, torch.tensor([rotulo], dtype=torch.float32)  # rótulo [1] -> batch [B,1]

    def _rng(self, caminho, finalidade):
        texto = f"{self.semente}|{Path(caminho).resolve()}|{finalidade}"
        valor = int.from_bytes(hashlib.sha256(texto.encode()).digest()[:8], "big")
        return random.Random(valor), valor % (2**63)

    def _preparar_especial(self, imagem, caminho, variante):
        imagem = transforms.functional.resize(
            imagem, [224, 224], interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True)
        base = imagem

        if variante == "cinza":
            base = ImageOps.grayscale(base).convert("RGB")
        tensor = transforms.functional.to_tensor(base)

        if variante == "ruido_sal":
            _, semente_ruido = self._rng(caminho, "ruido_gaussiano")
            gerador = torch.Generator().manual_seed(semente_ruido)
            desvio = math.sqrt(self.ruido_gaussiano_variancia)
            tensor = (tensor + torch.randn(tensor.shape, generator=gerador) * desvio).clamp_(0, 1)
            rng_sal, _ = self._rng(caminho, "pixels_brancos")
            quantidade = round(self.pixels_brancos_fracao * 224 * 224)
            indices = rng_sal.sample(range(224 * 224), quantidade)
            ys = torch.tensor([i // 224 for i in indices])
            xs = torch.tensor([i % 224 for i in indices])
            tensor[:, ys, xs] = 1.0
        elif variante in ("cutout", "quadrados_coloridos"):
            tensor = tensor.clone()
            for _ in range(self.cutout_retangulos):
                lado = int(torch.randint(self.cutout_lado_minimo, self.cutout_lado_maximo + 1, ()).item())
                y = int(torch.randint(0, 225 - lado, ()).item())
                x = int(torch.randint(0, 225 - lado, ()).item())
                cor = torch.rand(3, 1, 1) if variante == "quadrados_coloridos" else 0.0
                tensor[:, y:y + lado, x:x + lado] = cor
        return transforms.functional.normalize(tensor, PRESET.mean, PRESET.std)
