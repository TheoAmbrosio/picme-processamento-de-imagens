"""Download e vetorização SVD das imagens Cats vs. Dogs."""

from pathlib import Path
import shutil
import warnings
import zipfile

import numpy as np
from PIL import Image
from scipy.sparse.linalg import ArpackError, ArpackNoConvergence, svds
import torch
from torch.utils.data import Dataset


TAMANHO = 300
EXTENSOES = {".jpg", ".jpeg", ".png"}
KAGGLE_DATASET = "shaunthesheep/microsoft-catsvsdogs-dataset"


def tem_imagens(pasta: Path) -> bool:
    return pasta.is_dir() and any(
        arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES
        for arquivo in pasta.iterdir()
    )


def garantir_dataset(diretorio: Path) -> None:
    """Baixa o dataset se as duas pastas de classe ainda não existem."""
    if tem_imagens(diretorio / "Cat") and tem_imagens(diretorio / "Dog"):
        print(f"Dataset encontrado: {diretorio}")
        return
    if diretorio.exists():
        raise RuntimeError(f"Dataset incompleto em {diretorio}; revise a pasta antes de baixar.")
    try:
        import kagglehub
    except ImportError as erro:
        raise ImportError("Instale kagglehub: python -m pip install kagglehub") from erro

    destino_download = diretorio.parent / "_download_kaggle"
    destino_download.mkdir(parents=True, exist_ok=True)
    kagglehub.dataset_download(KAGGLE_DATASET, output_dir=str(destino_download))
    candidatos = [
        pasta for pasta in destino_download.rglob("PetImages")
        if tem_imagens(pasta / "Cat") and tem_imagens(pasta / "Dog")
    ]
    if candidatos:
        shutil.move(str(candidatos[0]), str(diretorio))
    else:
        for arquivo_zip in destino_download.rglob("*.zip"):
            with zipfile.ZipFile(arquivo_zip) as pacote:
                membros = [
                    item for item in pacote.infolist()
                    if not item.is_dir()
                    and "PetImages" in Path(item.filename).parts
                    and Path(item.filename).suffix.lower() in EXTENSOES
                ]
                classes = {
                    partes[indice + 1]
                    for item in membros
                    for partes in [Path(item.filename).parts]
                    for indice in [partes.index("PetImages")]
                    if len(partes) > indice + 1
                }
                if not {"Cat", "Dog"}.issubset(classes):
                    continue
                for item in membros:
                    partes = Path(item.filename).parts
                    indice = partes.index("PetImages")
                    if len(partes) != indice + 3 or partes[indice + 1] not in ("Cat", "Dog"):
                        continue
                    destino = diretorio / partes[indice + 1] / partes[indice + 2]
                    destino.parent.mkdir(parents=True, exist_ok=True)
                    with pacote.open(item) as origem, destino.open("wb") as saida:
                        shutil.copyfileobj(origem, saida)
                break
    if not (tem_imagens(diretorio / "Cat") and tem_imagens(diretorio / "Dog")):
        raise RuntimeError(f"Não encontrei PetImages/Cat e Dog em {destino_download}.")
    print(f"Dataset pronto: {diretorio}")


def preparar_imagem(imagem: Image.Image, escala_cinza: bool) -> np.ndarray:
    imagem = imagem.convert("L" if escala_cinza else "RGB")
    largura, altura = imagem.size
    lado = min(largura, altura)
    imagem = imagem.crop(((largura - lado) // 2, (altura - lado) // 2,
                          (largura + lado) // 2, (altura + lado) // 2))
    if imagem.size != (TAMANHO, TAMANHO):
        imagem = imagem.resize((TAMANHO, TAMANHO))
    return np.asarray(imagem, dtype=float)


def fatorar_canal(canal: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        u, sigma, vt = svds(canal, k=k, which="LM")
    except (ArpackError, ArpackNoConvergence) as erro:
        warnings.warn(f"ARPACK falhou; usando SVD densa: {erro}", RuntimeWarning)
        u, sigma, vt = np.linalg.svd(canal, full_matrices=False)
        u, sigma, vt = u[:, :k], sigma[:k], vt[:k, :]
    ordem = np.argsort(sigma)[::-1]
    return u[:, ordem], sigma[ordem, None] * vt[ordem, :]


def vetor_svd(imagem: Image.Image, k: int, escala_cinza: bool) -> np.ndarray:
    matriz = preparar_imagem(imagem, escala_cinza)
    canais = [matriz] if escala_cinza else [matriz[:, :, indice] for indice in range(3)]
    fatores = [fator for canal in canais for fator in fatorar_canal(canal, k)]
    return np.concatenate([fator.ravel() for fator in fatores]).astype(np.float32)


class ConjuntoImagensSVD(Dataset):
    def __init__(self, diretorio: Path, k: int, escala_cinza: bool,
                 usar_todos_dados: bool, n: int, semente: int):
        if not 1 <= k < TAMANHO:
            raise ValueError("k deve estar entre 1 e 299.")
        if not isinstance(usar_todos_dados, bool):
            raise TypeError("usar_todos_dados deve ser bool.")
        if not usar_todos_dados and (isinstance(n, bool) or not isinstance(n, int) or n < 2 or n % 2):
            raise ValueError("N deve ser inteiro par e maior ou igual a 2.")
        self.k = k
        self.escala_cinza = escala_cinza
        self.amostras = []
        for classe, rotulo in (("Cat", 0.0), ("Dog", 1.0)):
            pasta = diretorio / classe
            if not pasta.is_dir():
                raise FileNotFoundError(pasta)
            validas = []
            for caminho in sorted(pasta.iterdir()):
                if not caminho.is_file() or caminho.suffix.lower() not in EXTENSOES:
                    continue
                try:
                    with Image.open(caminho) as imagem:
                        imagem.verify()
                except (OSError, ValueError):
                    print(f"Imagem ilegível ignorada: {caminho}")
                else:
                    validas.append((classe, caminho, rotulo))
            if not usar_todos_dados:
                quantidade = n // 2
                if len(validas) < quantidade:
                    raise ValueError(f"{classe}: há {len(validas)} imagens, mas N exige {quantidade}.")
                gerador = np.random.default_rng(semente + int(rotulo))
                indices = gerador.choice(len(validas), size=quantidade, replace=False)
                validas = [validas[int(indice)] for indice in indices]
            self.amostras.extend(validas)
        if not self.amostras:
            raise ValueError("Nenhuma imagem legível encontrada.")
        modo = "cinza" if escala_cinza else "rgb"
        self.cache = diretorio.parent.parent / "datasets_processados" / "modelo" / "300x300" / f"k_{k}" / modo
        self.cache.mkdir(parents=True, exist_ok=True)
        self.entrada_dim = 2 * TAMANHO * k * (1 if escala_cinza else 3)

    def __len__(self) -> int:
        return len(self.amostras)

    def __getitem__(self, indice: int):
        classe, caminho, rotulo = self.amostras[indice]
        pasta = self.cache / classe
        pasta.mkdir(exist_ok=True)
        arquivo_cache = pasta / f"{caminho.name}.npy"
        vetor = None
        if arquivo_cache.exists() and arquivo_cache.stat().st_mtime >= caminho.stat().st_mtime:
            try:
                vetor = np.load(arquivo_cache, allow_pickle=False)
            except (OSError, ValueError):
                pass
        if vetor is None or vetor.shape != (self.entrada_dim,):
            with Image.open(caminho) as imagem:
                vetor = vetor_svd(imagem, self.k, self.escala_cinza)
            np.save(arquivo_cache, vetor)
        return torch.from_numpy(vetor.astype(np.float32, copy=False)), torch.tensor([rotulo], dtype=torch.float32)
