"""Carrega um checkpoint AlexTheo e informa P(cachorro)."""
import argparse
from pathlib import Path
import torch
from torchvision.models import alexnet

from criar_dataset_ampliado import abrir_rgb, aplicar_avaliacao
from .alex_theo import AlexTheo
from .config_alex_theo import ConfigTreino
from .dados_alex_theo import ROTULOS, normalizar_imagem


def carregar_checkpoint(caminho, dispositivo):
    estado = torch.load(caminho, map_location="cpu", weights_only=True)
    if (estado["nome_modelo"] != "alex_theo" or estado["rotulos"] != ROTULOS
            or estado["dataset"]["lado"] != 224):
        raise ValueError("Checkpoint incompatível com AlexTheo RGB 224x224.")
    config = ConfigTreino(**estado["config"])
    config.validar()
    modelo = AlexTheo(alexnet(weights=None), dropout_probabilidade=config.dropout_probabilidade)
    modelo.dropout_comum.carregar_estado(estado["dropout"])
    modelo.load_state_dict(estado["modelo"], strict=True)
    return modelo.to(dispositivo).eval(), estado


def probabilidade_cachorro(modelo, caminho):
    imagem = abrir_rgb(Path(caminho).read_bytes())
    x = normalizar_imagem(aplicar_avaliacao(imagem)).unsqueeze(0)
    return modelo.predict_proba(x.to(next(modelo.parameters()).device))[0, 0].item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("imagem", type=Path)
    args = parser.parse_args()
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo, _ = carregar_checkpoint(args.checkpoint, dispositivo)
    p = probabilidade_cachorro(modelo, args.imagem)
    print(f"Dispositivo: {dispositivo}; P(cachorro)={p:.8f}; classe={int(p >= 0.5)}")


if __name__ == "__main__":
    main()
