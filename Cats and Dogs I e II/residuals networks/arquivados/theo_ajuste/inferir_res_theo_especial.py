"""Carrega um checkpoint do ResTheo especial e informa P(cachorro)."""
import argparse
from pathlib import Path

import torch

from criar_dataset_ampliado import abrir_rgb, aplicar_avaliacao
from .dados_res_theo_especial import ROTULOS, normalizar_imagem
from .res_theo_especial import ResTheo


def carregar_checkpoint(caminho: Path, dispositivo: torch.device):
    estado = torch.load(caminho, map_location="cpu", weights_only=True)
    if (estado["nome_modelo"] != "theo_ajuste" or estado["rotulos"] != ROTULOS
            or estado["dataset"]["lado"] != 224):
        raise ValueError("Checkpoint incompatível com res_theo_especial RGB 224x224.")
    modelo = ResTheo()
    from .config_res_theo_especial import ConfigTreino
    from .dropout_comum import instalar_dropout
    config = ConfigTreino(**estado["config"])
    config.validar()
    instalar_dropout(modelo, config)
    modelo.dropout_comum.carregar_estado(estado["dropout"])
    modelo.load_state_dict(estado["modelo"], strict=True)
    return modelo.to(dispositivo).eval(), estado


def probabilidade_cachorro(modelo: ResTheo, caminho: Path) -> float:
    imagem = abrir_rgb(caminho.read_bytes())
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
