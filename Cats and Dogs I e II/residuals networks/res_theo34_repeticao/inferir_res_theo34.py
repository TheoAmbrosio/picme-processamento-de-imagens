"""Carrega um checkpoint local e informa P(cachorro) de uma imagem."""
import argparse
from pathlib import Path

import torch
from torchvision.models import resnet34

from criar_dataset_ampliado import abrir_rgb, aplicar_avaliacao
from .dados_res_theo34 import ROTULOS, normalizar_imagem
from .res_theo34 import ResTheo


def carregar_checkpoint(caminho: Path, dispositivo: torch.device):
    estado = torch.load(caminho, map_location="cpu", weights_only=True)
    if (estado["nome_modelo"] not in ("res_theo34", "res_theo34_dropout", "res_theo") or estado["rotulos"] != ROTULOS
            or estado["dataset"]["lado"] != 224):
        raise ValueError("Checkpoint incompatível com RGB 224x224 e Cat=0/Dog=1.")
    # Reconstitui sem download de ImageNet, pois o checkpoint já contém todos os pesos.
    modelo = ResTheo(resnet34(weights=None))
    from .config_res_theo34 import ConfigTreino
    from .dropout_comum import instalar_dropout
    if estado["nome_modelo"] == "res_theo":
        # Legado sem dropout: não troca a arquitetura pela configuração atual.
        if "config" in estado:
            legado = dict(estado["config"])
            legado.update(dropout_probabilidade=0, dropout_posicoes=())
            estado["config"] = legado
            config = ConfigTreino(**legado)
            config.validar()
    else:
        config = ConfigTreino(**estado["config"])
        config.validar()
        instalar_dropout(modelo, config)
        modelo.dropout_comum.carregar_estado(estado["dropout"])
    modelo.load_state_dict(estado["modelo"], strict=True)
    return modelo.to(dispositivo).eval(), estado


def probabilidade_cachorro(modelo: ResTheo, caminho: Path) -> float:
    imagem = abrir_rgb(caminho.read_bytes())
    # Geometria determinística igual à avaliação e normalização aplicada uma vez.
    x = normalizar_imagem(aplicar_avaliacao(imagem)).unsqueeze(0)
    dispositivo = next(modelo.parameters()).device
    return modelo.predict_proba(x.to(dispositivo))[0, 0].item()


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
