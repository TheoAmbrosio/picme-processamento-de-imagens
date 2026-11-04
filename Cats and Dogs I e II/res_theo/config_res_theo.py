"""Edite CONFIG: escolhas metodológicas obrigatórias não recebem defaults novos."""
from dataclasses import dataclass, field
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]  # Cats and Dogs I e II: dados compartilhados.


@dataclass
class ConfigTreino:
    # Preencha os valores definidos pelo participante antes de treinar.
    otimizador: str | None = None  # Nome exato da classe em torch.optim.
    learning_rate: float | None = None
    batch_size: int | None = None
    epocas: int | None = None
    frac_validacao: float | None = None
    frac_teste: float | None = None  # treino = 1 - validação - teste
    argumentos_otimizador: dict = field(default_factory=dict)

    manifesto: str = str(RAIZ / "dataset_ampliado/224x224/manifesto_variantes.csv")
    checkpoint: str = str(RAIZ / "res_theo/resultados/ultima_epoca.pt")
    semente: int = 42  # Mantém a semente já utilizada no projeto.
    congelar_backbone: bool = False
    passos_backbone_congelado: int = 50  # Só fc aprende nos primeiros N optimizer.step().
    num_workers: int = min(4, max(0, (os.cpu_count() or 1) - 1))  # Ajustável; 0 desativa workers.
    precisao_mista: bool = True  # Habilitada somente em CUDA compatível.
    cudnn_benchmark: bool = False
    channels_last: bool = False  # Sem benefício medido aqui; opcional.
    compilar_modelo: bool = False  # Sem benefício medido aqui; opcional.

    def validar(self) -> None:
        obrigatorios = ("otimizador", "learning_rate", "batch_size", "epocas",
                        "frac_validacao", "frac_teste")
        faltantes = [nome for nome in obrigatorios if getattr(self, nome) is None]
        if faltantes:
            raise ValueError("Defina em res_theo/config_res_theo.py: " + ", ".join(faltantes))
        if self.learning_rate <= 0 or self.batch_size < 1 or self.epocas < 1:
            raise ValueError("learning_rate, batch_size e epocas devem ser positivos.")
        if self.num_workers < 0:
            raise ValueError("num_workers não pode ser negativo.")
        if (isinstance(self.passos_backbone_congelado, bool)
                or not isinstance(self.passos_backbone_congelado, int)
                or self.passos_backbone_congelado < 0):
            raise ValueError("passos_backbone_congelado deve ser um inteiro não negativo.")
        if not (0 < self.frac_validacao < 1 and 0 < self.frac_teste < 1
                and self.frac_validacao + self.frac_teste < 1):
            raise ValueError("As três frações devem ser positivas e somar 1.")
        if "lr" in self.argumentos_otimizador:
            raise ValueError("Use learning_rate para configurar lr.")


CONFIG = ConfigTreino()
