"""Edite CONFIG: escolhas metodológicas obrigatórias não recebem defaults novos."""
from dataclasses import dataclass, field
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]  # Cats and Dogs I e II: dados compartilhados.


@dataclass
class ConfigTreino:
    # Preencha os valores definidos pelo participante antes de treinar.
    otimizador: str | None = None  # Nome exato da classe em torch.optim.
    learning_rate: float | None = None
    batch_size: int | None = None
    epocas: int | None = None
    frac_validacao: float | None = None
    frac_teste: float | None = None  # treino = 1 - validação - teste
    divisao: str = "hash"
    aceitar_rotulos_duplicados: bool = False
    # True: 16 variantes de treino do manifesto. False: cada original uma vez,
    # redimensionado diretamente (esticado) para RGB 224x224 durante a leitura.
    usar_dados_ampliados: bool = True
    usar_pesos_pre_treinados: bool = True
    usar_pesos_pre_treinados_conv1: bool = True
    epocas_conv1_congelada: int = 5
    inicializacao_do_zero: str = "he_normal_fan_out"
    dropout_probabilidade: float = 0.1
    dropout_crescente: bool = False
    dropout_epoca_inicio: int = 20
    dropout_epoca_maximo: int = 40
    # Posições fixas: nomes dos módulos após os quais aplicar dropout.
    dropout_posicoes: tuple[str, ...] | None = None
    argumentos_otimizador: dict = field(default_factory=dict)

    manifesto: str = str(RAIZ / "dataset_ampliado/224x224/manifesto_variantes.csv")
    checkpoint: str = str(RAIZ / "resultados_residuals_networks/res_theo_especial/ultima_epoca.pt")
    semente: int = 42  # Mantém a semente já utilizada no projeto.
    congelar_backbone: bool = False
    epocas_backbone_congelado: int | None = None  # Quando definido, tem prioridade sobre passos.
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
            raise ValueError("Defina em res_theo_especial/config_res_theo_especial.py: " + ", ".join(faltantes))
        if self.learning_rate <= 0 or self.batch_size < 1 or self.epocas < 1:
            raise ValueError("learning_rate, batch_size e epocas devem ser positivos.")
        if self.epocas_backbone_congelado is not None and (
                isinstance(self.epocas_backbone_congelado, bool)
                or not isinstance(self.epocas_backbone_congelado, int)
                or self.epocas_backbone_congelado < 0):
            raise ValueError("epocas_backbone_congelado deve ser um inteiro não negativo ou None.")
        if self.num_workers < 0:
            raise ValueError("num_workers não pode ser negativo.")
        if not isinstance(self.usar_dados_ampliados, bool):
            raise ValueError("usar_dados_ampliados deve ser True ou False.")
        if not isinstance(self.usar_pesos_pre_treinados, bool):
            raise ValueError("usar_pesos_pre_treinados deve ser True ou False.")
        if self.inicializacao_do_zero != "he_normal_fan_out":
            raise ValueError("inicializacao_do_zero deve ser 'he_normal_fan_out'.")
        if self.usar_pesos_pre_treinados:
            raise ValueError("res_theo_especial exige usar_pesos_pre_treinados=False.")
        if not isinstance(self.usar_pesos_pre_treinados_conv1, bool):
            raise ValueError("usar_pesos_pre_treinados_conv1 deve ser True ou False.")
        if (isinstance(self.epocas_conv1_congelada, bool)
                or not isinstance(self.epocas_conv1_congelada, int)
                or self.epocas_conv1_congelada < 0):
            raise ValueError("epocas_conv1_congelada deve ser um inteiro não negativo.")
        if (isinstance(self.passos_backbone_congelado, bool)
                or not isinstance(self.passos_backbone_congelado, int)
                or self.passos_backbone_congelado < 0):
            raise ValueError("passos_backbone_congelado deve ser um inteiro não negativo.")
        if not (0 < self.frac_validacao < 1 and 0 <= self.frac_teste < 1
                and self.frac_validacao + self.frac_teste < 1):
            raise ValueError("Treino e validação devem ser positivos; teste pode ser zero; as frações devem somar 1.")
        validar_dropout(self)
        if "lr" in self.argumentos_otimizador:
            raise ValueError("Use learning_rate para configurar lr.")


def validar_dropout(config):
    if not 0 <= config.dropout_probabilidade < 1:
        raise ValueError("dropout_probabilidade deve estar em [0, 1).")
    if config.dropout_probabilidade == 0:
        return
    if not isinstance(config.dropout_crescente, bool):
        raise ValueError("dropout_crescente deve ser True ou False.")
    if (isinstance(config.dropout_epoca_inicio, bool)
            or not isinstance(config.dropout_epoca_inicio, int)
            or isinstance(config.dropout_epoca_maximo, bool)
            or not isinstance(config.dropout_epoca_maximo, int)
            or config.dropout_epoca_inicio < 1
            or config.dropout_epoca_maximo <= config.dropout_epoca_inicio):
        raise ValueError("Épocas do dropout devem satisfazer 1 <= início < máximo.")
    if config.dropout_posicoes is None:
        raise ValueError("Variante ainda não liberada para treino: falta definir dropout_posicoes.")
    nomes = config.dropout_posicoes
    if not nomes or any(not isinstance(n, str) or not n for n in nomes) or len(set(nomes)) != len(nomes):
        raise ValueError("Informe posições fixas de dropout distintas.")


CONFIG = ConfigTreino(
    otimizador="Adam", learning_rate=0.001, batch_size=16, epocas=60,
    frac_validacao=0.2, frac_teste=0.0, epocas_backbone_congelado=0,
    divisao="aleatoria_balanceada", aceitar_rotulos_duplicados=True,
    usar_dados_ampliados=False,
    usar_pesos_pre_treinados=False, inicializacao_do_zero="he_normal_fan_out",
    usar_pesos_pre_treinados_conv1=True, epocas_conv1_congelada=5,
    dropout_probabilidade=0.1,
    dropout_crescente=True, dropout_epoca_inicio=20, dropout_epoca_maximo=40,
    dropout_posicoes=(
        'bn1',
        'relu',
        'maxpool',
        'layer1.0.conv1',
        'layer1.0.bn1',
        'layer1.0.relu',
        'layer1.0.conv2',
        'layer1.0.bn2',
        'layer1.0.downsample.0',
        'layer1.0.downsample.1',
        'layer1.1.conv1',
        'layer1.1.bn1',
        'layer1.1.relu',
        'layer1.1.conv2',
        'layer1.1.bn2',
        'layer2.0.conv1',
        'layer2.0.bn1',
        'layer2.0.relu',
        'layer2.0.conv2',
        'layer2.0.bn2',
        'layer2.0.downsample.0',
        'layer2.0.downsample.1',
        'layer2.1.conv1',
        'layer2.1.bn1',
        'layer2.1.relu',
        'layer2.1.conv2',
        'layer2.1.bn2',
        'layer3.0.conv1',
        'layer3.0.bn1',
        'layer3.0.relu',
        'layer3.0.conv2',
        'layer3.0.bn2',
        'layer3.0.downsample.0',
        'layer3.0.downsample.1',
        'layer3.1.conv1',
        'layer3.1.bn1',
        'layer3.1.relu',
        'layer3.1.conv2',
        'layer3.1.bn2',
        'layer4.0.conv1',
        'layer4.0.bn1',
        'layer4.0.relu',
        'layer4.0.conv2',
        'layer4.0.bn2',
        'layer4.0.downsample.0',
        'layer4.0.downsample.1',
        'layer4.1.conv1',
        'layer4.1.bn1',
        'layer4.1.relu',
        'layer4.1.conv2',
        'layer4.1.bn2',
        'avgpool',
        'flatten',
    ),
    manifesto=str(RAIZ / "dataset_ampliado/224x224/divisao_balanceada/manifesto_variantes.csv"),
)
