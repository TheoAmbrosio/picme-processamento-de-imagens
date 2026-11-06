"""Edite CONFIG: escolhas metodológicas obrigatórias não recebem defaults novos."""
from dataclasses import dataclass, field, replace
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]  # Cats and Dogs I e II: dados compartilhados.


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
    inicializacao_do_zero: str = "he_normal_fan_out"
    dropout_probabilidade: float = 0.1
    dropout_crescente: bool = False
    dropout_epoca_inicio: int = 20
    dropout_epoca_maximo: int = 40
    # Posições fixas: nomes dos módulos após os quais aplicar dropout.
    dropout_posicoes: tuple[str, ...] | None = None
    argumentos_otimizador: dict = field(default_factory=dict)
    checkpoint_origem: str = str(RAIZ / 'resultados_residuals_networks/res_theo34_aug8/20261105_163938_589179_bs64_ep30_cong0ep_do10_dadosamp_pre/melhor_modelo.pt')
    epoca_origem: int = 29
    epoca_acumulada_origem: int = 62
    perda: str = 'CrossEntropyLoss'
    classes_saida: int = 2
    conversao_fc: str = 'logits_iniciais_0_e_logit_original'
    blur_kernel: int = 10
    blur_tipo: str = 'media_reflect'
    faixa_min: int = 0
    faixa_max: int = 50
    modulos_grupos: tuple = (("conv1", "bn1", "layer1", "layer2"), ("layer3",), ("fc",))
    learning_rates_grupos: tuple = (1e-5, 5e-5, 1e-4)
    scheduler_nome: str = 'CosineAnnealingLR'
    scheduler_t_max: int = 30
    scheduler_eta_min: float = 1e-7
    ampliacao_esquema: str = 'aug12_faixa_e_blur10'
    ruido_variancia: float = 0.08
    pixels_brancos_fracao: float = 0.02
    quadrados_quantidade: int = 8
    quadrado_min: int = 20
    quadrado_max: int = 40
    regiao_centros: tuple = (150,150)
    fator_esticamento: float = 2.0

    manifesto: str = str(RAIZ / "dataset_ampliado/224x224/manifesto_variantes.csv")
    checkpoint: str = str(RAIZ / "resultados_residuals_networks/res_theo34/ultima_epoca.pt")
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
            raise ValueError("Defina em res_theo34/config_res_theo34.py: " + ", ".join(faltantes))
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


CONFIG_ATUAL_ANTERIOR = ConfigTreino(
    otimizador="Adam", learning_rate=0.001, batch_size=16, epocas=50,
    frac_validacao=0.2, frac_teste=0.0, epocas_backbone_congelado=0,
    divisao="aleatoria_balanceada", aceitar_rotulos_duplicados=True,
    usar_dados_ampliados=False,
    usar_pesos_pre_treinados=False, inicializacao_do_zero="he_normal_fan_out",
    dropout_probabilidade=0.1,
    dropout_crescente=False, dropout_epoca_inicio=20, dropout_epoca_maximo=40,
    dropout_posicoes=(
        'bn1',
        'relu',
        'maxpool',
        'layer1.0.conv1',
        'layer1.0.bn1',
        'layer1.0.relu',
        'layer1.0.conv2',
        'layer1.0.bn2',
        'layer1.1.conv1',
        'layer1.1.bn1',
        'layer1.1.relu',
        'layer1.1.conv2',
        'layer1.1.bn2',
        'layer1.2.conv1',
        'layer1.2.bn1',
        'layer1.2.relu',
        'layer1.2.conv2',
        'layer1.2.bn2',
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
        'layer2.2.conv1',
        'layer2.2.bn1',
        'layer2.2.relu',
        'layer2.2.conv2',
        'layer2.2.bn2',
        'layer2.3.conv1',
        'layer2.3.bn1',
        'layer2.3.relu',
        'layer2.3.conv2',
        'layer2.3.bn2',
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
        'layer3.2.conv1',
        'layer3.2.bn1',
        'layer3.2.relu',
        'layer3.2.conv2',
        'layer3.2.bn2',
        'layer3.3.conv1',
        'layer3.3.bn1',
        'layer3.3.relu',
        'layer3.3.conv2',
        'layer3.3.bn2',
        'layer3.4.conv1',
        'layer3.4.bn1',
        'layer3.4.relu',
        'layer3.4.conv2',
        'layer3.4.bn2',
        'layer3.5.conv1',
        'layer3.5.bn1',
        'layer3.5.relu',
        'layer3.5.conv2',
        'layer3.5.bn2',
        'avgpool',
        'flatten',
    ),
    manifesto=str(RAIZ / "dataset_ampliado/224x224/divisao_balanceada/manifesto_variantes.csv"),
)

# Valores recuperados do checkpoint histórico e confirmados pelo log original.
CONFIG = ConfigTreino(
    otimizador="Adam", learning_rate=0.001, batch_size=64, epocas=50,
    frac_validacao=0.2, frac_teste=0.0, divisao="aleatoria_balanceada",
    aceitar_rotulos_duplicados=True, usar_dados_ampliados=True,
    usar_pesos_pre_treinados=True, epocas_backbone_congelado=5,
    congelar_backbone=False, passos_backbone_congelado=50, semente=42,
    num_workers=4, precisao_mista=True, cudnn_benchmark=False,
    channels_last=False, compilar_modelo=False,
    dropout_probabilidade=0.0, dropout_crescente=False, dropout_posicoes=(),
    manifesto=str(RAIZ / "dataset_ampliado/224x224/divisao_balanceada/manifesto_variantes.csv"),
    checkpoint=str(RAIZ / "resultados_residuals_networks/res_theo34_repeticao/ultima_epoca.pt"),
)

CONFIG = replace(CONFIG, epocas=30, epocas_backbone_congelado=0,
                 passos_backbone_congelado=0, otimizador='AdamW',
                 learning_rate=1e-4, argumentos_otimizador={'weight_decay': 0.01},
                 dropout_probabilidade=0.10, dropout_crescente=False,
                 dropout_posicoes=('flatten',),
                 checkpoint=str(RAIZ / 'resultados_residuals_networks/res_theo34_aug12/ultima_epoca.pt'))
