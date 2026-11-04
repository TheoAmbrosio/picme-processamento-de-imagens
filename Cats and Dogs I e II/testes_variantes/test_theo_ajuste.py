import hashlib
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset


def test_f1_e_grafico_por_epoca(tmp_path):
    from theo_ajuste.treinar_res_theo_especial import executar_epoca, salvar_historico_e_graficos
    modelo = torch.nn.Identity()
    loader = DataLoader(TensorDataset(torch.tensor([[1.], [1.], [-1.], [-1.]]),
                                     torch.tensor([[1.], [0.], [1.], [0.]])), batch_size=2)
    metricas = {}
    executar_epoca(modelo, loader, torch.device('cpu'), metricas=metricas)
    assert metricas['f1'] == pytest.approx(0.5)
    assert metricas['acuracia'] == pytest.approx(0.5)
    salvar_historico_e_graficos([dict(epoca=1, bce_treino=.5, bce_validacao=.6,
        acuracia_treino=.5, acuracia_validacao=.5, f1_treino=.5, f1_validacao=.5)],
        tmp_path, graficos=True)
    assert (tmp_path / 'graficos_treinamento.png').exists()


def test_checkpoint_e_agendas():
    from theo_ajuste.config_res_theo_especial import CONFIG
    from theo_ajuste.treinar_res_theo_especial import probabilidade_dropout_epoca, taxa_aprendizado_epoca
    CONFIG.validar()
    estado = torch.load(CONFIG.checkpoint_origem, map_location='cpu', weights_only=False)
    assert estado['epoca'] == 50
    assert estado['manifesto_sha256'] == hashlib.sha256(Path(CONFIG.manifesto).read_bytes()).hexdigest()
    assert CONFIG.epocas == 40 and CONFIG.epocas_iniciais_sem_ampliacao == 0
    assert [probabilidade_dropout_epoca(CONFIG, e) for e in (1,15,25,40)] == pytest.approx([0,.15,.05,.05])
    assert taxa_aprendizado_epoca(CONFIG, 1) == pytest.approx(1e-6)
    assert taxa_aprendizado_epoca(CONFIG, 25) == pytest.approx(.001)


def test_quadrados_variaveis_e_orientacao(tmp_path):
    from theo_ajuste.dados_res_theo_especial import DatasetManifesto, VARIANTES_ESPECIAIS
    caminho = tmp_path / 'imagem.png'
    Image.new('RGB', (224,224), (80,120,160)).save(caminho)
    ds = DatasetManifesto([(caminho,0,v) for v in VARIANTES_ESPECIAIS],
                         esquema_ampliacao='especial_5_variantes')
    assert 'rotacionada' not in VARIANTES_ESPECIAIS
    assert not torch.equal(ds[3][0],ds[3][0])
    assert not torch.equal(ds[4][0],ds[4][0])


def test_novas_arquiteturas():
    from theo_curto.theo_curto import criar_modelo as curto
    from theo_48_88.theo_48_88 import criar_modelo as estreito
    for fabrica in (curto, estreito):
        modelo = fabrica().eval()
        with torch.no_grad():
            assert modelo(torch.zeros(2,3,224,224)).shape == (2,1)
        if fabrica is curto:
            assert not hasattr(modelo,'layer3')
            assert modelo.fc.in_features == 64
        else:
            assert modelo.layer2[0].conv1.out_channels == 48
            assert modelo.layer3[0].conv1.out_channels == 88
            assert modelo.fc.in_features == 88
