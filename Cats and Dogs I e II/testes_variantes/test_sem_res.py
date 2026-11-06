from dataclasses import asdict
import importlib
import pytest
import torch
from torch import nn


@pytest.mark.parametrize('nome,canais', [
    ('theo_sem_res_28_42_64',(28,42,64)),
    ('theo_sem_res_28_42_64_88',(28,42,64,88))])
def test_sem_res_arquitetura_treino_e_checkpoint(nome,canais,tmp_path):
    from theo_28_42_64.config_theo_28_42_64 import CONFIG as REFERENCIA
    from modelo_sem_res import TheoSemRes
    cfg = importlib.import_module(f'{nome}.config_{nome}').CONFIG
    cfg.validar()
    a,b = asdict(cfg),asdict(REFERENCIA)
    a.pop('dropout_posicoes');b.pop('dropout_posicoes')
    assert a == b
    modulo = importlib.import_module(f'{nome}.{nome}')
    modelo,resumo = modulo.criar_para_treino(config=cfg)
    assert resumo['residuos'] is False
    assert modelo.conv1.out_channels == 28
    convs = [m for m in modelo.modules() if isinstance(m,nn.Conv2d)]
    assert len(convs) == len(canais)+1
    for i,c in enumerate(canais,1):
        stage = getattr(modelo,f'layer{i}')
        assert len(stage) == 3
        assert stage[0].out_channels == c
        assert stage[0].stride == ((1,1) if i == 1 else (2,2))
    assert modelo.fc.in_features == canais[-1]
    x = torch.randn(2,3,224,224)
    y = modelo(x)
    assert y.shape == (2,1)
    nn.functional.binary_cross_entropy_with_logits(y,torch.zeros_like(y)).backward()
    assert torch.isfinite(modelo.conv1.weight.grad).all()
    modelo.eval()
    with torch.no_grad():
        assert torch.equal(modelo(x),modelo(x))
    caminho = tmp_path/'pesos.pt'
    torch.save(modelo.state_dict(),caminho)
    novo,_ = modulo.criar_para_treino(config=cfg)
    novo.load_state_dict(torch.load(caminho,weights_only=True),strict=True)
    novo.eval()
    with torch.no_grad():
        assert torch.equal(modelo(x),novo(x))
