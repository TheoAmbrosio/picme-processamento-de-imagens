import torch
import pytest
from dataclasses import replace


def test_grupos_scheduler_e_pesos():
    from res_theo34_finetuning.config_res_theo34 import CONFIG
    from res_theo34_finetuning.res_theo34 import criar_res_theo34
    from res_theo34_finetuning.treinar_res_theo34 import criar_otimizador
    CONFIG.validar()
    modelo,_=criar_res_theo34(False,replace(CONFIG,usar_pesos_pre_treinados=False))
    estado=torch.load(CONFIG.checkpoint_origem,map_location='cpu',weights_only=False)
    assert estado['epoca']==33
    modelo.load_state_dict(estado['modelo'],strict=True)
    assert all(torch.equal(v,estado['modelo'][k]) for k,v in modelo.state_dict().items())
    assert modelo.dropout_comum.posicoes==('flatten',)
    assert all(p.requires_grad for p in modelo.parameters())
    opt=criar_otimizador(modelo,CONFIG)
    assert [g['lr'] for g in opt.param_groups]==[1e-5,5e-5,1e-4]
    assert all(g['weight_decay']==.01 for g in opt.param_groups)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=30,eta_min=1e-7)
    for _ in range(30):
        opt.step();scheduler.step()
    assert [g['lr'] for g in opt.param_groups]==pytest.approx([1e-7]*3)
