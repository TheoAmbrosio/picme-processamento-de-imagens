import torch
from PIL import Image


def test_barras_e_variantes(tmp_path):
    from res_theo34_barras.config_res_theo34 import CONFIG
    from res_theo34_barras.ampliacao_barras import DatasetManifesto,VARIANTES,ELEGIVEIS
    CONFIG.validar()
    assert len(VARIANTES)==6
    assert ELEGIVEIS==('ruido','quadrados','ruido_quadrados')
    p=tmp_path/'original.png'
    Image.new('RGB',(224,224),'white').save(p)
    ds=DatasetManifesto([(p,0,v) for v in VARIANTES],config=CONFIG)
    xs=[ds[i][0] for i in range(6)]
    assert all(x.shape==(3,224,224) and torch.isfinite(x).all() for x in xs)
    assert not torch.equal(ds[0][0],ds[0][0])
    negro=torch.tensor([-.485/.229,-.456/.224,-.406/.225])[:,None,None]
    assert ((xs[0]-negro).abs()<1e-5).all(0).sum()>224*19


def test_origem_melhor_aug8():
    from res_theo34_barras.config_res_theo34 import CONFIG
    s=torch.load(CONFIG.checkpoint_origem,map_location='cpu',weights_only=False)
    assert s['epoca']==CONFIG.epoca_origem==29
    assert s['historico'][-1]['acuracia_validacao']==.9832
