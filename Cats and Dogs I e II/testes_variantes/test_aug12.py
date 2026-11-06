from dataclasses import replace
import torch
from PIL import Image
from torch.utils.data import TensorDataset,DataLoader


def test_aug12_e_rotulos(tmp_path):
    from res_theo34_aug12.ampliacao12 import VARIANTES,DatasetManifesto,blur_media
    from res_theo34_aug12.config_res_theo34 import CONFIG
    CONFIG.validar()
    assert len(VARIANTES)==12
    p=tmp_path/'imagem.png'
    Image.new('RGB',(224,224),(100,180,220)).save(p)
    d=DatasetManifesto([(p,1,v) for v in VARIANTES],config=CONFIG)
    for i in range(12):
        x,y=d[i]
        assert x.shape==(3,224,224) and torch.isfinite(x).all()
        assert y.shape==() and y.dtype==torch.long and y.item()==1
    assert blur_media(torch.ones(3,224,224),10).shape==(3,224,224)


def test_conversao_preserva_probabilidades():
    from res_theo34_aug12.res_theo34 import converter_pesos_binarios
    from res_theo34_aug12.config_res_theo34 import CONFIG
    s=torch.load(CONFIG.checkpoint_origem,map_location='cpu',weights_only=False)
    convertido=converter_pesos_binarios(s['modelo'])
    x=torch.randn(8,256)
    z=torch.nn.functional.linear(x,s['modelo']['fc.weight'],s['modelo']['fc.bias'])
    logits=torch.nn.functional.linear(x,convertido['fc.weight'],convertido['fc.bias'])
    assert torch.allclose(logits.softmax(1)[:,1],z.sigmoid().flatten(),atol=1e-6)


def test_crossentropy_e_metricas():
    from res_theo34_aug12.treinar_res_theo34 import executar_epoca
    x=torch.tensor([[2.,0.],[0.,2.]])
    y=torch.tensor([0,1])
    loader=DataLoader(TensorDataset(x,y),batch_size=2)
    metricas={}
    loss=executar_epoca(torch.nn.Identity(),loader,torch.device('cpu'),metricas=metricas)
    assert abs(loss-torch.nn.functional.cross_entropy(x,y).item())<1e-6
    assert metricas['acuracia']==metricas['f1']==1
