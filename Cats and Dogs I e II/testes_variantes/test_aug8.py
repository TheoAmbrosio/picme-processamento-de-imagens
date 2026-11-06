import torch
from PIL import Image


def test_oito_variantes_e_validacao(tmp_path):
    from res_theo34_aug8.ampliacao8 import DatasetManifesto,VARIANTES,NORMALIZAR
    from res_theo34_aug8.config_res_theo34 import CONFIG
    CONFIG.validar()
    assert CONFIG.dropout_probabilidade==.1
    p=tmp_path/'imagem.png'
    pixels=torch.arange(224,dtype=torch.uint8).view(1,224,1).expand(224,224,3).numpy()
    Image.fromarray(pixels).save(p)
    ds=DatasetManifesto([(p,0,v) for v in VARIANTES]+[(p,0,'avaliacao')],config=CONFIG)
    assert len(ds)==8+1
    resultados=[ds[i][0] for i in range(8)]
    assert all(x.shape==(3,224,224) and torch.isfinite(x).all() for x in resultados)
    assert not torch.equal(resultados[0],resultados[3])
    assert not torch.equal(ds[2][0],ds[2][0])
    with Image.open(p) as im:
        assert torch.equal(ds[8][0],NORMALIZAR(im.convert('RGB')))
