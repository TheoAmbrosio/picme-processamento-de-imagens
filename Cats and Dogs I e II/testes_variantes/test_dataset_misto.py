from PIL import Image
import torch


def test_misto_parametros_e_tipos(tmp_path, monkeypatch):
    from dataset_misto import CONFIG,VARIANTES,DatasetMisto,aplicar_quadrados
    CONFIG.validar()
    assert CONFIG.quadrados_quantidade==10
    assert abs(CONFIG.ruido_variancia-.1)<1e-12
    assert CONFIG.blur_kernel==12
    assert VARIANTES.count('original')==VARIANTES.count('esticada_aleatoria')==1
    assert not any('rotacao' in v or 'espelho' in v for v in VARIANTES)
    p=tmp_path/'foto.png'
    Image.new('RGB',(224,224),(70,120,190)).save(p)
    ds=DatasetMisto([(p,0,v) for v in VARIANTES])
    for i in range(len(ds)):
        x,y=ds[i]
        assert x.shape==(3,224,224) and torch.isfinite(x).all()
        assert y.shape==(1,) and y.item()==0
    chamadas = []
    def sortear_cor(*args, **kwargs):
        chamadas.append(args)
        return torch.zeros(3, 1, 1)
    monkeypatch.setattr('dataset_misto.torch.rand', sortear_cor)
    aplicar_quadrados(torch.ones(3,224,224),CONFIG,coloridos=True)
    assert len(chamadas)==10


def test_avaliacao_sem_corrupcao(tmp_path):
    from dataset_misto import DatasetMisto
    p=tmp_path/'foto.png'
    Image.new('RGB',(224,224),'white').save(p)
    ds=DatasetMisto([(p,1,'avaliacao')])
    assert torch.equal(ds[0][0],ds[0][0])
    assert torch.allclose(ds[0][0][:,0,0],torch.tensor([(1-.485)/.229,(1-.456)/.224,(1-.406)/.225]))
