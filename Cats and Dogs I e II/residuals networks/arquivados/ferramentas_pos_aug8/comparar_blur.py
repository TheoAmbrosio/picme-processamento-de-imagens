"""Prévia de blur por média com janelas exatas 5x5 e 10x10."""
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw,ImageFont
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor,to_pil_image

RAIZ=Path(__file__).resolve().parents[1]


def blur_media(imagem,kernel):
    x=to_tensor(imagem).unsqueeze(0)
    antes=(kernel-1)//2
    depois=kernel-1-antes
    x=F.pad(x,(antes,depois,antes,depois),mode='reflect')
    return to_pil_image(F.avg_pool2d(x,kernel,stride=1).squeeze(0))


def gerar():
    torch.set_num_threads(2)
    fonte=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    painel=Image.new('RGB',(1008,566),'white')
    desenho=ImageDraw.Draw(painel)
    for linha,classe in enumerate(('Cat','Dog')):
        with Image.open(RAIZ/'dataset/PetImages'/classe/'0.jpg') as im:
            original=ImageOps.exif_transpose(im).convert('RGB').resize((224,224),Image.Resampling.BILINEAR)
        imagens=(original,blur_media(original,5),blur_media(original,10),blur_media(original,20))
        for col,(titulo,im) in enumerate(zip(('Original','Blur médio 5×5','Blur médio 10×10','Blur médio 20×20'),imagens)):
            x,y=16+col*248,14+linha*278
            desenho.text((x,y),titulo,font=fonte,fill='black')
            painel.paste(im,(x,y+32))
    destino=RAIZ/'resultados_residuals_networks/res_theo34_aug8/comparacao_blur_5_10_20.png'
    painel.save(destino)
    print(destino)


if __name__=='__main__':gerar()
