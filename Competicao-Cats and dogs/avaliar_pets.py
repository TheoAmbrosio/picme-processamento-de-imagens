"""Inferência dos quatro checkpoints fechados no ZIP de Pets; sem treinamento."""
from pathlib import Path
import argparse,ast,csv,hashlib,importlib.util,json,zipfile,shutil,textwrap
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from PIL import Image,ImageOps
import torch
from torch import nn
from torchvision import transforms as T
from torchvision.models import resnet34
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RAIZ=Path(__file__).resolve().parent
THEO=RAIZ.parent/'Cats and Dogs I e II'
NOMES=('Theo','Lisandra','Dulce','Bandeira')

def modulo(nome,p):
    spec=importlib.util.spec_from_file_location(nome,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def classe_notebook(p,nome,extras=None):
    nb=json.loads(p.read_text())
    for c in nb['cells']:
        s=''.join(c.get('source',[]))
        if f'class {nome}' not in s:continue
        node=next(n for n in ast.parse(s).body if isinstance(n,ast.ClassDef) and n.name==nome)
        ns={'torch':torch,'nn':nn,**(extras or {})}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns)
        return ns[nome]
    raise ValueError(nome)

def construir(nome):
    norm=T.Normalize([.485,.456,.406],[.229,.224,.225])
    if nome=='Theo':
        peso=THEO/'resultados_residuals_networks/res_theo34_aug8/20261105_163938_589179_bs64_ep30_cong0ep_do10_dadosamp_pre/melhor_modelo.pt'
        estado=torch.load(peso,map_location='cpu',weights_only=True)
        assert estado['rotulos']=={'Cat':0,'Dog':1} and estado['epoca']==29
        mod=modulo('theo_fechado',THEO/'residuals networks/res_theo34_aug8/res_theo34.py')
        model=mod.ResTheo(resnet34(weights=None));sd=estado['modelo']
        geom=modulo('geometria_theo',THEO/'criar_dataset_ampliado.py')
        def transformar(im):return norm(T.ToTensor()(geom.aplicar_avaliacao(im)))
        desc='EXIF→RGB; menor lado256 bilinear PIL; CenterCrop224; ToTensor; Normalize ImageNet. Dropout desativado em eval (hook sem pesos).'
    elif nome=='Lisandra':
        peso=RAIZ/'Lisandra/modelo principal/checkpoints/versao_com_batchnorm/alexnet_cats_dogs_best.pth'
        model=classe_notebook(RAIZ/'Lisandra/projeto/CatsandDogs_classification.ipynb','AlexNetBinary')()
        sd=torch.load(peso,map_location='cpu',weights_only=True)
        transformar=T.Compose([T.Resize((224,224)),T.ToTensor(),norm]);desc='RGB; Resize direto224×224 bilinear; ToTensor; Normalize ImageNet.'
    elif nome=='Dulce':
        peso=RAIZ/'Dulce/projeto/Cats vs Dogs/modelo_cats_and_dogs_224_v2.pt'
        model=classe_notebook(RAIZ/'Dulce/projeto/Cats vs Dogs/cats_and_dogs_224_v2.ipynb','Rede',{'FILTROS':[32,64,64,128,128]})(224,224)
        sd=torch.load(peso,map_location='cpu',weights_only=True)
        def transformar(im):
            w,h=im.size;escala=max(224/h,224/w);ch,cw=min(224/escala,h),min(224/escala,w)
            x,y=max((w-cw)/2,0),max((h-ch)/2,0)
            # Mesmo box central e interpolação do redimensionar() publicado.
            im=im.resize((224,224),Image.Resampling.BILINEAR,box=(x,y,x+cw,y+ch))
            return T.ToTensor()(im)
        desc='RGB; crop central proporcional em box PIL, bilinear224×224; uint8→float/255; sem Normalize. Softmax já aplicado no forward; coluna Dog=1.'
    else:
        repo=RAIZ/'Bandeira/projeto';peso=repo/'data/best_model_4.0_Final.pth'
        model=modulo('bandeira_fechado',repo/'src/model.py').CNN();sd=torch.load(peso,map_location='cpu',weights_only=True)
        stats=json.loads((repo/'data/norm_stats_3.0.json').read_text())
        transformar=T.Compose([T.Resize((384,384)),T.ToTensor(),T.Normalize(stats['mean'],stats['std'])])
        desc='RGB; Resize direto384×384; ToTensor; Normalize norm_stats_3.0.json; fluxo para imagem nova de src/predict.py. Sem preparação offline adicional.'
    model.load_state_dict(sd,strict=True)
    return model.eval().cuda(),transformar,dict(checkpoint=str(peso),sha256=hashlib.sha256(peso.read_bytes()).hexdigest(),preprocessamento=desc,carregamento='strict=True')

def graficos(out,paths,probs):
    plt.rcParams.update({'svg.fonttype':'none','font.size':11})
    fig,ax=plt.subplots(figsize=(10,4.7));im=ax.imshow(probs.T,aspect='auto',vmin=0,vmax=1,cmap='coolwarm')
    ax.set(yticks=range(4),yticklabels=NOMES,xlabel='Imagem (ordem do manifesto)',title='Probabilidade de ser cão — mesmos arquivos, quatro modelos')
    ticks=np.arange(0,len(paths),max(1,len(paths)//15));ax.set_xticks(ticks,ticks+1)
    fig.colorbar(im,ax=ax,label='P(Cão)');fig.tight_layout()
    for ext in ('svg','png'):fig.savefig(out/f'probabilidades.{ext}',dpi=180)
    plt.close(fig)
    labels=probs>=.5;agree=np.array([[np.mean(labels[:,i]==labels[:,j]) for j in range(4)] for i in range(4)])
    fig,ax=plt.subplots(figsize=(7,6));ax.imshow(agree,vmin=0,vmax=1,cmap='Blues');ax.set(xticks=range(4),yticks=range(4),xticklabels=NOMES,yticklabels=NOMES,title='Concordância entre as previsões')
    for i in range(4):
        for j in range(4):ax.text(j,i,f'{agree[i,j]:.1%}',ha='center',va='center',color='white' if agree[i,j]>.65 else 'black')
    fig.tight_layout()
    for ext in ('svg','png'):fig.savefig(out/f'concordancia.{ext}',dpi=160)
    plt.close(fig)
    # Primeiro divergências; depois imagens em que as probabilidades ficam perto do limiar.
    spread=np.ptp(probs,axis=1);ranking=sorted(range(len(paths)),key=lambda i:(not (labels[i].min()!=labels[i].max()),-spread[i]))
    indices=ranking[:min(12,len(paths))]
    fig,axs=plt.subplots(int(np.ceil(len(indices)/4)),4,figsize=(17,4.6*int(np.ceil(len(indices)/4))),squeeze=False)
    for ax in axs.flat:ax.axis('off')
    for ax,i in zip(axs.flat,indices):
        with Image.open(paths[i]) as im:foto=ImageOps.exif_transpose(im).convert('RGB');foto.thumbnail((600,600));ax.imshow(foto)
        linhas=[f'{n}: {"Cão" if labels[i,j] else "Gato"} · P(Cão) {probs[i,j]:.1%}' for j,n in enumerate(NOMES)]
        ax.set_title('\n'.join(textwrap.wrap(paths[i].name,32))+'\n'+'\n'.join(linhas),fontsize=9)
    fig.suptitle('Amostra de classificações — divergências primeiro',fontsize=18);fig.tight_layout(rect=(0,0,1,.97))
    fig.savefig(out/'amostra_classificacoes.png',dpi=150);plt.close(fig)
    return int(np.all(labels==labels[:,:1],axis=1).sum())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('zip',type=Path);args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA necessária; não executar silenciosamente em CPU.')
    torch.set_num_threads(4)
    out=RAIZ/'coisas para apresentacao/Resultados Pets'/datetime.now(ZoneInfo('America/Sao_Paulo')).strftime('%Y%m%d_%H%M%S');out.mkdir(parents=True)
    paths=[];manifest=[]
    with zipfile.ZipFile(args.zip) as z:
        for info in sorted(z.infolist(),key=lambda f:f.filename):
            if info.is_dir() or Path(info.filename).suffix.lower() not in ('.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'):continue
            relative=Path(info.filename)
            if relative.is_absolute() or '..' in relative.parts:raise ValueError(info.filename)
            p=out/'imagens'/relative;p.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as src,p.open('wb') as dst:shutil.copyfileobj(src,dst)
            with Image.open(p) as im:im.verify()
            paths.append(p);manifest.append(dict(filename=p.name,arquivo_zip=info.filename,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    if len(set(p.name for p in paths))!=len(paths):raise ValueError('Nomes duplicados no ZIP; usar caminhos relativos para distinguir.')
    if not paths:raise ValueError('ZIP sem imagens')
    (out/'manifesto.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    probs=np.zeros((len(paths),4));meta={}
    for j,nome in enumerate(NOMES):
        model,transformar,info=construir(nome);valores=[]
        with torch.inference_mode():
            for start in range(0,len(paths),8):
                xs=[]
                for p in paths[start:start+8]:
                    with Image.open(p) as im:
                        # Cada autor mantém a sua conversão; EXIF somente no pipeline Theo.
                        rgb=ImageOps.exif_transpose(im).convert('RGB') if nome=='Theo' else im.convert('RGB')
                        xs.append(transformar(rgb))
                y=model(torch.stack(xs).cuda());pr=y[:,1] if nome=='Dulce' else y.reshape(-1).sigmoid()
                valores.extend(pr.cpu().tolist())
        probs[:,j]=valores;assert np.isfinite(probs[:,j]).all() and ((probs[:,j]>=0)&(probs[:,j]<=1)).all()
        with (out/f'{nome}.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['filename','probability','pred_label']);w.writerows((p.name,format(v,'.10g'),int(v>=.5)) for p,v in zip(paths,valores))
        meta[nome]=info;print(f'{nome}: {len(valores)} predições exportadas',flush=True);del model;torch.cuda.empty_cache()
    consenso=graficos(out,paths,probs)
    summary=dict(imagens=len(paths),concordancia_unanime=consenso,divergencias=len(paths)-consenso,device=torch.cuda.get_device_name(0),torch_version=str(torch.__version__),zip=str(args.zip),zip_sha256=hashlib.sha256(args.zip.read_bytes()).hexdigest(),limiar=.5,classe_positiva='Cão=1; Gato=0',modelos=meta)
    (out/'execucao.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (out/'README.md').write_text(f'''# Predições dos quatro modelos — Pets

{len(paths)} imagens, mesmo manifesto e mesma ordem nos quatro CSVs.
Colunas: filename, probability=P(Cão), pred_label (1 se probability>=0,5;0 caso contrário).
Inferência CUDA em float32, eval(), sem augmentation ou treinamento; pesos carregados estritamente.

- Theo.csv: melhor Aug8/29.
- Lisandra.csv: checkpoint recebido com BatchNorm/5 (época declarada pela autora).
- Dulce.csv: CNN224 v2/112.
- Bandeira.csv: V4 Final/10.

Os pipelines próprios estão registrados em execucao.json. Bandeira segue o código de
inferência para imagem nova (resize direto), sem reproduzir a preparação offline do treino.
Nenhuma métrica de acerto foi calculada: pastas não foram usadas como rótulos nesta execução.
Concordância não significa acerto. Probabilidades são saídas dos modelos, sem calibração adicional.
{consenso} imagens com unanimidade;{len(paths)-consenso} com divergência.

Veja probabilidades.svg, concordancia.svg e amostra_classificacoes.png.
manifesto.json relaciona número/ordem, nome e arquivo original. Checkpoints e ZIP identificados por SHA256.
''')
    with zipfile.ZipFile(out/'predicoes_csv.zip','w',zipfile.ZIP_DEFLATED) as z:
        for n in NOMES:z.write(out/f'{n}.csv',f'{n}.csv')
    print(f'SAIDA={out}\nUNANIMIDADE={consenso}/{len(paths)}',flush=True)

if __name__=='__main__':main()
