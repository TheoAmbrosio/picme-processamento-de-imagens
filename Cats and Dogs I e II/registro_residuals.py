"""Identificação de execuções residuais, sem escolher parâmetros de treinamento."""
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import json
import subprocess

RAIZ = Path(__file__).resolve().parent
RESULTADOS = RAIZ / 'resultados_residuals_networks'


def novo_diretorio(config, modelo):
    carimbo = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    congelamento = (f'cong{config.epocas_backbone_congelado}ep'
                    if config.epocas_backbone_congelado is not None
                    else f'cong{config.passos_backbone_congelado}passos')
    dropout = int(getattr(config, 'dropout_probabilidade', 0) * 100)
    pasta = RESULTADOS / modelo / f'{carimbo}_bs{config.batch_size}_ep{config.epocas}_{congelamento}_do{dropout}'
    pasta.mkdir(parents=True, exist_ok=False)
    return pasta


def registrar_execucao(config, modelo, extras=None):
    pasta = Path(config.checkpoint).resolve().parent
    pasta.mkdir(parents=True, exist_ok=True)
    destino = pasta / 'parametros.json'
    anterior = json.loads(destino.read_text()) if destino.exists() else {}
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=RAIZ, text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    metadados = dict(anterior)
    metadados.update(modelo=modelo, config=asdict(config),
                     criado_em=anterior.get('criado_em', datetime.now().astimezone().isoformat()),
                     caminho_resultados=str(pasta), commit_base=commit,
                     observacao_codigo='Código local pode conter alterações além do commit-base.',
                     arquitetura={'backbone': 'ResNet-18' if modelo == 'res_theo18' else 'ResNet-34',
                                  'layer4_removida': True, 'classificador': 'Linear(256, 1)',
                                  'dropout_probabilidade': getattr(config, 'dropout_probabilidade', 0),
                                  'dropout_posicoes': getattr(config, 'dropout_posicoes', ())},
                     rotulos={'Cat': 0, 'Dog': 1}, resolucao=[224, 224], estado=anterior.get('estado', 'iniciado'))
    # Snapshot dos fontes para preservar a implementação associada aos pesos.
    pacote = RAIZ / 'residuals networks' / modelo
    fontes = pasta / 'codigo'
    fontes.mkdir(exist_ok=True)
    for fonte in (*pacote.glob('*.py'), pacote / 'requirements.txt', Path(__file__), RAIZ / 'criar_dataset_ampliado.py'):
        alvo = fontes / fonte.name
        if fonte.exists() and not alvo.exists():
            alvo.write_bytes(fonte.read_bytes())
    try:
        import torch, torchvision
        metadados['ambiente'] = {'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
                                 'cuda_runtime': torch.version.cuda,
                                 'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None}
    except ImportError:
        pass
    if extras:
        metadados.update(extras)
    temporario = destino.with_suffix('.json.tmp')
    temporario.write_text(json.dumps(metadados, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporario.replace(destino)
    resumo = f'''# Execução: {modelo}

- Batch: {config.batch_size}
- Épocas planejadas: {config.epocas}
- Épocas iniciais congeladas: {config.epocas_backbone_congelado}
- Passos iniciais congelados (usados somente se épocas=None): {config.passos_backbone_congelado}
- Congelamento permanente: {config.congelar_backbone}
- Otimizador: {config.otimizador}
- Learning rate: {config.learning_rate}
- Dropout: {getattr(config, 'dropout_probabilidade', 0)}
- Estado: {metadados['estado']}

`parametros.json` contém todas as configurações, incluindo semente, dataset,
opções do otimizador e posições do dropout. `codigo/` preserva os fontes usados.
`ultima_epoca.pt` contém pesos, configuração, identidade do manifesto e estado do
modelo/otimizador. `historico.csv` e os gráficos pertencem somente a esta execução.
A configuração salva prevalece sobre alterações posteriores no código do projeto.
'''
    (pasta / 'IDENTIFICACAO.md').write_text(resumo, encoding='utf-8')
    return metadados
