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
    dados = 'amp' if getattr(config, 'usar_dados_ampliados', True) else 'orig'
    origem = 'pre' if getattr(config, 'usar_pesos_pre_treinados', True) else 'zero'
    agenda = (f'_docresc{config.dropout_epoca_inicio}a{config.dropout_epoca_maximo}'
              if getattr(config, 'dropout_crescente', False) else '')
    conv1 = (f'_conv1pre{config.epocas_conv1_congelada}ep'
             if getattr(config, 'usar_pesos_pre_treinados_conv1', False) else '')
    pasta = RESULTADOS / modelo / f'{carimbo}_bs{config.batch_size}_ep{config.epocas}_{congelamento}_do{dropout}{agenda}_dados{dados}_{origem}{conv1}'
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
                     arquitetura={'backbone': ({
                                                'res_theo18': 'ResNet-18',
                                                'res_theo34': 'ResNet-34',
                                                'res_theo_especial': 'ResNet-18 estreita principal até layer3 (32/32/64/128)',
                                                'theo_ajuste': 'ResNet-18 estreita até layer3 (32/32/64/128), fine-tuning do checkpoint 155432',
                                                'theo_curto': 'BasicBlocks [2,2], canais 32/32/64, layer3 e layer4 removidas',
                                                'theo_48_88': 'BasicBlocks [2,2,2], canais 32/32/48/88, layer4 removida',
                                                'theo_28_42_64': 'BasicBlocks [2,2,2], canais 28/28/42/64, layer4 removida',
                                                'res_theo_especial34': 'ResNet-34 estreita completa (64/32/64/128/256)',
                                                'res_theo_especial50': 'ResNet-50 Bottleneck 1/4 completa (64/16/32/64/128)',
                                                'alex_theo': 'AlexNet ImageNet com saída binária',
                                               }.get(modelo, modelo)),
                                  'layer4_removida': (None if modelo == 'alex_theo' else
                                                      True if modelo in ('res_theo_especial', 'theo_ajuste')
                                                      else not modelo.startswith('res_theo_especial')),
                                  'classificador': ('Linear(64, 1)' if modelo in ('theo_curto', 'theo_28_42_64')
                                                    else 'Linear(88, 1)' if modelo == 'theo_48_88'
                                                    else 'Linear(512, 1)' if modelo == 'res_theo_especial50'
                                                    else 'Linear(4096, 1)' if modelo == 'alex_theo'
                                                    else 'Linear(128, 1)' if modelo in ('res_theo_especial', 'theo_ajuste')
                                                    else 'Linear(256, 1)' if modelo.startswith('res_theo_especial')
                                                    else 'Linear(256, 1)'),
                                  'dropout_probabilidade': getattr(config, 'dropout_probabilidade', 0),
                                  'dropout_posicoes': getattr(config, 'dropout_posicoes', ())},
                     rotulos={'Cat': 0, 'Dog': 1}, resolucao=[224, 224], estado=anterior.get('estado', 'iniciado'))
    # Snapshot dos fontes para preservar a implementação associada aos pesos.
    pacote = RAIZ / 'residuals networks' / modelo
    fontes = pasta / 'codigo'
    fontes.mkdir(exist_ok=True)
    for fonte in (*pacote.glob('*.py'), pacote / 'requirements.txt', Path(__file__), RAIZ / 'criar_dataset_ampliado.py', RAIZ / 'checkpoint_validacao.py'):
        alvo = fontes / fonte.name
        if fonte.exists() and not alvo.exists():
            alvo.write_bytes(fonte.read_bytes())
    if modelo in ('theo_curto', 'theo_48_88', 'theo_28_42_64'):
        dependencias = fontes / 'dependencias'
        dependencias.mkdir(exist_ok=True)
        for fonte in (RAIZ / 'residuals networks/agenda_duas_fases.py',
                      *(RAIZ / 'residuals networks/theo_ajuste').glob('*.py'),
                      *(RAIZ / 'residuals networks/res_theo_especial').glob('*.py')):
            subpasta = dependencias / fonte.parent.name
            subpasta.mkdir(exist_ok=True)
            alvo = subpasta / fonte.name
            if not alvo.exists():
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
- Dropout crescente: {getattr(config, 'dropout_crescente', False)}
- Intervalo do dropout crescente: épocas {getattr(config, 'dropout_epoca_inicio', None)} a {getattr(config, 'dropout_epoca_maximo', None)}
- Dados ampliados: {getattr(config, 'usar_dados_ampliados', True)}
- Preparação quando não ampliados: RGB + Resize((224, 224), bilinear), sem corte
- Pesos pré-treinados: {getattr(config, 'usar_pesos_pre_treinados', True)}
- Inicialização do zero: {getattr(config, 'inicializacao_do_zero', None)}
- Conv1 com pesos pré-treinados: {getattr(config, 'usar_pesos_pre_treinados_conv1', False)}
- Épocas com conv1 congelada: {getattr(config, 'epocas_conv1_congelada', 0)}
- Estado: {metadados['estado']}

`parametros.json` contém todas as configurações, incluindo semente, dataset,
opções do otimizador e posições do dropout. `codigo/` preserva os fontes usados.
`ultima_epoca.pt` contém pesos, configuração, identidade do manifesto e estado do
modelo/otimizador. `historico.csv` e os gráficos pertencem somente a esta execução.
A configuração salva prevalece sobre alterações posteriores no código do projeto.
'''
    (pasta / 'IDENTIFICACAO.md').write_text(resumo, encoding='utf-8')
    return metadados
