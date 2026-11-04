#!/usr/bin/env python3
"""Baixa os originais do Kaggle quando necessário e gera dataset ampliado.

Cada imagem de TREINO legível gera 16 JPEGs RGB de 224x224, mantendo a classe.
Validação e teste recebem uma única versão determinística, sem ampliações.
Originais: dataset/PetImages; saídas: dataset_ampliado/224x224/PetImages.

    2 geometrias (recorte quadrado central | imagem inteira esticada)
  x 2 espelhamentos (normal | espelhada horizontalmente)
  x 4 rotações (0, 90, 180 e 270 graus, sentido anti-horário)

Nome das saídas:
    Cat_12_jpg__recorte__espelhada__r090.jpg
    ^^^^^^^^^^  ^^^^^^^  ^^^^^^^^^  ^^^^
    origem      geometria espelho   rotação

Treino/validação/teste: cada original é sorteado ANTES de gerar variantes, a partir do
SHA-256 do seu conteúdo. Assim, as 16 variantes ficam sempre do mesmo lado e
cópias byte a byte de uma mesma imagem também. As pastas não mudam; o sorteio
fica em manifesto_originais.csv e manifesto_variantes.csv.

Retomada: cada saída é gravada em arquivo temporário e renomeada ao final.
Basta rodar de novo: o que já está pronto é pulado e nada é duplicado.

Dependências: Pillow >= 9.1 e kagglehub >= 1.0 (Python 3.10+)

PowerShell:
    py -m pip install -r requirements-dataset.txt
    py ".\\criar_dataset_ampliado.py"
    py ".\\criar_dataset_ampliado.py" --apenas-verificar

Código de saída: 0 se a checagem final passou; 1 se não passou; 130 se
interrompido.
"""
from __future__ import annotations

import argparse
import collections
import csv
import errno
import hashlib
import io
import json
import multiprocessing
import os
import re
import shutil
import signal
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageFile, ImageOps
sys.path.insert(0, str(Path(__file__).resolve().parent / "residuals networks"))
from res_theo34.config_res_theo34 import CONFIG

RAIZ = Path(__file__).resolve().parent
ORIGEM = RAIZ / "dataset" / "PetImages"
LADO = 224
SAIDA_BASE = RAIZ / "dataset_ampliado" / f"{LADO}x{LADO}"
SAIDA = SAIDA_BASE / "PetImages"
MANIFESTO_ORIGINAIS = SAIDA_BASE / "manifesto_originais.csv"
MANIFESTO_VARIANTES = SAIDA_BASE / "manifesto_variantes.csv"
LOG_ERROS = SAIDA_BASE / "erros.log"
RELATORIO = SAIDA_BASE / "relatorio.txt"

CLASSES = ("Cat", "Dog")
# Arquivos JPG por classe no Kaggle; inclui eventuais imagens ilegíveis.
CONTAGEM_INFORMADA = {"Cat": 12_500, "Dog": 12_500}
KAGGLE_DATASET = "shaunthesheep/microsoft-catsvsdogs-dataset"

QUALIDADE_JPEG = 90
GEOMETRIAS = ("recorte", "esticada")
ESPELHOS = ("normal", "espelhada")
ROTACOES = {
    0: None,
    90: Image.Transpose.ROTATE_90,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_270,
}
VARIANTES_POR_IMAGEM = len(GEOMETRIAS) * len(ESPELHOS) * len(ROTACOES)
BYTES_ESTIMADOS_POR_SAIDA = 40_000

COLUNAS = [
    "classe", "arquivo", "tamanho", "mtime_ns", "sha256",
    "largura", "altura", "split", "rotulo", "status", "erro",
]
STATUS_VALIDOS = {"ok", "ilegivel", "erro_saida"}

# Um arquivo truncado precisa contar como ilegível, não ser parcialmente aceito.
ImageFile.LOAD_TRUNCATED_IMAGES = False


class SemEspaco(Exception):
    """O disco encheu: continuar só produziria mais falhas."""


# ---------------------------------------------------------------- nomes


def id_origem(arquivo: str) -> str:
    """Identificador seguro e injetivo para o nome do arquivo de origem."""
    base = re.sub(r"[^0-9A-Za-z-]+", "_", arquivo).strip("_") or "arquivo"
    if re.fullmatch(r"[0-9A-Za-z-]+\.[0-9A-Za-z]+", arquivo):
        return base
    # Nomes incomuns poderiam colidir ao serem limpos ("a b.jpg" e "a_b.jpg").
    return f"{base}_{hashlib.sha1(arquivo.encode('utf-8', 'surrogatepass')).hexdigest()[:8]}"


def variantes(classe: str, arquivo: str, split: str = "treino"):
    """16 saídas no treino; uma versão determinística em validação/teste."""
    origem = id_origem(arquivo)
    if split != "treino":
        return [(f"{classe}_{origem}__avaliacao.jpg", "avaliacao", "normal", 0)]
    return [
        (f"{classe}_{origem}__{geo}__{esp}__r{rot:03d}.jpg", geo, esp, rot)
        for geo in GEOMETRIAS
        for esp in ESPELHOS
        for rot in ROTACOES
    ]


# ---------------------------------------------------------------- imagem


def para_rgb(im: Image.Image) -> Image.Image:
    """Converte qualquer modo para RGB; transparência vira fundo branco."""
    if im.mode in ("I", "I;16", "I;16L", "I;16B", "I;16N"):
        convertida = im.convert("I")
        maximo = convertida.getextrema()[1]
        if maximo > 255:  # 16 bits: sem reescalar, tudo saturaria em branco
            convertida = convertida.point(lambda v: v * (1 / 256))
        im = convertida.convert("L")
    elif im.mode == "F":
        im = im.convert("L")

    if im.mode in ("RGBA", "LA", "PA", "RGBa", "La") or "transparency" in im.info:
        rgba = im.convert("RGBA")
        fundo = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(fundo, rgba).convert("RGB")
    return im if im.mode == "RGB" else im.convert("RGB")


def abrir_rgb(dados: bytes) -> Image.Image:
    """Decodifica por inteiro (detecta truncamento), aplica EXIF e passa a RGB."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # avisos de EXIF corrompido não invalidam
        with Image.open(io.BytesIO(dados)) as im:
            im.load()
            try:
                im = ImageOps.exif_transpose(im)
            except Exception:
                pass  # EXIF inválido: os pixels continuam utilizáveis
            return para_rgb(im)


def aplicar_geometria(im: Image.Image, geometria: str) -> Image.Image:
    if geometria == "recorte":
        largura, altura = im.size
        lado = min(largura, altura)
        esquerda, topo = (largura - lado) // 2, (altura - lado) // 2
        im = im.crop((esquerda, topo, esquerda + lado, topo + lado))
    return im.resize((LADO, LADO), Image.Resampling.LANCZOS)


def salvar_atomico(im: Image.Image, destino: str) -> None:
    """Só um JPEG completo recebe o nome final; retomar nunca vê arquivo parcial."""
    temporario = destino + ".tmp"
    try:
        with open(temporario, "wb") as arquivo:
            im.save(arquivo, format="JPEG", quality=QUALIDADE_JPEG)
        for tentativa in range(6):
            try:
                os.replace(temporario, destino)
                return
            except PermissionError:  # OneDrive/antivírus podem travar por instantes
                if tentativa == 5:
                    raise
                time.sleep(0.2 * 2**tentativa)
    except BaseException:
        try:
            os.remove(temporario)
        except OSError:
            pass
        raise


def aplicar_avaliacao(im: Image.Image) -> Image.Image:
    """Geometria do preset ImageNet V1: menor lado 256, bilinear, crop central 224.

    Não normaliza pixels; a normalização ImageNet acontece uma vez no DataLoader.
    """
    largura, altura = im.size
    if largura <= altura:
        tamanho = (256, int(256 * altura / largura))
    else:
        tamanho = (int(256 * largura / altura), 256)
    im = im.resize(tamanho, Image.Resampling.BILINEAR)
    esquerda = int(round((im.width - LADO) / 2))
    topo = int(round((im.height - LADO) / 2))
    return im.crop((esquerda, topo, esquerda + LADO, topo + LADO))


def gerar_saidas(rgb, classe, arquivo, pasta, sobrescrever, split) -> None:
    nomes = variantes(classe, arquivo, split)
    por_geometria = collections.defaultdict(list)
    for variante in nomes:
        por_geometria[variante[1]].append(variante)

    for geometria, grupo in por_geometria.items():
        pendentes = [
            v for v in grupo
            if sobrescrever or not _arquivo_nao_vazio(os.path.join(pasta, v[0]))
        ]
        if not pendentes:
            continue
        base = aplicar_avaliacao(rgb) if geometria == "avaliacao" else aplicar_geometria(rgb, geometria)
        espelhada = base.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        for nome, _, espelho, rotacao in pendentes:
            im = espelhada if espelho == "espelhada" else base
            operacao = ROTACOES[rotacao]
            if operacao is not None:
                im = im.transpose(operacao)
            salvar_atomico(im, os.path.join(pasta, nome))


def _arquivo_nao_vazio(caminho: str) -> bool:
    try:
        return os.path.getsize(caminho) > 0
    except OSError:
        return False


def descrever_erro(exc: Exception) -> str:
    """Mensagem em uma linha e sem endereços de memória, estável entre execuções."""
    texto = re.sub(r"<_io\.BytesIO object at 0x[0-9A-Fa-f]+>", "(conteúdo do arquivo)", str(exc))
    return f"{type(exc).__name__}: {' '.join(texto.split())}"


def processar(tarefa) -> dict:
    """Executado nos processos auxiliares; uma imagem por vez fica na memória."""
    (classe, caminho, arquivo, tamanho, mtime_ns, pasta, sobrescrever,
     frac_validacao, frac_teste, semente) = tarefa[:10]
    split_planejado = tarefa[10] if len(tarefa) > 10 else None
    linha = dict.fromkeys(COLUNAS, "")
    linha.update(classe=classe, arquivo=arquivo, tamanho=tamanho, mtime_ns=mtime_ns,
                 rotulo=0 if classe == "Cat" else 1)

    try:
        with open(caminho, "rb") as origem:
            dados = origem.read()
        linha["sha256"] = hashlib.sha256(dados).hexdigest()
        linha["split"] = split_planejado or split_de(linha["sha256"], semente, frac_teste, frac_validacao)
        rgb = abrir_rgb(dados)
        del dados
        linha["largura"], linha["altura"] = rgb.size
    except Exception as exc:
        linha.update(status="ilegivel", erro=descrever_erro(exc))
        return linha

    # Uma falha aqui é do disco, não da imagem: não pode reduzir as entradas válidas.
    try:
        gerar_saidas(rgb, classe, arquivo, pasta, sobrescrever, linha["split"])
    except Exception as exc:
        linha.update(status="erro_saida", erro=descrever_erro(exc))
        linha["_sem_espaco"] = getattr(exc, "errno", None) == errno.ENOSPC
        return linha

    linha["status"] = "ok"
    return linha


def _iniciar_auxiliar() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C é tratado pelo principal


# ---------------------------------------------------------------- arquivos de estado


def garantir_dataset() -> None:
    """Completa os originais a partir do cache/download do Kaggle, sem sobrescrever."""
    def contar_jpg(pasta: Path) -> int:
        if not pasta.is_dir():
            return 0
        return sum(f.is_file() and f.suffix.lower() == ".jpg" for f in pasta.iterdir())

    if all(contar_jpg(ORIGEM / c) >= CONTAGEM_INFORMADA[c] for c in CLASSES):
        print(f"Dataset original encontrado: {ORIGEM}", flush=True)
        return
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "Instale as dependências: python -m pip install -r requirements-dataset.txt"
        ) from exc

    print(f"Obtendo dataset do Kaggle: {KAGGLE_DATASET}", flush=True)
    try:
        raiz_download = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    except Exception as exc:
        raise RuntimeError(
            "Falha ao baixar do Kaggle. Confira a conexão e, se solicitado, "
            "autentique-se com kagglehub.login()."
        ) from exc
    candidatos = [raiz_download, *raiz_download.rglob("PetImages")]
    fonte = next((
        p for p in candidatos
        if all(contar_jpg(p / c) >= CONTAGEM_INFORMADA[c] for c in CLASSES)
    ), None)
    if fonte is None:
        raise RuntimeError(f"Dataset completo não encontrado no download: {raiz_download}")

    copiados = 0
    for classe in CLASSES:
        destino_classe = ORIGEM / classe
        destino_classe.mkdir(parents=True, exist_ok=True)
        for arquivo in sorted((fonte / classe).iterdir()):
            if not arquivo.is_file() or arquivo.suffix.lower() != ".jpg":
                continue
            destino = destino_classe / arquivo.name
            if destino.exists():
                continue
            temporario = destino.with_name(destino.name + ".download.tmp")
            shutil.copyfile(arquivo, temporario)
            os.replace(temporario, destino)
            copiados += 1
    print(f"Dataset original pronto: {ORIGEM} ({copiados} arquivos copiados)", flush=True)


def split_de(sha256: str, semente: int, frac_teste: float, frac_validacao: float) -> str:
    digest = hashlib.sha256(f"{semente}:{sha256}".encode()).digest()
    valor = int.from_bytes(digest[:8], "big") / 2**64
    if valor < frac_teste:
        return "teste"
    return "validacao" if valor < frac_teste + frac_validacao else "treino"


def listar_origens(classe: str) -> dict:
    with os.scandir(ORIGEM / classe) as entradas:
        return {
            e.name: (e.stat().st_size, e.stat().st_mtime_ns)
            for e in entradas if e.is_file() and Path(e.name).suffix.lower() in {".jpg", ".jpeg", ".png"}
        }


def listar_saidas(classe: str) -> dict:
    """Nome -> bytes dos arquivos em disco, ignorando temporários."""
    pasta = SAIDA / classe
    if not pasta.is_dir():
        return {}
    with os.scandir(pasta) as entradas:
        return {
            e.name: e.stat().st_size
            for e in entradas if e.is_file() and not e.name.endswith(".tmp")
        }


def remover_temporarios() -> None:
    for classe in CLASSES:
        with os.scandir(SAIDA / classe) as entradas:
            sobras = [e.path for e in entradas if e.name.endswith(".tmp")]
        for caminho in sobras:
            os.remove(caminho)


def carregar_manifesto() -> dict:
    """Lê o estado; uma última linha cortada pela interrupção é descartada."""
    if not MANIFESTO_ORIGINAIS.exists():
        return {}
    linhas = {}
    with open(MANIFESTO_ORIGINAIS, encoding="utf-8", newline="") as f:
        for linha in csv.DictReader(f):
            if linha.get("status") not in STATUS_VALIDOS or linha.get("classe") not in CLASSES:
                continue
            linha["erro"] = linha.get("erro") or ""
            linhas[(linha["classe"], linha["arquivo"])] = {c: linha.get(c, "") for c in COLUNAS}
    return linhas


def gravar_csv_atomico(destino: Path, colunas, linhas) -> None:
    temporario = destino.with_name(destino.name + ".tmp")
    with open(temporario, "w", encoding="utf-8", newline="") as f:
        escritor = csv.writer(f)
        escritor.writerow(colunas)
        escritor.writerows(linhas)
    os.replace(temporario, destino)


def gravar_manifesto_estado(linhas: dict) -> None:
    ordenadas = sorted(linhas.values(), key=lambda l: (l["classe"], l["arquivo"]))
    gravar_csv_atomico(
        MANIFESTO_ORIGINAIS, COLUNAS, ([l[c] for c in COLUNAS] for l in ordenadas)
    )


def atual(linha, assinatura) -> bool:
    return linha is not None and (
        str(linha["tamanho"]), str(linha["mtime_ns"])
    ) == (str(assinatura[0]), str(assinatura[1]))


# ---------------------------------------------------------------- execução


def montar_tarefas(origens, manifesto, existentes, frac_validacao, frac_teste, semente, plano=None):
    """Gera apenas o trabalho que falta, sem abrir imagens já concluídas."""
    for classe in CLASSES:
        pasta = str(SAIDA / classe)
        for arquivo, assinatura in sorted(origens[classe].items()):
            linha = manifesto.get((classe, arquivo))
            mudou = linha is not None and not atual(linha, assinatura)
            if linha is not None and not mudou and linha["status"] == "ok":
                if all(existentes[classe].get(v[0], 0) > 0
                       for v in variantes(classe, arquivo, linha["split"])):
                    continue
            yield (
                classe, os.path.join(pasta_origem(classe), arquivo), arquivo,
                assinatura[0], assinatura[1], pasta, mudou, frac_validacao, frac_teste, semente,
                plano.get((classe, arquivo)) if plano is not None else None,
            )


def sortear_grupos_balanceados(grupos, semente, frac_validacao):
    """Sorteia grupos inteiros; cotas iguais de originais por classe na validação."""
    import random
    contagens = collections.Counter(classe for grupo in grupos for classe, _ in grupo)
    if contagens["Cat"] != contagens["Dog"]:
        raise ValueError(f"Contagens válidas diferentes; requer decisão do participante: {contagens}")
    alvo = round(contagens["Cat"] * frac_validacao)
    cotas = {classe: alvo for classe in CLASSES}
    grupos = list(grupos)
    random.Random(semente).shuffle(grupos)
    plano = {}
    for grupo in grupos:
        quantidade = collections.Counter(classe for classe, _ in grupo)
        validar = all(quantidade[c] <= cotas[c] for c in CLASSES)
        split = "validacao" if validar else "treino"
        if validar:
            for c in CLASSES:
                cotas[c] -= quantidade[c]
        for chave in grupo:
            plano[chave] = split
    if any(cotas.values()):
        raise ValueError(f"Grupos duplicados impediram as cotas exatas: {cotas}")
    return plano


def planejar_divisao_balanceada(origens, semente, frac_validacao, frac_teste):
    if frac_teste != 0:
        raise ValueError("Esta divisão foi definida pelo participante sem conjunto de teste.")
    grupos = collections.defaultdict(list)
    inventario = []
    for classe in CLASSES:
        for arquivo in sorted(origens[classe]):
            dados = (ORIGEM / classe / arquivo).read_bytes()
            sha = hashlib.sha256(dados).hexdigest()
            try:
                abrir_rgb(dados)
            except Exception as exc:
                inventario.append(dict(classe=classe, arquivo=arquivo, status="ilegivel", erro=descrever_erro(exc)))
                continue
            grupos[sha].append((classe, arquivo))
            inventario.append(dict(classe=classe, arquivo=arquivo, status="ok", sha256=sha))
    plano = sortear_grupos_balanceados(list(grupos.values()), semente, frac_validacao)
    registro = SAIDA_BASE / "divisao_originais.json"
    conteudo = dict(semente=semente, inventario=inventario,
                    splits=[dict(classe=c, arquivo=a, split=s) for (c,a),s in sorted(plano.items())],
                    duplicados={h: g for h,g in grupos.items() if len(g)>1})
    temporario = registro.with_suffix(".json.tmp")
    temporario.write_text(json.dumps(conteudo, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporario, registro)
    print("Divisão balanceada por originais:", collections.Counter((c,s) for (c,a),s in plano.items()), flush=True)
    return plano


def pasta_origem(classe: str) -> str:
    return str(ORIGEM / classe)


def registrar_erro(log, linha) -> None:
    carimbo = datetime.now().isoformat(timespec="seconds")
    log.write(f"{carimbo}\t{linha['status']}\t{linha['classe']}/{linha['arquivo']}\t{linha['erro']}\n")
    log.flush()


def executar(tarefas, total, manifesto, workers) -> None:
    with open(MANIFESTO_ORIGINAIS, "a", encoding="utf-8", newline="") as estado, \
            open(LOG_ERROS, "a", encoding="utf-8") as log:
        escritor = csv.writer(estado)

        def receber(resultados):
            inicio = ultimo = time.monotonic()
            feitos = 0
            for linha in resultados:
                sem_espaco = linha.pop("_sem_espaco", False)
                manifesto[(linha["classe"], linha["arquivo"])] = linha
                escritor.writerow([linha[c] for c in COLUNAS])
                estado.flush()
                if linha["status"] != "ok":
                    registrar_erro(log, linha)
                if sem_espaco:
                    raise SemEspaco(linha["erro"])
                feitos += 1
                agora = time.monotonic()
                if feitos == total or agora - ultimo >= 15:
                    ultimo = agora
                    ritmo = feitos / max(agora - inicio, 1e-9)
                    restante = (total - feitos) / max(ritmo, 1e-9)
                    print(
                        f"  {feitos:,}/{total:,} imagens | {ritmo:.1f} img/s | "
                        f"restam cerca de {restante / 60:.1f} min",
                        flush=True,
                    )

        if workers == 1:
            receber(map(processar, tarefas))
            return

        pool = multiprocessing.Pool(workers, initializer=_iniciar_auxiliar)
        try:
            receber(pool.imap_unordered(processar, tarefas, chunksize=8))
            pool.close()
        except BaseException:
            pool.terminate()
            raise
        finally:
            pool.join()


# ---------------------------------------------------------------- relatório


def auditar(origens, manifesto) -> dict:
    resumo = {}
    hashes = collections.Counter()
    for classe in CLASSES:
        contagem = collections.Counter()
        split_originais = collections.Counter()
        esperadas = set()
        erros = []
        for arquivo, assinatura in sorted(origens[classe].items()):
            linha = manifesto.get((classe, arquivo))
            if not atual(linha, assinatura):
                contagem["pendente"] += 1
                continue
            contagem[linha["status"]] += 1
            if linha["status"] == "ok":
                split_originais[linha["split"]] += 1
                hashes[linha["sha256"]] += 1
                esperadas.update(v[0] for v in variantes(classe, arquivo, linha["split"]))
            else:
                erros.append((linha["status"], arquivo, linha["erro"]))

        disco = listar_saidas(classe)
        presentes = {nome for nome, tamanho in disco.items() if tamanho > 0}
        resumo[classe] = {
            "arquivos": len(origens[classe]),
            "validas": contagem["ok"],
            "ilegiveis": contagem["ilegivel"],
            "falhas_saida": contagem["erro_saida"],
            "pendentes": contagem["pendente"],
            "esperadas": len(esperadas),
            "em_disco": len(disco),
            "bytes": sum(disco.values()),
            "faltando": len(esperadas - presentes),
            "sobrando": len(set(disco) - esperadas),
            "treino": split_originais["treino"],
            "validacao": split_originais["validacao"],
            "teste": split_originais["teste"],
            "erros": erros,
        }
    resumo["_duplicatas"] = sum(1 for n in hashes.values() if n > 1)
    return resumo


def montar_relatorio(resumo, frac_validacao, frac_teste, semente):
    cat, dog = resumo["Cat"], resumo["Dog"]
    aprovado = True
    linhas = [
        f"Relatório do dataset ampliado | {datetime.now().isoformat(timespec='seconds')}",
        f"Origem: {ORIGEM}",
        f"Saída:  {SAIDA}",
        f"Dimensões das variantes: {LADO}x{LADO}",
        "",
    ]
    for classe in CLASSES:
        r = resumo[classe]
        informado = CONTAGEM_INFORMADA[classe]
        ok = (
            r["em_disco"] == r["esperadas"]
            == r["treino"] * VARIANTES_POR_IMAGEM + r["validacao"] + r["teste"]
            and not (r["faltando"] or r["sobrando"] or r["pendentes"] or r["falhas_saida"])
        )
        aprovado = aprovado and ok
        linhas += [
            f"[{classe}]",
            f"  Arquivos na origem:       {r['arquivos']:>9,} (informados: {informado:,}; diferença: {r['arquivos'] - informado:+,})",
            f"  Entradas válidas:         {r['validas']:>9,}",
            f"  Ilegíveis:                {r['ilegiveis']:>9,}",
            f"  Falhas ao gravar:         {r['falhas_saida']:>9,}",
            f"  Ainda não processados:    {r['pendentes']:>9,}",
            f"  Saídas esperadas:         {r['esperadas']:>9,}",
            f"  Saídas em disco:          {r['em_disco']:>9,}",
            f"  Faltando / sem origem:    {r['faltando']:>9,} / {r['sobrando']:,}",
            f"  Originais treino / validação / teste: {r['treino']:,} / {r['validacao']:,} / {r['teste']:,}",
            f"  Variantes treino / validação / teste: {r['treino'] * VARIANTES_POR_IMAGEM:,} / {r['validacao']:,} / {r['teste']:,}",
            f"  Tamanho em disco:         {r['bytes'] / 1024**3:>9.2f} GiB",
            f"  Checagem da contagem:     {'OK' if ok else 'FALHOU'}",
            "",
        ]
    linhas += [
        "[Diferença entre classes]",
        f"  Originais válidos, Cat - Dog: {cat['validas'] - dog['validas']:+,}",
        f"  Saídas em disco, Cat - Dog:   {cat['em_disco'] - dog['em_disco']:+,}",
        "  Nenhuma imagem foi criada para igualar as classes.",
        "",
        "[Separação treino/validação/teste]",
        f"  Fração de validação: {frac_validacao} | teste: {frac_teste} | semente: {semente}",
        "  Sorteio por conteúdo do original; lista completa em manifesto_variantes.csv.",
        f"  Conteúdos repetidos entre originais válidos: {resumo['_duplicatas']:,} (sempre no mesmo lado)",
        "",
    ]
    erros = [(c, *e) for c in CLASSES for e in resumo[c]["erros"]]
    linhas.append(f"[Arquivos com erro: {len(erros):,}]")
    linhas += [f"  {c}/{arquivo} [{status}] {erro}" for c, status, arquivo, erro in erros]
    linhas += ["", f"RESULTADO DA CHECAGEM: {'APROVADO' if aprovado else 'REPROVADO'}"]
    return "\n".join(linhas) + "\n", aprovado


def gravar_manifesto_variantes(origens, manifesto) -> None:
    def linhas():
        for classe in CLASSES:
            for arquivo in sorted(origens[classe]):
                linha = manifesto.get((classe, arquivo))
                if linha is None or linha["status"] != "ok":
                    continue
                split = linha["split"]
                for nome, geo, esp, rot in variantes(classe, arquivo, split):
                    yield [split, classe, linha["rotulo"], f"PetImages/{classe}/{nome}",
                           arquivo, linha["sha256"], geo, esp, rot]

    gravar_csv_atomico(
        MANIFESTO_VARIANTES,
        ["split", "classe", "rotulo", "caminho", "arquivo_origem", "sha256",
         "geometria", "espelho", "rotacao"],
        linhas(),
    )


# ---------------------------------------------------------------- principal


def ler_argumentos():
    parser = argparse.ArgumentParser(description="Gera dataset_ampliado com 16 variantes por imagem.")
    parser.add_argument("--divisao", choices=("hash", "aleatoria_balanceada"), default=CONFIG.divisao)
    parser.add_argument("--workers", type=int, default=max(1, min((os.cpu_count() or 2) - 1, 8)),
                        help="processos paralelos (padrão: núcleos - 1, no máximo 8)")
    parser.add_argument("--frac-teste", type=float, default=CONFIG.frac_teste,
                        help="fração de teste definida pelo participante")
    parser.add_argument("--frac-validacao", type=float, default=CONFIG.frac_validacao,
                        help="fração de validação definida pelo participante")
    parser.add_argument("--semente", type=int, default=CONFIG.semente,
                        help="semente da divisão por conteúdo")
    parser.add_argument("--saida", type=Path, default=Path(CONFIG.manifesto).parent,
                        help="outra pasta de saída, se já houver um dataset antigo")
    parser.add_argument("--apenas-verificar", action="store_true",
                        help="não gera imagens; apenas confere e imprime as contagens")
    args = parser.parse_args()
    if args.frac_validacao is None or args.frac_teste is None:
        parser.error("Defina frac_validacao e frac_teste em residuals networks/res_theo34/config_res_theo34.py ou nas opções.")
    if not (0 <= args.frac_teste < 1 and 0 < args.frac_validacao < 1
            and args.frac_teste + args.frac_validacao < 1):
        parser.error("Treino e validação devem ser positivos; teste pode ser zero; as frações devem somar 1.")
    if args.workers < 1:
        parser.error("--workers deve ser pelo menos 1")
    return args


def configurar_dataset(args) -> None:
    """Bloqueia retomada com splits diferentes ou saídas do esquema antigo."""
    global SAIDA_BASE, SAIDA, MANIFESTO_ORIGINAIS, MANIFESTO_VARIANTES, LOG_ERROS, RELATORIO
    SAIDA_BASE = args.saida if args.saida.is_absolute() else RAIZ / args.saida
    SAIDA = SAIDA_BASE / "PetImages"
    MANIFESTO_ORIGINAIS = SAIDA_BASE / "manifesto_originais.csv"
    MANIFESTO_VARIANTES = SAIDA_BASE / "manifesto_variantes.csv"
    LOG_ERROS = SAIDA_BASE / "erros.log"
    RELATORIO = SAIDA_BASE / "relatorio.txt"
    configuracao = {
        "schema": 2, "rotulos": {"Cat": 0, "Dog": 1}, "lado": LADO,
        "frac_validacao": args.frac_validacao, "frac_teste": args.frac_teste,
        "semente": args.semente, "normalizacao_salva": False,
        "avaliacao": "Resize(256, bilinear) + CenterCrop(224)",
    }
    if getattr(args, "divisao", "hash") != "hash":
        configuracao.update(divisao=args.divisao, aceitar_rotulos_duplicados=CONFIG.aceitar_rotulos_duplicados)
    arquivo = SAIDA_BASE / "config_dataset.json"
    if arquivo.exists():
        if json.loads(arquivo.read_text(encoding="utf-8")) != configuracao:
            raise ValueError("Configuração do dataset diverge. Use --saida com uma pasta nova.")
    elif MANIFESTO_ORIGINAIS.exists() or (SAIDA.exists() and any(SAIDA.rglob("*.jpg"))):
        raise ValueError("Dataset antigo sem divisão segura em três conjuntos. Use --saida com uma pasta nova.")
    elif args.apenas_verificar:
        raise ValueError("Dataset ainda não foi preparado nesta pasta.")
    else:
        SAIDA_BASE.mkdir(parents=True, exist_ok=True)
        temporario = arquivo.with_suffix(".json.tmp")
        temporario.write_text(json.dumps(configuracao, indent=2), encoding="utf-8")
        os.replace(temporario, arquivo)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    args = ler_argumentos()
    try:
        configurar_dataset(args)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not args.apenas_verificar:
        try:
            garantir_dataset()
        except (OSError, RuntimeError) as exc:
            print(f"Não foi possível preparar os originais: {exc}", file=sys.stderr)
            return 1

    faltantes = [str(ORIGEM / c) for c in CLASSES if not (ORIGEM / c).is_dir()]
    if faltantes:
        print("Pasta de origem não encontrada:\n  " + "\n  ".join(faltantes), file=sys.stderr)
        return 1

    origens = {c: listar_origens(c) for c in CLASSES}
    manifesto = carregar_manifesto()

    if not args.apenas_verificar:
        for classe in CLASSES:
            (SAIDA / classe).mkdir(parents=True, exist_ok=True)
        remover_temporarios()
        gravar_manifesto_estado(manifesto)  # elimina uma possível linha cortada
        existentes = {c: listar_saidas(c) for c in CLASSES}
        plano = None
        if getattr(args, "divisao", "hash") == "aleatoria_balanceada":
            plano = planejar_divisao_balanceada(origens, args.semente, args.frac_validacao, args.frac_teste)
        tarefas = list(montar_tarefas(origens, manifesto, existentes,
                                     args.frac_validacao, args.frac_teste, args.semente, plano))
        total = len(tarefas)
        ja_prontas = sum(len(origens[c]) for c in CLASSES) - total
        print(f"Originais: Cat={len(origens['Cat']):,} | Dog={len(origens['Dog']):,}")
        print(f"Já concluídos: {ja_prontas:,} | a processar: {total:,} | processos: {args.workers}")

        necessario = total * VARIANTES_POR_IMAGEM * BYTES_ESTIMADOS_POR_SAIDA
        livre = shutil.disk_usage(SAIDA_BASE).free
        if necessario > livre:
            print(
                f"AVISO: estimativa de {necessario / 1024**3:.1f} GiB necessários; "
                f"há {livre / 1024**3:.1f} GiB livres."
            )

        if total:
            try:
                executar(tarefas, total, manifesto, args.workers)
            except KeyboardInterrupt:
                print("\nInterrompido. Rode o mesmo comando para continuar de onde parou.")
                return 130
            except SemEspaco as exc:
                print(f"\nSem espaço em disco ({exc}). Libere espaço e rode de novo.", file=sys.stderr)
                return 1

        # Remove do estado os originais que deixaram de existir.
        manifesto = {
            chave: linha for chave, linha in manifesto.items()
            if chave[1] in origens[chave[0]]
        }
        gravar_manifesto_estado(manifesto)
        gravar_manifesto_variantes(origens, manifesto)

    resumo = auditar(origens, manifesto)
    texto, aprovado = montar_relatorio(resumo, args.frac_validacao, args.frac_teste, args.semente)
    print("\n" + texto)
    if not args.apenas_verificar:
        temporario = RELATORIO.with_name(RELATORIO.name + ".tmp")
        temporario.write_text(texto, encoding="utf-8")
        os.replace(temporario, RELATORIO)
        print(f"Relatório salvo em: {RELATORIO}")
    return 0 if aprovado else 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
