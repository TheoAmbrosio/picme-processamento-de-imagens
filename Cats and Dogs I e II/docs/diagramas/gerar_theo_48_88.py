"""Diagrama vertical Theo 48/88 com os seis caminhos residuais explícitos."""
from gerar_res_theo_especial18_antigo import DESTINO, desenhar, plt


def gerar():
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, ax = plt.subplots(figsize=(8,18))
    fim = desenhar(ax, nome='theo_48_88', canais_estagios=(32,48,88), rodape=(
        'AdamW · weight decay 0,01 · batch 16 · He normal fan_out',
        '80 épocas: originais 1–50; ampliadas 51–80 · validação original',
        'LR: 0,001 até 50; 0,000001→0,001 entre 51–75; fixa depois',
        'Dropout: 0% na 1 → 20% na 25 → 10% na 50 → 5% na 75–80'))
    ax.set_ylim(fim+.8,0)
    fig.set_size_inches(8,(fim+1)*.81)
    fig.subplots_adjust(left=.02,right=.98,top=.99,bottom=.045)
    fig.text(.5,.026,'D: dropout após operações selecionadas; desligado na validação. Conv1 e fc sem dropout.',ha='center',fontsize=8.6)
    fig.text(.5,.014,'Setas laterais: identidade ou projeção 1×1. Somas (+) sem dropout.',ha='center',fontsize=8.6)
    for extensao in ('png','svg','pdf'):
        fig.savefig(DESTINO / f'arquitetura_theo_48_88.{extensao}',dpi=180,facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    gerar()
