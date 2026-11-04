"""SVGs das variantes Theo Curto e Theo 28/42/64."""
from gerar_res_theo_especial18_antigo import DESTINO, desenhar, plt


def gerar():
    plt.rcParams['font.family'] = 'DejaVu Sans'
    rodape = (
        'AdamW · weight decay 0,01 · batch 16 · He normal fan_out',
        '80 épocas: originais 1–50; ampliadas 51–80 · validação original',
        'LR: 0,001 até 50; 0,000001→0,001 entre 51–75; fixa depois',
        'Dropout: 0% na 1 → 20% na 25 → 10% na 50 → 5% na 75–80')
    for nome, canais in (('theo_curto', (32,64)),
                         ('theo_28_42_64', (28,42,64))):
        fig, ax = plt.subplots(figsize=(8,18))
        fim = desenhar(ax, nome=nome, canais_estagios=canais, rodape=rodape)
        ax.set_ylim(fim+.8,0)
        fig.set_size_inches(8,(fim+1)*.81)
        fig.subplots_adjust(left=.02,right=.98,top=.99,bottom=.045)
        fig.text(.5,.026,'D: dropout após operações selecionadas; desligado na validação. Conv1 e fc sem dropout.',ha='center',fontsize=8.6)
        fig.text(.5,.014,'Setas laterais: identidade ou projeção 1×1. Somas (+) sem dropout.',ha='center',fontsize=8.6)
        for extensao in ('svg','png'):
            fig.savefig(DESTINO / f'arquitetura_{nome}.{extensao}',dpi=120,facecolor='white')
        plt.close(fig)


if __name__ == '__main__':
    gerar()
