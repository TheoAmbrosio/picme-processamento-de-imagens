# Conferência da atualização —06/11/2026

Commit c02a775 da main. Repositório original mantido sem alterações locais.

## Pendências resolvidas

- Pesos, históricos por época e predições de validação das quatro versões publicados.
- README distingue a arquitetura base V1–V3 da V4 e declara versões independentes,
  treinadas do zero, sem congelamento e com checkpoint por menor val_loss.
- Históricos contêm15/23/30/30 épocas, numeradas sem lacunas ou repetições.
- Predições:5000 exemplos por versão,2500 gatos e2500 cachorros. Limiar0.5
  consistente com as classes gravadas. Acurácia coincide com a melhor época
  por val_loss do histórico; BCE difere por menos que3e-7, compatível com
  diferenças de arredondamento/cálculo. F1 reproduz o resultado documentado.
- Pesos das quatro versões lidos com weights_only=True, todos finitos.
  V4 carregada estritamente na CNN atual; entrada artificial1×3×384×384
  produz saída1×1 finita. Não foi executada avaliação real nas imagens.

| Versão | Épocas registradas | Época selecionada por val_loss | Acurácia de validação |
|---|---:|---:|---:|
| V1 |15|15|91,54%|
| V2 |23|18|93,26%|
| V3 |30|28|92,42%|
| V4 |30|30|94,64%|

As épocas selecionadas são determinadas pelos históricos/README. Os arquivos
de pesos são state_dict, sem número de época gravado internamente.

## Ponto a confirmar com o autor: imagem nova sem preparação

`data/data-setting.ipynb` prepara os arquivos com resize mantendo proporção
e recorte central. Porém `src/predict.py`, para uma nova imagem crua, usa
`get_default_transform()` de src/dataset.py, cujo Resize((384,384)) estica
a imagem sem esse recorte proporcional inicial.

Assim, há diferença entre a preparação offline dos arquivos e a transformação
de uma imagem nova. Não alteramos o código nem escolhemos qual fluxo deve
valer no teste geral; o autor precisa confirmar o pré-processamento pretendido.
Essa observação não invalida os cálculos dos históricos recebidos.

## Material organizado

Evolução agora tem CSVs e plots SVG/PNG/PDF por versão, além do painel geral.
Resultados numéricos detalhados em CONFERENCIA_ATUALIZACAO.json.
Não foi executado treino, teste final ou alteração metodológica.


## V4 Final — bd92c13

Nova etapa de10 épocas, derivada do checkpoint V4/30. Checkpoint por menor
val_loss na10 (94,76%), máximo de acurácia94,80% nas3 e9. Conferência detalhada
em CONFERENCIA_FINAL.json: carregamento estrito, pesos finitos, forward artificial
1×3×384×384→1×1;5000 predições conferidas com o histórico. Não avaliado nas imagens.
O ponto sobre preparação offline versus imagem nova descrito acima permanece.
O congelamento anunciado se aplica aos parâmetros; os buffers de BatchNorm
continuam atualizáveis no código do autor. Nenhum código original foi alterado.
