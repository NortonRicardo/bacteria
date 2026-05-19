"""
1.2 - Detecção da Colônia e Extração da ROI
Para cada amostra:
  1. Calibração radiométrica + corte de bandas ruidosas (via 1.1)
  2. RGB sintético com bandas fixas (180, 120, 70) ajustadas ao corte
  3. Detecção circular via Transformada de Hough — seleciona o círculo
     mais próximo do centro da imagem
  4. Máscara ROI recuada 20 px; fallback full-image se Hough falhar
  5. Visualização antes/depois por amostra (banda 128)
  6. DataFrame final com pixels do miolo biológico de todas as amostras
"""

import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

DATA_DIR  = Path("data")
BAND_VIZ  = 128  # banda usada nas visualizações — posição central do espectro útil
SHRINK_PX = 20   # pixels a recuar do raio detectado para evitar a borda da colônia

# Bandas do espectro original (indexação 1-based do sensor) usadas para
# compor a imagem RGB sintética. Valores fixos garantem que todas as amostras
# usem exatamente as mesmas regiões espectrais para a detecção, independente
# do tamanho de cada imagem. Serão ajustadas pelo corte de N_BANDAS_INICIO.
BANDAS_RGB_ORIG = (180, 120, 70)

# Parâmetros da Transformada de Hough Circular (CHT).
# dp=1.4   → resolução do acumulador (menor = mais preciso, mais lento)
# min_dist → distância mínima entre centros; valor pequeno (59) permite que o
#            algoritmo encontre múltiplos candidatos para depois escolhermos o melhor
# param1   → limiar alto do Canny interno (170 = detecta apenas bordas fortes,
#            reduz falsos positivos de scan lines)
# param2   → limiar do acumulador (quanto maior, mais exigente; 30 é um balanço
#            entre sensibilidade e especificidade para colônias em placa)
CHT = dict(
    dp=1.4,
    min_dist=59,
    param1=170,
    param2=30,
)

# ---------------------------------------------------------------------------
# Importa funções do 1.1
# O nome "1.1_preprocessamento.py" começa com número, então não pode ser
# importado com `import` padrão — usamos importlib para carregar pelo caminho.
# ---------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location(
    "preprocessamento", Path(__file__).parent / "1.1_preprocessamento.py"
)
_pre = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pre)

_load_raw               = _pre._load_raw
calibracao_radiometrica = _pre.calibracao_radiometrica
corte_bandas_ruidosas   = _pre.corte_bandas_ruidosas
N_BANDAS_INICIO         = _pre.N_BANDAS_INICIO  # necessário para ajustar índices RGB


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def _normalizar_uint8(arr: np.ndarray) -> np.ndarray:
    """Normaliza qualquer array para uint8 [0, 255] via min-max global.

    Normalização global (min/max de todo o array, incluindo os 3 canais juntos)
    preserva a relação de intensidade entre canais — importante para que a
    conversão para escala de cinza posterior reflita corretamente as diferenças
    espectrais.
    """
    arr = arr.astype(np.float32)
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin < 1e-8:
        # imagem completamente uniforme — evita divisão por zero
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - vmin) / (vmax - vmin) * 255).astype(np.uint8)


def rgb_sintetico(imagem: np.ndarray, bandas_rgb: tuple = BANDAS_RGB_ORIG) -> np.ndarray:
    """Gera uma imagem RGB uint8 a partir de três bandas NIR/SWIR do cubo.

    O sensor captura apenas NIR/SWIR (~909–2512 nm), invisível ao olho humano.
    Para guiar o algoritmo de detecção de círculos, criamos uma representação
    visual mapeando três comprimentos de onda para os canais R, G, B.

    Os índices são passados como 1-based (numeração original do sensor) e
    ajustados para 0-based pós-corte subtraindo N_BANDAS_INICIO.
    np.clip garante que nunca saiam dos limites do cubo cortado.
    """
    n = imagem.shape[2]
    idxs = [int(np.clip(b - 1 - N_BANDAS_INICIO, 0, n - 1)) for b in bandas_rgb]
    rgb = np.stack([imagem[:, :, i] for i in idxs], axis=2)
    return _normalizar_uint8(rgb)


# ---------------------------------------------------------------------------
# Detecção de círculo
# ---------------------------------------------------------------------------

def detectar_circulo_hough(imagem: np.ndarray) -> tuple | None:
    """Localiza o círculo da colônia bacteriana via Transformada de Hough Circular.

    Pipeline de pré-processamento antes do Hough:
      1. RGB sintético → escala de cinza
         Reduz o cubo 3D a uma única imagem 2D para o OpenCV processar.

      2. medianBlur(5)
         Remove o ruído impulsivo das scan lines horizontais do sensor Specim.
         O filtro mediano preserva bordas (como a borda da colônia) enquanto
         elimina padrões lineares, ao contrário do blur gaussiano que borraria
         a borda circular que queremos detectar.

      3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
         Equalização de histograma LOCAL em janelas de 8×8 px.
         Preferível ao equalizeHist global porque não amplifica as linhas de
         varredura na região de background — o global tende a saturar regiões
         uniformes e criar bordas falsas.

      4. GaussianBlur(7,7)
         Suavização final para reduzir ruído de alta frequência residual antes
         de alimentar o detector de bordas Canny interno do HoughCircles.

    Seleção do círculo:
      O OpenCV pode retornar múltiplos candidatos. Escolhemos o mais próximo
      do centro geométrico da imagem porque as colônias são sempre fotografadas
      centralizadas na placa. Tiebreak: raio maior (colônia maior = mais confiável).

    Retorna (cx, cy, raio) ou None se nenhum candidato for encontrado.
    """
    gray = cv2.cvtColor(rgb_sintetico(imagem), cv2.COLOR_RGB2GRAY)

    # passo 1: supprime scan lines horizontais sem borrar a borda da colônia
    gray = cv2.medianBlur(gray, 5)

    # passo 2: equalização local — realça a borda colônia/ágar sem amplificar ruído
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # passo 3: suavização final antes do Canny interno do HoughCircles
    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)

    h, w = gray.shape

    # limites de raio baseados no tamanho da imagem:
    # colônias ocupam entre 20% e 48% da menor dimensão da imagem
    min_r = max(2, int(min(h, w) * 0.20))
    max_r = max(min_r + 1, int(min(h, w) * 0.48))

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=CHT["dp"],
        minDist=CHT["min_dist"],
        param1=CHT["param1"],
        param2=CHT["param2"],
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is None:
        return None

    candidatos = np.round(circles[0]).astype(int)
    cx_img, cy_img = w / 2.0, h / 2.0

    # seleciona o candidato mais próximo do centro; em caso de empate, o maior raio
    candidatos = sorted(
        candidatos,
        key=lambda c: ((c[0] - cx_img) ** 2 + (c[1] - cy_img) ** 2, -c[2]),
    )
    x, y, r = candidatos[0]

    # np.clip garante que o centro detectado não caia fora dos limites da imagem
    return int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1)), int(max(1, r))


# ---------------------------------------------------------------------------
# Máscara ROI
# ---------------------------------------------------------------------------

def criar_mascara_roi(shape: tuple, cx: int, cy: int, raio: int,
                      shrink_px: int = SHRINK_PX) -> np.ndarray:
    """Cria máscara booleana do miolo biológico puro da colônia.

    Por que recuar o raio em 20 px (SHRINK_PX)?
    A borda da colônia contém:
      - pixels mistos (ágar + bactéria) que não representam o espectro puro
      - efeitos ópticos de borda (reflexão, difração) do próprio limite físico
    Recuando 20 px garantimos que apenas o miolo central — com espectro
    homogêneo e representativo da bactéria — entre no treinamento.

    Usa np.ogrid para criar matrizes de coordenadas eficientes e calcular
    a distância euclidiana ao quadrado de cada pixel ao centro detectado.
    """
    h, w = shape
    raio_roi = max(raio - shrink_px, 1)  # nunca deixa o raio virar zero
    Y, X = np.ogrid[:h, :w]
    return (X - cx) ** 2 + (Y - cy) ** 2 <= raio_roi ** 2


# ---------------------------------------------------------------------------
# Visualização
# ---------------------------------------------------------------------------

def plotar_antes_depois(imagem: np.ndarray, mascara: np.ndarray,
                        circulo: tuple | None, nome: str):
    """Exibe lado a lado a banda BAND_VIZ antes e depois da máscara ROI.

    Painel esquerdo (Antes):
      - imagem completa com o círculo externo da colônia (branco tracejado)
      - círculo da ROI recuada (verde sólido)
      Permite verificar visualmente se o Hough acertou a posição.

    Painel direito (Depois):
      - apenas os pixels dentro da ROI; fora = NaN renderizado como preto
      - cmap.set_bad("black") faz o matplotlib pintar NaN de preto,
        distinguindo "sem dado" de valores próximos a zero (que seriam azul no jet)
    """
    banda      = imagem[:, :, BAND_VIZ]
    vmin, vmax = np.percentile(banda, [2, 98])  # percentis para evitar outliers no contraste

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # --- painel esquerdo: imagem original com círculos sobrepostos ---
    ax1.imshow(banda, cmap="jet", vmin=vmin, vmax=vmax)
    if circulo is not None:
        cx, cy, r = circulo
        raio_roi = max(r - SHRINK_PX, 1)
        ax1.add_patch(plt.Circle((cx, cy), r,        color="white", fill=False, lw=1.5, ls="--"))
        ax1.add_patch(plt.Circle((cx, cy), raio_roi, color="lime",  fill=False, lw=1.5))
    ax1.set_title(f"Antes  |  banda {BAND_VIZ}\nbranco=colônia  verde=ROI")
    ax1.axis("off")

    # --- painel direito: miolo isolado ---
    # np.full_like com NaN cria fundo transparente/preto fora da ROI
    miolo = np.full_like(banda, np.nan)
    miolo[mascara] = banda[mascara]
    cmap = plt.get_cmap("jet").copy()
    cmap.set_bad("black")
    ax2.imshow(miolo, cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.set_title(f"Depois  |  miolo ROI\n{mascara.sum():,} pixels")
    ax2.axis("off")

    plt.suptitle(nome, fontsize=11)
    plt.tight_layout()
    plt.show()


def plotar_mosaico(imagens: dict, titulo: str, cols: int = 4):
    """Exibe um grid resumo com todas as amostras em uma única figura.

    Recebe um dicionário {nome: imagem_2d_float} onde NaN representa ausência
    de dado (fora da ROI). Útil para comparar visualmente todas as detecções
    de uma vez após o loop principal.

    -(-n // cols) é o equivalente a math.ceil(n / cols) sem importar math.
    """
    n    = len(imagens)
    rows = -(-n // cols)
    cmap = plt.get_cmap("jet").copy()
    cmap.set_bad("black")

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.atleast_1d(axes).flatten()

    for ax, (nome, img) in zip(axes, imagens.items()):
        # masked_invalid converte NaN/inf em valores mascarados que o matplotlib
        # renderiza com a cor set_bad — mantém consistência com plotar_antes_depois
        ax.imshow(np.ma.masked_invalid(img), cmap=cmap, vmin=0, vmax=1)
        ax.set_title(nome[:18], fontsize=8)
        ax.axis("off")

    # oculta os subplots vazios que sobram quando n não é múltiplo de cols
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(titulo, fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Pipeline por amostra
# ---------------------------------------------------------------------------

def processar_amostra_roi(sample_dir: Path, visualizar: bool = True
                          ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Executa o pipeline completo de uma amostra e retorna dados + imagens de viz.

    Fluxo:
      _load_raw × 3 → calibracao_radiometrica → corte_bandas_ruidosas
      → detectar_circulo_hough → criar_mascara_roi → extração de pixels

    Retorna:
      df           : DataFrame (n_pixels_roi, n_bandas+1) com coluna 'sample'
      img_demarcada: banda BAND_VIZ normalizada [0,1] com círculo desenhado
      img_recortada: mesma banda com NaN fora da ROI

    Fallback: se o Hough não detectar nenhum círculo, usa máscara full-image
    para não perder a amostra — o aviso indica que a detecção falhou.
    """
    name = sample_dir.name
    cap  = sample_dir / "capture"

    # carrega imagem principal e as duas referências de calibração
    imagem = _load_raw(cap / f"{name}.hdr")
    dark   = _load_raw(cap / f"DARKREF_{name}.hdr")
    white  = _load_raw(cap / f"WHITEREF_{name}.hdr")

    calibrada = calibracao_radiometrica(imagem, dark, white)
    cortada   = corte_bandas_ruidosas(calibrada)

    circulo = detectar_circulo_hough(cortada)

    if circulo is None:
        # fallback: sem detecção, inclui todos os pixels para não descartar a amostra
        print(f"  [AVISO] {name}: círculo não detectado — usando imagem completa.")
        mascara = np.ones(cortada.shape[:2], dtype=bool)
    else:
        cx, cy, raio = circulo
        mascara = criar_mascara_roi(cortada.shape[:2], cx, cy, raio)

    if visualizar:
        plotar_antes_depois(cortada, mascara, circulo, name)

    # --- prepara imagens 2D para os mosaicos finais ---
    banda = cortada[:, :, BAND_VIZ].astype(np.float32)
    vmin, vmax = np.percentile(banda, [2, 98])
    # normaliza para [0,1] usando os mesmos percentis da visualização individual
    banda_norm = np.clip((banda - vmin) / (vmax - vmin + 1e-9), 0, 1)

    # demarcada: desenha o círculo com cv2 (pixel-accurate) sobre a banda normalizada
    img_demarcada = banda_norm.copy()
    if circulo is not None:
        img_demarcada_u8 = (banda_norm * 255).astype(np.uint8)
        cx, cy, r = circulo
        cv2.circle(img_demarcada_u8, (cx, cy), r, 255, thickness=2)
        img_demarcada = img_demarcada_u8.astype(np.float32) / 255.0

    # recortada: pixels fora da ROI viram NaN para serem pintados de preto no mosaico
    img_recortada = np.where(mascara, banda_norm, np.nan)

    # --- extrai pixels da ROI para o DataFrame ---
    # cortada[mascara] seleciona apenas as linhas onde mascara=True
    # resultado: (n_pixels_roi, n_bandas) — cada pixel é uma linha
    pixels  = cortada[mascara]
    colunas = [f"band_{i}" for i in range(pixels.shape[1])]
    df = pd.DataFrame(pixels, columns=colunas)
    df.insert(0, "sample", name)

    # também retorna a máscara booleana — necessária no 2.3 para reconstruir
    # os mapas espaciais de clusters (projetar labels de volta na imagem 2D)
    return df, img_demarcada, img_recortada, mascara


# ---------------------------------------------------------------------------
# Construção do DataFrame final
# ---------------------------------------------------------------------------

def construir_dataframe_roi(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Itera todas as amostras, coleta pixels da ROI e exibe mosaicos resumo.

    Acumula os DataFrames em lista e usa pd.concat no final — mais eficiente
    que concatenar incrementalmente pois evita cópias intermediárias.
    Também acumula as imagens 2D para exibir dois mosaicos ao final:
      1. todas as colônias com o círculo demarcado (diagnóstico da detecção)
      2. todas as colônias já recortadas pela ROI (resultado final)
    As máscaras booleanas são salvas em masks.pkl para uso no 2.3.
    """
    sample_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    partes     = []
    demarcadas = {}
    recortadas = {}
    masks      = {}  # {nome: ndarray bool (H, W)} — persistido para o 2.3

    for sd in sample_dirs:
        print(f"  {sd.name} ...", end=" ", flush=True)
        df_roi, img_dem, img_rec, mascara = processar_amostra_roi(sd, visualizar=True)
        partes.append(df_roi)
        demarcadas[sd.name] = img_dem
        recortadas[sd.name] = img_rec
        masks[sd.name]      = mascara
        print(f"{len(df_roi):,} pixels")

    # mosaicos finais — visão geral de todas as amostras de uma vez
    plotar_mosaico(demarcadas, "Mosaico — círculo demarcado (CHT)")
    plotar_mosaico(recortadas, "Mosaico — imagens recortadas pela ROI")

    return pd.concat(partes, ignore_index=True), masks


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------

PICKLE_PATH = DATA_DIR / "roi.pkl"
MASKS_PATH  = DATA_DIR / "masks.pkl"

if __name__ == "__main__":
    print(f"Detectando colônias em '{DATA_DIR}' ...\n")
    df, masks = construir_dataframe_roi()

    print(f"\nDataFrame ROI final:")
    print(f"  {len(df):,} pixels  |  {df['sample'].nunique()} amostras  |  {df.shape[1]-1} bandas")
    print(df.groupby("sample").size().rename("pixels_roi").to_string())

    # pickle preserva os dtypes float32 exatos sem conversão e não sofre do
    # problema de duplo-registro do pyarrow quando executado via %run no Jupyter
    df.to_pickle(PICKLE_PATH)
    print(f"Salvo em '{PICKLE_PATH}'")

    # salva as máscaras ROI de cada amostra para uso no 2.3
    # (necessárias para reprojetar os labels K-Means de volta ao espaço 2D)
    import pickle
    with open(MASKS_PATH, "wb") as f:
        pickle.dump(masks, f)
    print(f"Salvo em '{MASKS_PATH}'")
