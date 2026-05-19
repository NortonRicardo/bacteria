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
BAND_VIZ  = 128   # banda usada na visualização antes/depois
SHRINK_PX = 20    # pixels a recuar do raio detectado

# Bandas RGB no espectro original (1-based), ajustadas após o corte
BANDAS_RGB_ORIG = (180, 120, 70)

CHT = dict(
    dp=1.4,
    min_dist=59,
    param1=170,
    param2=30,
)

# ---------------------------------------------------------------------------
# Importa funções do 1.1 (nome de arquivo começa com número)
# ---------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location(
    "preprocessamento", Path(__file__).parent / "1.1_preprocessamento.py"
)
_pre = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pre)

_load_raw               = _pre._load_raw
calibracao_radiometrica = _pre.calibracao_radiometrica
corte_bandas_ruidosas   = _pre.corte_bandas_ruidosas
N_BANDAS_INICIO         = _pre.N_BANDAS_INICIO


# ---------------------------------------------------------------------------
# Funções
# ---------------------------------------------------------------------------

def _normalizar_uint8(arr: np.ndarray) -> np.ndarray:
    """Normaliza array para uint8 [0, 255]."""
    arr = arr.astype(np.float32)
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - vmin) / (vmax - vmin) * 255).astype(np.uint8)


def rgb_sintetico(imagem: np.ndarray, bandas_rgb: tuple = BANDAS_RGB_ORIG) -> np.ndarray:
    """Gera RGB uint8 usando bandas fixas ajustadas ao corte de N_BANDAS_INICIO."""
    n = imagem.shape[2]
    # converte 1-based original → índice 0-based pós-corte
    idxs = [int(np.clip(b - 1 - N_BANDAS_INICIO, 0, n - 1)) for b in bandas_rgb]
    rgb = np.stack([imagem[:, :, i] for i in idxs], axis=2)
    return _normalizar_uint8(rgb)


def detectar_circulo_hough(imagem: np.ndarray) -> tuple | None:
    """Detecta o círculo da colônia via Transformada de Hough.

    Entre os candidatos retornados pelo OpenCV, seleciona o mais próximo
    do centro geométrico da imagem (tiebreak: maior raio).

    Retorna (cx, cy, raio) ou None se nenhum candidato encontrado.
    """
    gray = cv2.cvtColor(rgb_sintetico(imagem), cv2.COLOR_RGB2GRAY)
    # median blur suprime scan lines horizontais antes da equalização
    gray = cv2.medianBlur(gray, 5)
    # CLAHE: equalização local — não amplifica as linhas de varredura
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)

    h, w = gray.shape
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

    # ordena por distância ao centro (↑ próximo) e desempata por raio (↑ maior)
    candidatos = sorted(
        candidatos,
        key=lambda c: ((c[0] - cx_img) ** 2 + (c[1] - cy_img) ** 2, -c[2]),
    )
    x, y, r = candidatos[0]
    return int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1)), int(max(1, r))


def criar_mascara_roi(shape: tuple, cx: int, cy: int, raio: int,
                      shrink_px: int = SHRINK_PX) -> np.ndarray:
    """Máscara booleana do miolo biológico (raio recuado em shrink_px)."""
    h, w = shape
    raio_roi = max(raio - shrink_px, 1)
    Y, X = np.ogrid[:h, :w]
    return (X - cx) ** 2 + (Y - cy) ** 2 <= raio_roi ** 2


def plotar_antes_depois(imagem: np.ndarray, mascara: np.ndarray,
                        circulo: tuple | None, nome: str):
    """Exibe lado a lado: imagem com círculo demarcado e miolo isolado."""
    banda        = imagem[:, :, BAND_VIZ]
    vmin, vmax   = np.percentile(banda, [2, 98])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # --- Antes ---
    ax1.imshow(banda, cmap="jet", vmin=vmin, vmax=vmax)
    if circulo is not None:
        cx, cy, r = circulo
        raio_roi = max(r - SHRINK_PX, 1)
        ax1.add_patch(plt.Circle((cx, cy), r,        color="white", fill=False, lw=1.5, ls="--"))
        ax1.add_patch(plt.Circle((cx, cy), raio_roi, color="lime",  fill=False, lw=1.5))
    ax1.set_title(f"Antes  |  banda {BAND_VIZ}\nbranco=colônia  verde=ROI")
    ax1.axis("off")

    # --- Depois ---
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


def processar_amostra_roi(sample_dir: Path, visualizar: bool = True) -> pd.DataFrame:
    """Pipeline completo de uma amostra: calibra → corta → detecta → mascara.

    Retorna DataFrame (n_pixels_roi × n_bandas + coluna 'sample').
    Se o Hough falhar, usa máscara full-image como fallback.
    """
    name = sample_dir.name
    cap  = sample_dir / "capture"

    imagem = _load_raw(cap / f"{name}.hdr")
    dark   = _load_raw(cap / f"DARKREF_{name}.hdr")
    white  = _load_raw(cap / f"WHITEREF_{name}.hdr")

    calibrada = calibracao_radiometrica(imagem, dark, white)
    cortada   = corte_bandas_ruidosas(calibrada)

    circulo = detectar_circulo_hough(cortada)

    if circulo is None:
        print(f"  [AVISO] {name}: círculo não detectado — usando imagem completa.")
        mascara = np.ones(cortada.shape[:2], dtype=bool)
    else:
        cx, cy, raio = circulo
        mascara = criar_mascara_roi(cortada.shape[:2], cx, cy, raio)

    if visualizar:
        plotar_antes_depois(cortada, mascara, circulo, name)

    pixels  = cortada[mascara]
    colunas = [f"band_{i}" for i in range(pixels.shape[1])]
    df = pd.DataFrame(pixels, columns=colunas)
    df.insert(0, "sample", name)
    return df


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------

def construir_dataframe_roi(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    sample_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    partes = []

    for sd in sample_dirs:
        print(f"  {sd.name} ...", end=" ", flush=True)
        df_roi = processar_amostra_roi(sd, visualizar=True)
        partes.append(df_roi)
        print(f"{len(df_roi):,} pixels")

    return pd.concat(partes, ignore_index=True)


if __name__ == "__main__":
    print(f"Detectando colônias em '{DATA_DIR}' ...\n")
    df = construir_dataframe_roi()

    print(f"\nDataFrame ROI final:")
    print(f"  {len(df):,} pixels  |  {df['sample'].nunique()} amostras  |  {df.shape[1]-1} bandas")
    print(df.groupby("sample").size().rename("pixels_roi").to_string())
