"""
1.1 - Pré-processamento das imagens hiperespectrais
Carrega todas as 19 amostras de data/, aplica calibração radiométrica
e corte de bandas ruidosas, e consolida tudo em um único DataFrame.
"""

import numpy as np
import pandas as pd
import spectral.io.envi as envi  # biblioteca dedicada ao formato ENVI (hdr+raw)
from pathlib import Path

DATA_DIR = Path("data")

# As pontas do espectro do sensor Specim são sempre ruidosas (ruído eletrônico
# do detector nas frequências extremas). Removemos 25 bandas de cada lado para
# garantir que apenas a região espectral confiável entre no modelo.
N_BANDAS_INICIO = 25
N_BANDAS_FIM    = 25


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _load_raw(hdr_path: Path) -> np.ndarray:
    """Carrega um par .hdr/.raw no formato ENVI e retorna (lines, samples, bands).

    Usamos spectral.io.envi em vez de numpy puro porque a biblioteca lida
    automaticamente com o data_type, byte_order e interleave declarados no
    cabeçalho — evitando o bug do data_type=12 que os headers Specim reportam
    incorretamente.
    """
    raw_path = hdr_path.with_suffix('.raw')
    img = envi.open(str(hdr_path), image=str(raw_path))

    # img[:,:,:] carrega o cubo inteiro na memória como ndarray float32.
    # O spectral já aplica o reshape correto conforme o interleave (BIL/BSQ/BIP).
    return np.asarray(img[:, :, :], dtype=np.float32)


# ---------------------------------------------------------------------------
# Processamento
# ---------------------------------------------------------------------------

def calibracao_radiometrica(imagem: np.ndarray,
                             dark: np.ndarray,
                             white: np.ndarray) -> np.ndarray:
    """Converte a imagem bruta em reflectância relativa.

    O scanner captura em contagens digitais brutas (DN), que variam conforme
    a intensidade da iluminação no dia da coleta. Para comparar amostras de
    dias diferentes, precisamos normalizar para reflectância [0, 1].

    Fórmula:
        Reflectância = (imagem - média_escura) / (média_clara - média_escura)

    - média_escura: média da DarkRef  → representa o ruído de leitura do sensor
      (câmera tampada, sem luz). Subtrai o offset eletrônico.
    - média_clara:  média da WhiteRef → representa a reflectância máxima do
      painel branco de referência. Normaliza a variação de intensidade da fonte.

    Calculamos a média ao longo dos eixos espaciais (axis=0, keepdims) para
    obter um vetor (1, samples, bands) que faz broadcasting com a imagem inteira.
    """
    # média por coluna e banda — mantém keepdims para broadcast automático
    media_escura = dark.mean(axis=0, keepdims=True)   # (1, samples, bands)
    media_clara  = white.mean(axis=0, keepdims=True)  # (1, samples, bands)

    # proteção contra divisão por zero: ocorre quando dark ≈ white (sensor saturado)
    denominador = media_clara - media_escura
    denominador = np.where(denominador == 0, 1e-9, denominador)

    reflectancia = (imagem - media_escura) / denominador

    # valores < 0 (imagem mais escura que o dark) ou > 1 (saturação) são
    # artefatos físicos — clipamos para manter o intervalo válido de reflectância
    return np.clip(reflectancia, 0.0, 1.0)


def corte_bandas_ruidosas(imagem: np.ndarray,
                          n_inicio: int = N_BANDAS_INICIO,
                          n_fim: int = N_BANDAS_FIM) -> np.ndarray:
    """Remove as bandas espectrais ruidosas das extremidades do cubo.

    Sensores hiperespectrais NIR/SWIR têm resposta confiável apenas na faixa
    central do seu range. As primeiras e últimas bandas sofrem de:
      - ruído shot elevado (baixa resposta do detector nas bordas do range)
      - artefatos de calibração do sensor
    Removendo 25 de cada lado de 256 bandas totais, ficamos com 206 bandas
    limpas (~975 nm a ~2390 nm).
    """
    return imagem[:, :, n_inicio: imagem.shape[2] - n_fim]


# ---------------------------------------------------------------------------
# Pipeline por amostra
# ---------------------------------------------------------------------------

def processar_amostra(sample_dir: Path) -> pd.DataFrame:
    """Executa o pipeline completo em uma amostra e retorna DataFrame de pixels.

    Cada linha do DataFrame = 1 pixel espacial da imagem.
    Colunas: 'sample' (nome da amostra) + 'band_0' … 'band_N' (reflectância).

    Estrutura orientada a pixels porque é o formato esperado por algoritmos
    de ML — cada pixel é uma observação independente com seu perfil espectral.
    """
    name = sample_dir.name
    cap  = sample_dir / 'capture'

    # carrega os três arquivos necessários para a calibração
    imagem = _load_raw(cap / f'{name}.hdr')           # imagem da amostra
    dark   = _load_raw(cap / f'DARKREF_{name}.hdr')   # referência escura
    white  = _load_raw(cap / f'WHITEREF_{name}.hdr')  # referência clara

    calibrada = calibracao_radiometrica(imagem, dark, white)
    cortada   = corte_bandas_ruidosas(calibrada)

    lines, samples, bands = cortada.shape

    # achata as dimensões espaciais (lines × samples) → uma linha por pixel.
    # O resultado é uma matriz 2D: (n_pixels, n_bandas)
    pixels = cortada.reshape(-1, bands)

    colunas = [f'band_{i}' for i in range(bands)]
    df = pd.DataFrame(pixels, columns=colunas)

    # coluna 'sample' identifica de qual amostra cada pixel veio —
    # essencial para rotular os dados no treinamento do modelo
    df.insert(0, 'sample', name)
    return df


# ---------------------------------------------------------------------------
# Construção do DataFrame final
# ---------------------------------------------------------------------------

def construir_dataframe(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Itera todas as amostras em data_dir e concatena em um único DataFrame.

    O pd.concat no final é mais eficiente do que ir appendando linha a linha,
    pois evita realocações de memória intermediárias.
    """
    sample_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    partes = []

    for sd in sample_dirs:
        print(f"  processando {sd.name} ...", end=' ')
        df_amostra = processar_amostra(sd)
        partes.append(df_amostra)
        print(f"{len(df_amostra):,} pixels")

    df_final = pd.concat(partes, ignore_index=True)
    return df_final


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print(f"Carregando amostras de '{DATA_DIR}' ...\n")
    df = construir_dataframe()

    n_amostras = df['sample'].nunique()
    n_bandas   = df.shape[1] - 1
    print(f"\nDataFrame final: {len(df):,} pixels | {n_amostras} amostras | {n_bandas} bandas")
    print(df.head())
