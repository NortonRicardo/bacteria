"""
1.1 - Pré-processamento das imagens hiperespectrais
Carrega todas as 19 amostras de data/, aplica calibração radiométrica
e corte de bandas ruidosas, e consolida tudo em um único DataFrame.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
N_BANDAS_INICIO = 25  # bandas a remover do início (ruído do sensor)
N_BANDAS_FIM    = 25  # bandas a remover do final  (ruído do sensor)


# ---------------------------------------------------------------------------
# Funções auxiliares de I/O
# ---------------------------------------------------------------------------

def _parse_hdr(hdr_path: Path) -> dict:
    """Lê um arquivo .hdr ENVI e retorna um dicionário de metadados."""
    import re
    meta = {}
    text = hdr_path.read_text()

    for key, value in re.findall(r'([\w ]+)\s*=\s*\{([^}]*)\}', text, re.DOTALL):
        items = [v.strip() for v in re.split(r'[\n,]', value) if v.strip()]
        try:
            meta[key.strip().lower()] = [float(i) for i in items]
        except ValueError:
            meta[key.strip().lower()] = items

    for line in text.splitlines():
        if '=' in line and '{' not in line:
            key, _, val = line.partition('=')
            key, val = key.strip().lower(), val.strip()
            if key in meta:
                continue
            try:
                meta[key] = int(val)
            except ValueError:
                try:
                    meta[key] = float(val)
                except ValueError:
                    meta[key] = val.lower()
    return meta


def _load_raw(hdr_path: Path) -> np.ndarray:
    """Carrega um arquivo .raw ENVI e retorna array (lines, samples, bands)."""
    raw_path = hdr_path.with_suffix('.raw')
    meta = _parse_hdr(hdr_path)

    lines   = int(meta['lines'])
    samples = int(meta['samples'])
    bands   = int(meta['bands'])

    data = np.fromfile(raw_path, dtype='<u2').reshape(lines, bands, samples)
    return np.transpose(data, (0, 2, 1)).astype(np.float32)  # (lines, samples, bands)


# ---------------------------------------------------------------------------
# Funções de processamento
# ---------------------------------------------------------------------------

def calibracao_radiometrica(imagem: np.ndarray,
                             dark: np.ndarray,
                             white: np.ndarray) -> np.ndarray:
    """Calcula a reflectância usando a fórmula de calibração radiométrica.

    Reflectância = (imagem - média_escura) / (média_clara - média_escura)

    Parâmetros
    ----------
    imagem : ndarray (lines, samples, bands)
    dark   : ndarray  – imagem de referência escura (DarkRef)
    white  : ndarray  – imagem de referência clara (WhiteRef)

    Retorna
    -------
    ndarray (lines, samples, bands) com valores em [0, 1]
    """
    media_escura = dark.mean(axis=0, keepdims=True)   # (1, samples, bands)
    media_clara  = white.mean(axis=0, keepdims=True)  # (1, samples, bands)

    denominador = media_clara - media_escura
    denominador = np.where(denominador == 0, 1e-9, denominador)  # evita divisão por zero

    reflectancia = (imagem - media_escura) / denominador
    return np.clip(reflectancia, 0.0, 1.0)


def corte_bandas_ruidosas(imagem: np.ndarray,
                          n_inicio: int = N_BANDAS_INICIO,
                          n_fim: int = N_BANDAS_FIM) -> np.ndarray:
    """Remove bandas ruidosas das pontas do cubo hiperespectral.

    Parâmetros
    ----------
    imagem   : ndarray (lines, samples, bands)
    n_inicio : bandas a remover do início
    n_fim    : bandas a remover do final

    Retorna
    -------
    ndarray (lines, samples, bands - n_inicio - n_fim)
    """
    return imagem[:, :, n_inicio: imagem.shape[2] - n_fim]


# ---------------------------------------------------------------------------
# Pipeline por amostra
# ---------------------------------------------------------------------------

def processar_amostra(sample_dir: Path) -> pd.DataFrame:
    """Processa uma amostra completa e retorna um DataFrame de pixels.

    Cada linha do DataFrame corresponde a um pixel espacial.
    Colunas: 'sample' + 'band_0' ... 'band_N'.
    """
    name = sample_dir.name
    cap  = sample_dir / 'capture'

    imagem = _load_raw(cap / f'{name}.hdr')
    dark   = _load_raw(cap / f'DARKREF_{name}.hdr')
    white  = _load_raw(cap / f'WHITEREF_{name}.hdr')

    calibrada = calibracao_radiometrica(imagem, dark, white)
    cortada   = corte_bandas_ruidosas(calibrada)

    lines, samples, bands = cortada.shape
    pixels = cortada.reshape(-1, bands)  # (lines*samples, bands)

    colunas = [f'band_{i}' for i in range(bands)]
    df = pd.DataFrame(pixels, columns=colunas)
    df.insert(0, 'sample', name)
    return df


# ---------------------------------------------------------------------------
# Construção do DataFrame final
# ---------------------------------------------------------------------------

def construir_dataframe(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Itera sobre todas as amostras em data_dir e consolida em um único DataFrame."""
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
