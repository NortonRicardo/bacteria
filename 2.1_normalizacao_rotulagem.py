"""
2.1 - Normalização Global e Rotulagem
Recebe o DataFrame bruto de pixels da ROI (roi.pkl) gerado pelo 1.2,
aplica normalização Min-Max global por banda e adiciona os rótulos
(resistente / sensivel) de cada amostra.

Entrega: data/dados_normalizados.pkl
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR      = Path("data")
ROI_PATH      = DATA_DIR / "roi.pkl"
ROTULOS_PATH  = DATA_DIR / "rotulos.csv"
SAIDA_PATH    = DATA_DIR / "dados_normalizados.pkl"
PARAMS_PATH   = DATA_DIR / "normalizacao_params.pkl"

# Rótulo padrão para amostras que não estejam no rotulos.csv.
# Evita que o pipeline quebre se uma amostra nova for adicionada
# sem ter sido catalogada ainda.
ROTULO_FALLBACK = "sensivel"


# ---------------------------------------------------------------------------
# Funções
# ---------------------------------------------------------------------------

def carregar_rotulos(caminho: Path, fallback: str = ROTULO_FALLBACK) -> dict:
    """Lê o CSV de rótulos e retorna dicionário {nome_amostra: rotulo}.

    O CSV deve ter colunas 'nome' e 'rotulo'.
    O 'nome' deve ser idêntico ao nome da pasta da amostra em data/.

    Por que arquivo externo e não hardcoded?
    Os rótulos são conhecimento de domínio (biológico) externo ao código —
    separar em CSV permite atualizar sem tocar no pipeline.
    """
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo de rótulos não encontrado: '{caminho}'\n"
            "Crie um CSV com colunas 'nome' e 'rotulo'."
        )
    df = pd.read_csv(caminho)
    return {str(row["nome"]).strip(): str(row["rotulo"]).strip()
            for _, row in df.iterrows()}


def normalizar_minmax_global(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aplica normalização Min-Max global por banda a todas as amostras.

    Por que global (e não por amostra)?
    A normalização por amostra destruiria as diferenças de reflectância
    absoluta entre bactérias resistentes e sensíveis — que é exatamente
    o sinal que o K-Means precisa capturar.
    Ao usar o min/max do dataset inteiro, cada banda é escalada para [0, 1]
    preservando as relações relativas entre amostras.

    Por que por banda e não global único?
    Cada banda tem uma faixa de reflectância própria (algumas bandas são
    intrinsecamente mais intensas). Normalizar por banda garante que nenhuma
    banda domine o PCA simplesmente por ter valores maiores.

    Retorna o DataFrame normalizado e os parâmetros (min/max por banda)
    para que possam ser aplicados a novos dados no futuro.
    """
    colunas_banda = [c for c in df.columns if c.startswith("band_")]

    # extrai apenas as colunas espectrais como matriz numpy
    X = df[colunas_banda].values.astype(np.float32)

    # min e max calculados sobre todos os pixels de todas as amostras
    # shape: (n_bandas,) — um valor por banda
    min_global = X.min(axis=0)
    max_global = X.max(axis=0)

    # proteção contra banda completamente uniforme (divisão por zero)
    den = np.where(max_global - min_global == 0, 1e-8, max_global - min_global)

    X_norm = np.clip((X - min_global) / den, 0.0, 1.0)

    # substitui os valores no DataFrame preservando as colunas de metadados
    df_norm = df.copy()
    df_norm[colunas_banda] = X_norm

    params = {"min": min_global, "max": max_global}
    return df_norm, params


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # --- 1. Carrega o DataFrame de pixels da ROI ---
    # roi.pkl contém colunas: sample, band_0 … band_N (reflectância calibrada)
    print(f"Carregando '{ROI_PATH}' ...")
    df = pd.read_pickle(ROI_PATH)
    print(f"  {len(df):,} pixels  |  {df['sample'].nunique()} amostras  |  {df.shape[1]-1} bandas")

    # --- 2. Adiciona coluna de rótulo ---
    # map() aplica o dicionário de rótulos coluna a coluna;
    # fillna garante que amostras sem rótulo recebam o fallback
    print(f"\nCarregando rótulos de '{ROTULOS_PATH}' ...")
    rotulos_map = carregar_rotulos(ROTULOS_PATH)
    df["rotulo"] = df["sample"].map(rotulos_map).fillna(ROTULO_FALLBACK)

    # diagnóstico: quantos pixels por amostra e por rótulo
    resumo = df.groupby(["sample", "rotulo"]).size().rename("pixels")
    print("\nDistribuição de pixels por amostra:")
    print(resumo.to_string())

    # --- 3. Normalização Min-Max global ---
    print("\nAplicando normalização Min-Max global por banda ...")
    df_norm, params = normalizar_minmax_global(df)
    print(f"  range após normalização: [{df_norm.filter(like='band_').values.min():.4f}, "
          f"{df_norm.filter(like='band_').values.max():.4f}]")

    # --- 4. Salva resultados ---
    df_norm.to_pickle(SAIDA_PATH)
    print(f"\nSalvo em '{SAIDA_PATH}'")

    # salva os parâmetros de normalização separadamente para uso futuro
    # (ex.: aplicar a novos dados sem recalcular o min/max)
    with open(PARAMS_PATH, "wb") as f:
        pickle.dump(params, f)
    print(f"Salvo em '{PARAMS_PATH}'")
