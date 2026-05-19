"""
2.2 - Redução Dimensional: PCA de Treino + Visualizações
Recebe os dados normalizados (dados_normalizados.pkl) e executa:
  - PCA com 95% de variância explicada → espaço reduzido para o K-Means
  - PCA 2D e t-SNE 2D coloridos por rótulo → diagnóstico visual da separabilidade

Entrega: data/pca_model.pkl, data/Z_pca.pkl
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR    = Path("data")
ENTRADA     = DATA_DIR / "dados_normalizados.pkl"
PCA_MODEL   = DATA_DIR / "pca_model.pkl"
Z_PCA_PATH  = DATA_DIR / "Z_pca.pkl"

# Variância acumulada que o PCA deve preservar.
# 0.95 = mantém as componentes que juntas explicam 95% da variância total.
# Reduz ~206 bandas para tipicamente 10-30 componentes, eliminando
# redundância espectral (bandas adjacentes são altamente correlacionadas).
PCA_VARIANCIA    = 0.95

# Máximos de amostras para as visualizações 2D.
# PCA e t-SNE 2D são usados apenas para diagnóstico visual — não precisamos
# de todos os milhões de pixels para ver a estrutura do espaço espectral.
PCA_MAX_AMOSTRAS  = 10_000
TSNE_MAX_AMOSTRAS = 5_000

TSNE_PERPLEXITY = 30.0   # controla o "raio de vizinhança" no t-SNE
TSNE_MAX_ITER   = 1_000  # iterações do gradiente; 1000 é suficiente para convergir
RANDOM_STATE    = 42     # semente para reprodutibilidade


# ---------------------------------------------------------------------------
# Funções
# ---------------------------------------------------------------------------

def executar_pca_treino(X: np.ndarray, variancia: float = PCA_VARIANCIA
                        ) -> tuple:
    """Treina o PCA mantendo `variancia` de variância explicada acumulada.

    Por que PCA antes do K-Means?
    1. Remove redundância: bandas adjacentes são altamente correlacionadas —
       o PCA as colapsa em componentes ortogonais independentes.
    2. Melhora o K-Means: distâncias euclidianas em alta dimensão perdem
       significado (curse of dimensionality). No espaço PCA, os clusters
       são mais compactos e separáveis.
    3. Velocidade: K-Means em 15 componentes vs. 206 bandas é muito mais rápido.

    Retorna (modelo PCA ajustado, dados transformados Z).
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=float(variancia), random_state=RANDOM_STATE)
    Z   = pca.fit_transform(X)

    var_acum = pca.explained_variance_ratio_.cumsum()[-1]
    print(f"  PCA: {X.shape[1]} bandas → {Z.shape[1]} componentes "
          f"({var_acum:.1%} de variância explicada)")
    return pca, Z


def _subamostrar(X: np.ndarray, meta: pd.DataFrame, n_max: int
                 ) -> tuple[np.ndarray, pd.DataFrame]:
    """Seleciona aleatoriamente até n_max linhas de X e meta (indices alinhados)."""
    n = min(X.shape[0], n_max)
    idx = np.sort(
        np.random.default_rng(RANDOM_STATE).choice(X.shape[0], size=n, replace=False)
    )
    return X[idx], meta.iloc[idx].reset_index(drop=True)


def plotar_projecao(Z: np.ndarray, meta: pd.DataFrame, titulo: str):
    """Scatter plot 2D com pontos coloridos por rótulo (resistente/sensivel).

    Cada ponto é um pixel. A separação visual entre classes indica se o
    espaço espectral reduzido contém informação discriminativa suficiente
    para o clustering.
    """
    rotulos_unicos = sorted(meta["rotulo"].unique())
    cores = plt.get_cmap("tab10")(np.linspace(0, 1, max(len(rotulos_unicos), 2)))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, rot in enumerate(rotulos_unicos):
        mask = meta["rotulo"] == rot
        ax.scatter(Z[mask, 0], Z[mask, 1],
                   c=[cores[i]], s=3, alpha=0.5, linewidths=0, label=rot)

    ax.set_title(titulo)
    ax.legend(markerscale=4, fontsize=9)
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    plt.tight_layout()
    plt.show()


def executar_pca_2d(X: np.ndarray, meta: pd.DataFrame):
    """PCA reduzido a 2 componentes para visualização.

    Diferente do PCA de treino (que mantém 95% de variância), este usa
    exatamente 2 componentes para permitir o scatter plot 2D.
    Subamostrado para no máximo PCA_MAX_AMOSTRAS pixels.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X_sub, meta_sub = _subamostrar(X, meta, PCA_MAX_AMOSTRAS)

    # padronização (zero mean, unit variance) antes do PCA 2D —
    # garante que bandas com maior amplitude não dominem os eixos visuais
    X_scaled = StandardScaler().fit_transform(X_sub)

    Z = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_scaled)
    plotar_projecao(Z, meta_sub, f"PCA 2D — {len(X_sub):,} pixels")


def executar_tsne_2d(X: np.ndarray, meta: pd.DataFrame):
    """t-SNE reduzido a 2 componentes para visualização.

    Por que t-SNE além do PCA?
    O PCA é linear — só captura separações lineares. O t-SNE é não-linear
    e revela estruturas de cluster que o PCA não consegue mostrar.
    Se o t-SNE mostrar clusters claros por rótulo, o K-Means provavelmente
    conseguirá separar resistentes de sensíveis mesmo no espaço PCA.

    init="pca" inicializa o t-SNE com o PCA — converge mais rápido e
    produz resultados mais estáveis do que a inicialização aleatória.

    Subamostrado para TSNE_MAX_AMOSTRAS por custo computacional O(n²).
    """
    from sklearn.manifold import TSNE

    X_sub, meta_sub = _subamostrar(X, meta, TSNE_MAX_AMOSTRAS)

    # perplexidade não pode ser maior que n_amostras - 1
    perp = min(TSNE_PERPLEXITY, float(X_sub.shape[0] - 1))

    Z = TSNE(
        n_components=2,
        perplexity=perp,
        max_iter=TSNE_MAX_ITER,
        random_state=RANDOM_STATE,
        init="pca",        # inicialização PCA para estabilidade
    ).fit_transform(X_sub)

    plotar_projecao(Z, meta_sub, f"t-SNE 2D — {len(X_sub):,} pixels")


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # --- 1. Carrega dados normalizados ---
    print(f"Carregando '{ENTRADA}' ...")
    df = pd.read_pickle(ENTRADA)

    # separa features espectrais das colunas de metadados
    colunas_banda = [c for c in df.columns if c.startswith("band_")]
    X    = df[colunas_banda].values.astype(np.float32)  # (n_pixels, n_bandas)
    meta = df[["sample", "rotulo"]].copy()              # metadados alinhados

    print(f"  {X.shape[0]:,} pixels  |  {X.shape[1]} bandas")

    # --- 2. PCA de treino (95% variância) ---
    # este modelo será usado pelo 2.3 para transformar os dados antes do K-Means
    print("\nTreinando PCA (95% variância) ...")
    pca_model, Z_treino = executar_pca_treino(X)

    # --- 3. Salva modelo PCA e dados transformados ---
    # pca_model.pkl: objeto sklearn para reusar no 2.3 sem retreinar
    with open(PCA_MODEL, "wb") as f:
        pickle.dump(pca_model, f)
    print(f"Salvo em '{PCA_MODEL}'")

    # Z_pca.pkl: DataFrame com metadados + componentes PCA
    # mantemos sample e rotulo alinhados para facilitar o 2.3 e 2.4
    colunas_pca = [f"pca_{i}" for i in range(Z_treino.shape[1])]
    df_Z = pd.concat([
        meta.reset_index(drop=True),
        pd.DataFrame(Z_treino, columns=colunas_pca)
    ], axis=1)
    df_Z.to_pickle(Z_PCA_PATH)
    print(f"Salvo em '{Z_PCA_PATH}'")

    # --- 4. Visualizações 2D (diagnóstico) ---
    print("\nGerando PCA 2D ...")
    executar_pca_2d(X, meta)

    print("Gerando t-SNE 2D ...")
    executar_tsne_2d(X, meta)

    print("\n[2.2] Concluído.")
