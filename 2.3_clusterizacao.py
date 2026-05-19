"""
2.3 - Clusterização K-Means Global
Recebe o espaço PCA (Z_pca.pkl), os dados normalizados (dados_normalizados.pkl)
e as máscaras espaciais (masks.pkl) para:
  - Treinar K-Means (k=2) no espaço PCA
  - Reprojetar os labels de cluster de volta às imagens 2D
  - Visualizar mosaicos de mapas de cluster
  - Calcular importância das bandas via variância dos centróides
  - Scatter plots PCA 2D e t-SNE 2D coloridos por cluster

Entrega: data/kmeans_model.pkl, data/labels_kmeans.pkl
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

DATA_DIR        = Path("data")
Z_PCA_PATH      = DATA_DIR / "Z_pca.pkl"
DADOS_NORM_PATH = DATA_DIR / "dados_normalizados.pkl"
MASKS_PATH      = DATA_DIR / "masks.pkl"
KMEANS_MODEL    = DATA_DIR / "kmeans_model.pkl"
LABELS_PATH     = DATA_DIR / "labels_kmeans.pkl"

K            = 2    # número de clusters — resistente e sensível
N_INIT       = 20   # reinicializações do K-Means; mais reinícios = menor chance
                    # de convergir para ótimo local ruim
RANDOM_STATE = 42

# Limites de amostragem para os scatter plots 2D (diagnóstico visual apenas)
PCA_MAX_AMOSTRAS  = 10_000
TSNE_MAX_AMOSTRAS = 5_000
TSNE_MAX_ITER     = 1_000


# ---------------------------------------------------------------------------
# Funções
# ---------------------------------------------------------------------------

def treinar_kmeans(Z: np.ndarray, k: int = K, n_init: int = N_INIT) -> object:
    """Treina K-Means no espaço PCA com k=2 clusters.

    Por que k=2?
    O problema biológico é binário: resistente vs. sensível.
    Dois clusters mapeiam diretamente para as duas classes esperadas.

    Por que k-means++ (init="k-means++")?
    Inicialização inteligente que distribui os centróides iniciais de forma
    espaçada — reduz drasticamente a chance de convergência para ótimo local
    comparado à inicialização aleatória pura.

    n_init=20: roda o algoritmo 20 vezes com centróides iniciais diferentes
    e retorna o resultado com menor inertia (soma de distâncias ao centróide).
    """
    from sklearn.cluster import KMeans

    km = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=n_init,
        random_state=RANDOM_STATE,
        algorithm="lloyd",  # algoritmo clássico, mais estável que "elkan" para dados densos
    )
    km.fit(Z)
    print(f"  K-Means: k={k}, inertia={km.inertia_:.2f}, iterações={km.n_iter_}")
    return km


def reconstruir_mapa_2d(labels_roi: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """Reprojetar os labels de cluster (1D) de volta ao espaço 2D da imagem.

    Os labels saem do K-Means como um vetor 1D (um valor por pixel da ROI).
    Para visualizar, precisamos colocá-los de volta nas posições espaciais
    corretas dentro da imagem original.

    Estratégia:
      1. Cria mapa 2D cheio de NaN (pixels fora da ROI)
      2. Usa a máscara booleana para colocar os labels nas posições certas
         (mascara == True → posições onde existem pixels da ROI)

    NaN fora da ROI será pintado de preto no matplotlib com set_bad().
    """
    label_map = np.full(mascara.shape, np.nan, dtype=np.float32)
    label_map[mascara] = labels_roi.astype(np.float32)
    return label_map


def plotar_mosaico_clusters(label_maps: dict, k: int, cols: int = 4):
    """Plota grid com os mapas de cluster de todas as amostras.

    Usa colormap discreto (uma cor por cluster) em vez de contínuo
    para que a separação entre clusters seja visualmente clara.
    vmin=-0.5 e vmax=k-0.5 centralizam as cores nos inteiros 0 e 1.
    """
    n    = len(label_maps)
    rows = -(-n // cols)

    # colormap com k cores distintas — uma por cluster
    cmap_base = plt.get_cmap("gist_ncar")
    cores = cmap_base(np.linspace(0, 1, k))
    cmap_discreta = ListedColormap(cores)
    cmap_discreta.set_bad("black")  # NaN (fora da ROI) → preto

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.atleast_1d(axes).flatten()

    for ax, (nome, mapa) in zip(axes, label_maps.items()):
        im = ax.imshow(np.ma.masked_invalid(mapa), cmap=cmap_discreta,
                       vmin=-0.5, vmax=k - 0.5)
        ax.set_title(nome[:18], fontsize=8)
        ax.axis("off")

    for ax in axes[n:]:
        ax.set_visible(False)

    # barra de cores com ticks nos centros dos clusters (0, 1, ...)
    fig.colorbar(im, ax=axes[:n], ticks=range(k), shrink=0.5, label="cluster")
    fig.suptitle(f"Mapas de cluster K-Means (k={k})", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


def calcular_importancia_bandas(df_norm: pd.DataFrame,
                                 labels: np.ndarray,
                                 nome: str):
    """Calcula e plota a importância de cada banda via variância dos centróides.

    A variância entre os centróides dos dois clusters por banda indica
    o quanto aquela banda contribui para separar os grupos.
    Alta variância → banda discriminativa (diferença espectral relevante).
    Baixa variância → banda redundante para a classificação.

    Isto é equivalente a uma análise de importância de features para K-Means.
    """
    colunas_banda = [c for c in df_norm.columns if c.startswith("band_")]
    X = df_norm[colunas_banda].values.astype(np.float32)

    # centróide de cada cluster: média espectral dos pixels daquele cluster
    centroids = np.vstack([
        X[labels == c].mean(axis=0) if np.any(labels == c) else np.zeros(X.shape[1])
        for c in range(K)
    ])

    # variância entre centróides por banda (axis=0 = entre clusters)
    importancia = np.var(centroids, axis=0)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.bar(np.arange(len(importancia)), importancia, color="steelblue", alpha=0.8)
    ax.set_xlabel("Índice da banda")
    ax.set_ylabel("Variância entre centróides")
    ax.set_title(f"Importância das bandas — {nome}")
    plt.tight_layout()
    plt.show()

    return centroids


def plotar_scatter_clusters(Z: np.ndarray, labels: np.ndarray,
                             meta: pd.DataFrame, titulo: str,
                             max_amostras: int = PCA_MAX_AMOSTRAS):
    """Scatter 2D colorido por cluster K-Means.

    Complementa os scatter por rótulo do 2.2: permite ver se os clusters
    encontrados pelo K-Means coincidem com os rótulos reais (resistente/sensivel).
    """
    n   = min(Z.shape[0], max_amostras)
    idx = np.sort(
        np.random.default_rng(RANDOM_STATE).choice(Z.shape[0], size=n, replace=False)
    )
    Z_sub      = Z[idx]
    labels_sub = labels[idx]
    meta_sub   = meta.iloc[idx]

    cmap_k = ListedColormap(plt.get_cmap("tab10")(np.linspace(0, 1, K)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # esquerda: colorido por cluster K-Means
    sc = ax1.scatter(Z_sub[:, 0], Z_sub[:, 1], c=labels_sub,
                     cmap=cmap_k, vmin=-0.5, vmax=K - 0.5,
                     s=3, alpha=0.6, linewidths=0)
    fig.colorbar(sc, ax=ax1, ticks=range(K), label="cluster")
    ax1.set_title(f"{titulo} — cluster K-Means")

    # direita: colorido por rótulo real (para comparar visualmente)
    rotulos_unicos = sorted(meta_sub["rotulo"].unique())
    cores = plt.get_cmap("Set1")(np.linspace(0, 1, max(len(rotulos_unicos), 2)))
    for i, rot in enumerate(rotulos_unicos):
        m = meta_sub["rotulo"].values == rot
        ax2.scatter(Z_sub[m, 0], Z_sub[m, 1],
                    c=[cores[i]], s=3, alpha=0.6, linewidths=0, label=rot)
    ax2.legend(markerscale=4, fontsize=9)
    ax2.set_title(f"{titulo} — rótulo real")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # --- 1. Carrega entradas ---
    print("Carregando dados ...")
    df_Z    = pd.read_pickle(Z_PCA_PATH)       # componentes PCA + sample + rotulo
    df_norm = pd.read_pickle(DADOS_NORM_PATH)  # bandas normalizadas + sample + rotulo

    with open(MASKS_PATH, "rb") as f:
        masks = pickle.load(f)                 # {nome: ndarray bool (H, W)}

    # separa componentes PCA (features do K-Means) dos metadados
    colunas_pca = [c for c in df_Z.columns if c.startswith("pca_")]
    Z    = df_Z[colunas_pca].values.astype(np.float32)
    meta = df_Z[["sample", "rotulo"]].copy()

    print(f"  {Z.shape[0]:,} pixels  |  {Z.shape[1]} componentes PCA")

    # --- 2. Treina K-Means no espaço PCA ---
    print(f"\nTreinando K-Means (k={K}) ...")
    kmeans = treinar_kmeans(Z)
    labels = kmeans.labels_  # array 1D: cluster de cada pixel (0 ou 1)

    # --- 3. Reprojeção espacial e visualização por amostra ---
    print("\nReconstruindo mapas 2D de cluster ...")
    label_maps   = {}   # {nome: mapa 2D float com NaN fora da ROI}
    labels_dict  = {}   # {nome: labels 1D} para o 2.4

    for nome in sorted(meta["sample"].unique()):
        mask_amostra   = meta["sample"].values == nome
        labels_amostra = labels[mask_amostra]
        labels_dict[nome] = labels_amostra

        # reconstrói o mapa espacial 2D a partir da máscara booleana da ROI
        mapa = reconstruir_mapa_2d(labels_amostra, masks[nome])
        label_maps[nome] = mapa

        # importância de bandas desta amostra (quais comprimentos de onda
        # separam os dois clusters nesta colônia específica)
        df_amostra = df_norm[df_norm["sample"] == nome]
        calcular_importancia_bandas(df_amostra, labels_amostra, nome)

    # mosaico de todos os mapas de cluster
    plotar_mosaico_clusters(label_maps, K)

    # --- 4. Scatter plots coloridos por cluster ---
    # PCA 2D: usa os primeiros 2 componentes do Z_pca diretamente
    Z_2d = Z[:, :2]
    plotar_scatter_clusters(Z_2d, labels, meta, "PCA 2D")

    # t-SNE 2D: recalcula t-SNE sobre uma subamostra de Z
    print("\nGerando t-SNE 2D (pode demorar) ...")
    from sklearn.manifold import TSNE
    n_tsne = min(Z.shape[0], TSNE_MAX_AMOSTRAS)
    idx_tsne = np.sort(
        np.random.default_rng(RANDOM_STATE).choice(Z.shape[0], size=n_tsne, replace=False)
    )
    Z_tsne = TSNE(
        n_components=2,
        perplexity=min(30.0, float(n_tsne - 1)),
        max_iter=TSNE_MAX_ITER,
        random_state=RANDOM_STATE,
        init="pca",
    ).fit_transform(Z[idx_tsne])
    plotar_scatter_clusters(Z_tsne, labels[idx_tsne],
                             meta.iloc[idx_tsne], "t-SNE 2D",
                             max_amostras=n_tsne)

    # --- 5. Salva resultados ---
    with open(KMEANS_MODEL, "wb") as f:
        pickle.dump(kmeans, f)
    print(f"\nSalvo em '{KMEANS_MODEL}'")

    # labels_kmeans.pkl: DataFrame alinhado com dados_normalizados
    # contém sample, rotulo e cluster — usado pelo 2.4 para métricas
    df_labels = meta.copy()
    df_labels["cluster"] = labels
    df_labels.to_pickle(LABELS_PATH)
    print(f"Salvo em '{LABELS_PATH}'")

    print("\n[2.3] Concluído.")
