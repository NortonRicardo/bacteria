"""
2.4 - Avaliação e Métricas
Recebe os labels K-Means (labels_kmeans.pkl) e os dados normalizados
(dados_normalizados.pkl) para:
  - Identificar qual cluster corresponde a resistente via amostras ATCC
  - Calcular ARI, NMI e Pureza por amostra e global
  - Plotar perfis espectrais dos centróides de cada amostra
  - Exportar metricas.csv

Entrega: data/metricas.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR        = Path("data")
LABELS_PATH     = DATA_DIR / "labels_kmeans.pkl"
DADOS_NORM_PATH = DATA_DIR / "dados_normalizados.pkl"
METRICAS_PATH   = DATA_DIR / "metricas.csv"

# Amostras cuja resistência é biologicamente conhecida (cepas ATCC de referência).
# Usadas para descobrir qual cluster (0 ou 1) o K-Means associou às resistentes —
# o K-Means não conhece os rótulos, então os IDs de cluster são arbitrários.
AMOSTRAS_RESISTENTES_CONHECIDAS = [
    "ATCC13_240506-161053",
    "ATCC16_240506-161158",
    "ATCC27_240506-161129",
]

K = 2  # número de clusters (deve ser consistente com o 2.3)


# ---------------------------------------------------------------------------
# Funções
# ---------------------------------------------------------------------------

def identificar_cluster_resistente(df_labels: pd.DataFrame,
                                    amostras_resistentes: list[str]) -> int:
    """Descobre qual cluster (0 ou 1) o K-Means associou às amostras resistentes.

    Por que é necessário?
    O K-Means não tem noção de rótulos — ele apenas separa os dados em grupos.
    O cluster "0" pode ser resistente em uma execução e sensível em outra,
    dependendo da inicialização. Precisamos ancorar a interpretação em amostras
    com resistência biologicamente conhecida (as cepas ATCC).

    Estratégia:
      - Para cada ATCC, conta quantos pixels caíram no cluster 0 vs. cluster 1
      - O cluster que acumula mais pixels nas ATCCs é declarado "resistente"
    """
    votos = {0: 0, 1: 0}

    for nome in amostras_resistentes:
        subset = df_labels[df_labels["sample"] == nome]
        if subset.empty:
            print(f"  [AVISO] Amostra ATCC não encontrada nos labels: {nome}")
            continue
        # conta pixels por cluster nesta amostra resistente conhecida
        contagem = subset["cluster"].value_counts()
        c0 = int(contagem.get(0, 0))
        c1 = int(contagem.get(1, 0))
        tot = c0 + c1
        print(f"  {nome}: cluster0={c0/tot:.1%}  cluster1={c1/tot:.1%}  "
              f"predominante={'0' if c0 >= c1 else '1'}")
        votos[0] += c0
        votos[1] += c1

    cluster_resistente = 0 if votos[0] >= votos[1] else 1
    print(f"\n  → cluster_resistente = {cluster_resistente} "
          f"(acumulou {votos[cluster_resistente]:,} pixels nas ATCCs)")
    return cluster_resistente


def calcular_metricas(df_labels: pd.DataFrame,
                      cluster_resistente: int) -> pd.DataFrame:
    """Calcula ARI, NMI e Pureza para o dataset global e por imagem.

    Métricas:
    - ARI (Adjusted Rand Index): mede concordância entre clustering e rótulos
      reais, corrigindo para acertos por acaso. Varia de -1 a 1 (1 = perfeito).
    - NMI (Normalized Mutual Information): mede informação compartilhada entre
      clustering e rótulos. Varia de 0 a 1 (1 = perfeito).
    - Pureza: fração de pixels cujo cluster previsto coincide com o rótulo real.
      Métrica mais intuitiva: "quantos % acertamos?"

    cluster_resistente: o ID do cluster que representa resistência (0 ou 1),
    determinado pela função anterior a partir das ATCCs.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    # converte rótulos textuais para binários (1=resistente, 0=sensível)
    def rotulo_para_int(r: str) -> int:
        return 1 if str(r).strip().lower() == "resistente" else 0

    y_true = df_labels["rotulo"].apply(rotulo_para_int).values

    # alinha a predição do cluster com a semântica de rótulo:
    # se cluster_resistente=1 → cluster 1 vira predição "resistente" (1)
    # se cluster_resistente=0 → inverter: cluster 0 vira resistente (1)
    y_pred = df_labels["cluster"].values.copy()
    if cluster_resistente == 0:
        # inverte: 0↔1 para que o cluster resistente seja sempre predição=1
        y_pred = 1 - y_pred

    linhas = []

    # --- métricas globais ---
    ari_global = adjusted_rand_score(y_true, y_pred)
    nmi_global = normalized_mutual_info_score(y_true, y_pred)
    pureza_global = float(np.mean(y_true == y_pred))
    linhas.append({
        "imagem": "GLOBAL",
        "rotulo": "misto",
        "n_pixels": len(y_true),
        "ARI": round(ari_global, 4),
        "NMI": round(nmi_global, 4),
        "Pureza": round(pureza_global, 4),
    })
    print(f"\n  GLOBAL  ARI={ari_global:.4f}  NMI={nmi_global:.4f}  "
          f"Pureza={pureza_global:.4f}  n={len(y_true):,}")

    # --- métricas por imagem ---
    for nome in sorted(df_labels["sample"].unique()):
        mask = df_labels["sample"].values == nome
        yt = y_true[mask]
        yp = y_pred[mask]
        rot = df_labels.loc[mask, "rotulo"].iloc[0]
        pureza = float(np.mean(yt == yp))
        linhas.append({
            "imagem": nome,
            "rotulo": rot,
            "n_pixels": int(mask.sum()),
            "ARI": 0.0,   # ARI por imagem não é definido (classe única por imagem)
            "NMI": 0.0,
            "Pureza": round(pureza, 4),
        })
        print(f"  {nome[:20]:20s}  rotulo={rot:10s}  Pureza={pureza:.4f}  n={mask.sum():,}")

    return pd.DataFrame(linhas)


def plotar_metricas(df_metricas: pd.DataFrame):
    """Gráfico de barras horizontal com a pureza por amostra.

    Barras coloridas por rótulo (resistente=vermelho, sensível=azul)
    para identificar visualmente se o modelo erra mais em um tipo específico.
    """
    df_img = df_metricas[df_metricas["imagem"] != "GLOBAL"].copy()
    cores  = df_img["rotulo"].map({"resistente": "tomato", "sensivel": "steelblue"})

    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(df_img))))
    y_pos = np.arange(len(df_img))
    ax.barh(y_pos, df_img["Pureza"].values, color=cores.values, alpha=0.85)
    ax.axvline(0.5, color="gray", ls="--", lw=1, label="acaso (50%)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_img["imagem"].str[:20].values, fontsize=8)
    ax.set_xlabel("Pureza")
    ax.set_title("Pureza por amostra")
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plotar_centroides(df_norm: pd.DataFrame, df_labels: pd.DataFrame,
                      cluster_resistente: int):
    """Plota o perfil espectral médio de cada cluster por amostra.

    Os centróides espectrais mostram em quais comprimentos de onda os clusters
    diferem — estas são as "assinaturas espectrais" de resistente vs. sensível.
    Um bom modelo deve mostrar diferenças claras e consistentes entre amostras.
    """
    colunas_banda = [c for c in df_norm.columns if c.startswith("band_")]
    bandas_idx    = np.arange(len(colunas_banda))

    # nomes e cores dos clusters (o cluster resistente é sempre vermelho)
    nomes_cluster = {cluster_resistente: "Resistente", 1 - cluster_resistente: "Sensível"}
    cores_cluster = {cluster_resistente: "tomato",     1 - cluster_resistente: "steelblue"}

    amostras = sorted(df_norm["sample"].unique())
    cols     = 4
    rows     = -(-len(amostras) // cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3))
    axes = np.atleast_1d(axes).flatten()

    for ax, nome in zip(axes, amostras):
        mask_amostra = df_norm["sample"].values == nome
        X_amostra    = df_norm.loc[mask_amostra, colunas_banda].values.astype(np.float32)
        labs_amostra = df_labels.loc[df_labels["sample"] == nome, "cluster"].values
        rot          = df_labels.loc[df_labels["sample"] == nome, "rotulo"].iloc[0]

        for c in range(K):
            pxls = X_amostra[labs_amostra == c]
            if len(pxls) == 0:
                continue
            media = pxls.mean(axis=0)
            desvio = pxls.std(axis=0)
            ax.plot(bandas_idx, media,
                    color=cores_cluster[c], lw=1.5, label=nomes_cluster[c])
            ax.fill_between(bandas_idx,
                            media - desvio, media + desvio,
                            color=cores_cluster[c], alpha=0.15)

        ax.set_title(f"{nome[:14]} ({rot})", fontsize=8)
        ax.set_xlabel("Banda", fontsize=7)
        ax.set_ylabel("Reflectância", fontsize=7)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)

    for ax in axes[len(amostras):]:
        ax.set_visible(False)

    fig.suptitle("Perfis espectrais dos centróides por amostra", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # --- 1. Carrega entradas ---
    print("Carregando dados ...")
    df_labels = pd.read_pickle(LABELS_PATH)    # sample, rotulo, cluster
    df_norm   = pd.read_pickle(DADOS_NORM_PATH)  # sample, rotulo, band_*

    print(f"  {len(df_labels):,} pixels  |  {df_labels['sample'].nunique()} amostras")

    # --- 2. Identifica qual cluster = resistente via ATCCs ---
    print("\nIdentificando cluster resistente pelas amostras ATCC ...")
    cluster_resistente = identificar_cluster_resistente(
        df_labels, AMOSTRAS_RESISTENTES_CONHECIDAS
    )

    # --- 3. Calcula métricas ---
    print("\nCalculando métricas ...")
    df_metricas = calcular_metricas(df_labels, cluster_resistente)

    # --- 4. Salva CSV de métricas ---
    df_metricas.to_csv(METRICAS_PATH, index=False)
    print(f"\nSalvo em '{METRICAS_PATH}'")

    # --- 5. Visualizações ---
    plotar_metricas(df_metricas)
    plotar_centroides(df_norm, df_labels, cluster_resistente)

    print("\n[2.4] Concluído.")
