# Bactera — Hyperspectral Imaging Analysis

Análise de imagens hiperespectrais (NIR/SWIR, 909–2512 nm) de amostras de bactérias capturadas com sensor **Specim** no formato ENVI BIL.

---

## Requisitos

- macOS, Linux ou Windows (WSL2)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) ou [Anaconda](https://www.anaconda.com/download)

---

## 1. Instalar o Conda (Miniconda)

> Pule esta etapa se já tiver o conda instalado (`conda --version`).

**macOS (Apple Silicon ou Intel):**

```bash
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o miniconda.sh
# Intel Mac: use Miniconda3-latest-MacOSX-x86_64.sh
bash miniconda.sh -b -p "$HOME/miniconda3"
"$HOME/miniconda3/bin/conda" init zsh   # ou bash
exec $SHELL
```

**Linux:**

```bash
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b -p "$HOME/miniconda3"
"$HOME/miniconda3/bin/conda" init bash
exec $SHELL
```

**Windows (PowerShell):**

```powershell
# Baixe e execute o instalador gráfico em:
# https://docs.conda.io/en/latest/miniconda.html
# Após a instalação, abra o "Anaconda Prompt" e siga os passos abaixo.
```

---

## 2. Criar e ativar o ambiente

Na raiz do projeto (onde está o `environment.yml`):

```bash
conda env create -f environment.yml
conda activate bactera
```

Para confirmar que o ambiente está ativo:

```bash
conda info --envs        # deve mostrar * ao lado de bactera
python --version         # deve mostrar Python 3.11.x
```

---

## 3. Registrar o kernel no Jupyter

```bash
python -m ipykernel install --user --name bactera --display-name "Python (bactera)"
```

---

## 4. Executar o notebook

```bash
jupyter lab main.ipynb
```

Selecione o kernel **Python (bactera)** e execute as células em ordem.

---

## Estrutura do projeto

```
bactera/
├── data/               # amostras hiperespectrais (ignorado no git)
│   └── <amostra>/
│       ├── capture/
│       │   ├── <amostra>.hdr
│       │   ├── <amostra>.raw
│       │   ├── DARKREF_<amostra>.hdr/.raw
│       │   └── WHITEREF_<amostra>.hdr/.raw
│       └── metadata/
├── main.ipynb          # notebook principal
├── environment.yml     # definição do ambiente conda
└── README.md
```

---

## Atualizar o ambiente

Se o `environment.yml` for alterado:

```bash
conda env update -f environment.yml --prune
```

## Remover o ambiente

```bash
conda deactivate
conda env remove -n bactera
```
