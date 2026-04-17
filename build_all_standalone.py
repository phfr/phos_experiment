"""Build v3/all.ipynb — merged v2 pipeline, standalone (no project imports). Run: python v3/build_all_standalone.py"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_NB = HERE / "all.ipynb"

V2_CLUSTER_FIT = r"""
mask = finite_rows_mask(df)
sub = df.loc[mask].copy()
Xn = log2fc_matrix_from_df(sub).to_numpy(dtype=float)
Xp = Xn - np.nanmean(Xn, axis=1, keepdims=True)
Xp = StandardScaler().fit_transform(Xp)
labels, cinfo = cluster_shape_embedding(
    Xp,
    method=CLUSTER_METHOD,
    kmeans_n_clusters=KMEANS_N_CLUSTERS,
    random_state=RNG,
    hdbscan_min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
    hdbscan_min_samples=HDBSCAN_MIN_SAMPLES,
    hdbscan_selection=HDBSCAN_SELECTION,
    retry_if_all_noise=True,
)
sub["cluster"] = labels.astype(np.int64)
print(cinfo)
if "hint" in cinfo:
    print(cinfo["hint"])
print(
    f"→ {cinfo.get('n_clusters', 0)} cluster(s), {cinfo.get('n_noise', 0)} noise sites "
    f"(min_cluster_size={cinfo.get('min_cluster_size')}, min_samples={cinfo.get('min_samples')}, selection={cinfo.get('selection')!r})"
)
if cinfo.get("hdbscan_auto_retry"):
    print("  (used automatic retry after first fit had 0 clusters)")
"""


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in s.split("\n")]}


def code(s: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in s.split("\n")],
    }


LIBS_AND_LOAD = r'''%matplotlib inline
from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)
# upsetplot uses pandas patterns that trigger FutureWarning on newer pandas; warnings originate in their package
warnings.filterwarnings("ignore", category=FutureWarning, module="upsetplot")
# UpSet axes are not tight_layout-friendly (matplotlib)
warnings.filterwarnings(
    "ignore",
    message=".*not compatible with tight_layout.*",
    category=UserWarning,
)

try:
    from IPython.display import display, Markdown
except ImportError:
    display = print
    Markdown = str


# =============================================================================
# Opinionated defaults — edit here (thresholds, clustering, Fisher, plots, …)
# =============================================================================

# --- Significance gates (BH FDR q and |Log2FC|) ---
Q_MAX_SIG = 0.05
ABS_LFC_SIG = 1.0
Q_MAX_VOLCANO = 0.05
ABS_LFC_VOLCANO = 2.5

RUN_CONFIG = {
    "q_max_sig": Q_MAX_SIG,
    "abs_lfc_sig": ABS_LFC_SIG,
    "q_max_volcano": Q_MAX_VOLCANO,
    "abs_lfc_volcano": ABS_LFC_VOLCANO,
}

# --- Profile clustering (§04) ---
CLUSTER_METHOD = "hdbscan"  # "hdbscan" or "kmeans"
KMEANS_N_CLUSTERS = 8
KMEANS_N_INIT = 10
RNG = 42
HDBSCAN_MIN_CLUSTER_SIZE = None
HDBSCAN_MIN_SAMPLES = None
HDBSCAN_SELECTION = "leaf"

# --- Fisher / term enrichment ---
FISHER_BARS_PER_CLUSTER_MAX = 8
FISHER_MIN_SUPPORT_MULTIVIZ = 5
FISHER_TOP_TERMS_MULTIVIZ = 9
FISHER_MIN_SUPPORT_SPECIFICITY = 5
FISHER_TOP_TERMS_SPECIFICITY = 25
FISHER_MIN_SUPPORT_DIRECTION = 4
FISHER_TOP_TERMS_DIRECTION = 10
MULTIVIZ_MAX_SITES_LISTED = 400
MULTIVIZ_MAX_GENE_CHARS = 12000
MULTIVIZ_FISHER_MERGED_CAP = 30

# --- Specificity tail (§06) ---
SPECIFICITY_TAIL_QUANTILE = 0.9
SPECIFICITY_TAIL_MIN_SITES = 10

# --- GSEA prerank (§12) ---
GSEA_PERMUTATIONS = 100
GSEA_THREADS = 2
GSEA_MIN_GENESET = 10
GSEA_MAX_GENESET = 500
GSEA_TOP_PATHWAYS_BAR = 15

# --- Kinase z-scores (§13) ---
KINASE_MIN_SUBSTRATES = 5
KINASE_N_PERM = 200
KINASE_BAR_TOP_N = 15
KINASE_HEATMAP_MAX_KINASES = 40
KINASE_CLUSTERMAP_MAX_KINASES = 28
KINASE_CLUSTERMAP_METHOD = "average"
KINASE_CLUSTERMAP_METRIC = "euclidean"
KINASE_DOT_MAX_KINASES = 32
KINASE_LINEPLOT_TOP_N = 10
KINASE_VERBOSE = True

# --- Co-regulation / STRING / UpSet (§14) ---
GENE_CORR_MIN_SITES = 2
GENE_CORR_EDGE_THRESHOLD = 0.75
GENE_CORR_MAX_NODES_DRAW = 70
GENE_CORR_SPRING_ITERATIONS = 200
GENE_CORR_LABEL_TOP_NODES = 22
GENE_CORR_HUB_SUBGRAPH_N = 45
GENE_CORR_EMBED_MAX_NODES = 220
STRING_TOP_N_GENES = 25
STRING_REQUIRED_SCORE = 400
STRING_TIMEOUT_SEC = 45
STRING_NETWORK_LAYOUT_SEED = 42
STRING_SPRING_K = 0.4

# --- Multivariate outliers (§02b) ---
MAHAL_CHI2_QUANTILE = 0.99
MINCOV_DET_SUPPORT_FRACTION = 0.75
ISOFOREST_CONTAMINATION = 0.05
ISOFOREST_N_ESTIMATORS = 300
LOF_N_NEIGHBORS = 35
LOF_CONTAMINATION = 0.05
OUTLIER_SCORE_DISPLAY_Q = 0.995
OUTLIER_TOP_TABLE_N = 25

# --- UniProt windows (§11) ---
UNIPROT_MAX_ACCESSIONS = 2000
SEQUENCE_FLANK_AA = 7

# --- Figure export DPI ---
FIG_DPI_DEFAULT = 140
FIG_DPI_HEATMAP = 160
FIG_DPI_CLUSTER_MULTIVIZ = 155
FIG_DPI_DIRECTION_FISHER = 130


# --- resolve data.csv (prefer cwd, then v3/data.csv when walking parents) ---
def find_data_csv():
    start = Path.cwd().resolve()
    candidates = []
    for base in [start, *start.parents]:
        candidates.extend([base / "data.csv", base / "v3" / "data.csv"])
    seen = set()
    for p in candidates:
        r = p.resolve()
        if r in seen:
            continue
        seen.add(r)
        if r.is_file():
            return r
    raise FileNotFoundError(
        "Could not find data.csv. Put it next to this notebook (e.g. v3/data.csv) or run from a folder that contains it."
    )


DATA = find_data_csv()
OUT_ROOT = DATA.parent / "outputs_all"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
print("DATA =", DATA)
print("OUT_ROOT =", OUT_ROOT)


# --- io_contrasts (embedded) ---
@dataclass(frozen=True)
class Contrast:
    key: str
    log2fc_col: str
    neglog10_p_col: str
    label: str


CONTRASTS = (
    Contrast("aCD3_5min", "Log2FC_αCD3_5min", "-Log_p-value_αCD3_5min", "αCD3 5min"),
    Contrast("aCD3_aCD226_5min", "Log2FC_αCD3_αCD226_5min", "-Log_p-value_αCD3_αCD226_5min", "αCD3+αCD226 5min"),
    Contrast("aCD3_aICOS_5min", "Log2FC_αCD3_αICOS_5min", "-Log_p-value_αCD3_αICOS_5min", "αCD3+αICOS 5min"),
    Contrast("aCD3_aCD2_5min", "Log2FC_αCD3_αCD2_5min", "-Log_p-value_αCD3_αCD2_5min", "αCD3+αCD2 5min"),
    Contrast("aCD3_10min", "Log2FC_αCD3_10min", "-Log_p-value_αCD3_10min", "αCD3 10min"),
    Contrast(
        "aCD3_aCD226_10min",
        "Log2FC_αCD3_αCD226_10min",
        "-Log_p-value_αCD3+αCD226_10min",
        "αCD3+αCD226 10min",
    ),
    Contrast("aCD3_aICOS_10min", "Log2FC_αCD3_αICOS_10min", "-Log_p-value_αCD3_αICOS_10min", "αCD3+αICOS 10min"),
    Contrast(
        "aCD3_aCD2_10min",
        "Log2FC_αCD3_αCD2_10min",
        "-Log_p-value_αCD3+αCD2_10min",
        "αCD3+αCD2 10min",
    ),
)


def contrast_keys():
    return [c.key for c in CONTRASTS]


def contrast_dict_for_fdr():
    return {c.key: c.neglog10_p_col for c in CONTRASTS}


# --- analysis/fdr (embedded) ---
def neglog10p_to_p(neglog10_p):
    x = np.asarray(neglog10_p, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    p = np.power(10.0, -x)
    return np.clip(p, 1e-300, 1.0)


def bh_fdr(pvalues):
    p = np.asarray(pvalues, dtype=float)
    p = np.clip(p, 1e-300, 1.0)
    m = p.size
    if m == 0:
        return p.copy()
    order = np.argsort(p)
    p_sorted = p[order]
    ranks = np.arange(1, m + 1, dtype=float)
    tmp = p_sorted * m / ranks
    q_sorted = np.empty(m, dtype=float)
    q_sorted[m - 1] = float(min(tmp[m - 1], 1.0))
    for i in range(m - 2, -1, -1):
        q_sorted[i] = float(min(tmp[i], q_sorted[i + 1]))
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    out = np.empty(m, dtype=float)
    out[order] = q_sorted
    return out


def add_fdr_columns(df, keys, neglog10_p_cols):
    out = df.copy()
    for key in keys:
        col = neglog10_p_cols[key]
        p = neglog10p_to_p(out[col].to_numpy())
        out[f"p_{key}"] = p
        out[f"q_{key}"] = bh_fdr(p)
    return out


# --- v2._lib/load_data logic (embedded) ---
ANNOTATION_LAYERS = {
    "GOBP": "_GOBP_name_a",
    "GOMF": "_GOMF_name_a",
    "GOCC": "_GOCC_name_a",
    "KEGG": "_KEGG_name_a",
}


def load_data_csv(path, *, add_fdr=True):
    p = Path(path)
    df0 = pd.read_csv(p, sep="\t", low_memory=False)
    df0.columns = [str(c).strip() for c in df0.columns]
    for c in CONTRASTS:
        if c.log2fc_col not in df0.columns or c.neglog10_p_col not in df0.columns:
            raise ValueError(f"Missing contrast columns for {c.key!r}")
        df0[c.log2fc_col] = pd.to_numeric(df0[c.log2fc_col], errors="coerce")
        df0[c.neglog10_p_col] = pd.to_numeric(df0[c.neglog10_p_col], errors="coerce")
    if add_fdr:
        df0 = add_fdr_columns(df0, contrast_keys(), contrast_dict_for_fdr())
    return df0


def log2fc_matrix_from_df(df0):
    return pd.DataFrame({c.key: df0[c.log2fc_col] for c in CONTRASTS}, index=df0.index)


def finite_rows_mask(df0):
    cols = [c.log2fc_col for c in CONTRASTS]
    return df0[cols].apply(np.isfinite).all(axis=1)


def parse_terms_long(df0, *, id_col="id", gene_col="geneid"):
    rows = []
    for layer, col in ANNOTATION_LAYERS.items():
        if col not in df0.columns:
            continue
        for i in df0.index:
            raw = df0.at[i, col]
            sid = str(df0.at[i, id_col])
            gid = "" if gene_col not in df0.columns else str(df0.at[i, gene_col])
            if pd.isna(raw) or str(raw).strip() == "":
                continue
            for t in str(raw).split(","):
                t = t.strip()
                if t:
                    rows.append({"layer": layer, "term": t, "id": sid, "geneid": gid})
    return pd.DataFrame(rows)


def build_term_site_index(terms_long, layer):
    sub_t = terms_long[terms_long["layer"] == layer]
    out = {}
    for term, grp in sub_t.groupby("term"):
        out[str(term)] = set(grp["id"].astype(str))
    return out


def fisher_terms_vs_rest(query_ids, universe_ids, terms_long, layer, *, min_support=5, top_n=40):
    from scipy.stats import fisher_exact

    q = set(query_ids)
    u = set(universe_ids)
    if not q.issubset(u):
        q = q & u
    rest = u - q
    n_u = len(u)
    if n_u == 0 or len(q) == 0:
        return pd.DataFrame()

    idx = build_term_site_index(terms_long, layer)
    rows_out = []
    n_q = len(q)
    n_r = len(rest)

    for term, sites in idx.items():
        if len(sites & u) < min_support:
            continue
        in_q = len(sites & q)
        in_r = len(sites & rest)
        a = in_q
        b = n_q - in_q
        c = in_r
        d = n_r - in_r
        if a + b == 0 or c + d == 0:
            continue
        oddsr, p_two = fisher_exact([[a, b], [c, d]])
        rows_out.append(
            {
                "term": term,
                "n_query_with": a,
                "n_query": n_q,
                "n_rest_with": c,
                "n_rest": n_r,
                "odds_ratio": oddsr,
                "p_fisher": p_two,
            }
        )
    if not rows_out:
        return pd.DataFrame()
    res = pd.DataFrame(rows_out)
    res = res.sort_values("p_fisher").head(top_n)
    return res


def cluster_shape_embedding(
    Xp,
    *,
    method="hdbscan",
    kmeans_n_clusters=8,
    random_state=42,
    hdbscan_min_cluster_size=None,
    hdbscan_min_samples=None,
    hdbscan_selection="leaf",
    retry_if_all_noise=True,
):
    from sklearn.cluster import KMeans

    n = len(Xp)
    method = (method or "hdbscan").lower()
    info = {"method": method}

    if method == "kmeans":
        km = KMeans(n_clusters=int(kmeans_n_clusters), random_state=random_state, n_init=KMEANS_N_INIT)
        lab = km.fit_predict(Xp)
        info["n_clusters"] = int(kmeans_n_clusters)
        info["n_noise"] = 0
        return lab.astype(int), info

    if method != "hdbscan":
        raise ValueError("method must be 'hdbscan' or 'kmeans'")

    try:
        import hdbscan as _hdb
    except ImportError as e:
        raise ImportError("Install hdbscan for density-based clustering: pip install hdbscan") from e

    user_mcs = hdbscan_min_cluster_size is not None
    _mcs = hdbscan_min_cluster_size if user_mcs else max(5, n // 150)
    _mcs = max(2, int(_mcs))
    if hdbscan_min_samples is not None:
        _ms = max(2, int(hdbscan_min_samples))
    else:
        _ms = max(2, min(8, max(3, _mcs // 5)))
    _sel = (hdbscan_selection or "leaf").lower()
    if _sel not in ("eom", "leaf"):
        _sel = "leaf"

    def _fit(mcs: int, ms: int, sel: str) -> np.ndarray:
        h = _hdb.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            metric="euclidean",
            cluster_selection_method=sel,
        )
        return h.fit_predict(Xp).astype(int)

    lab = _fit(_mcs, _ms, _sel)
    n_cl = len(set(lab)) - (1 if -1 in lab else 0)

    if n_cl == 0 and retry_if_all_noise and not user_mcs:
        _mcs2 = max(5, _mcs // 2)
        _ms2 = max(2, min(5, _ms))
        lab = _fit(_mcs2, _ms2, "leaf")
        n_cl = len(set(lab)) - (1 if -1 in lab else 0)
        info["hdbscan_auto_retry"] = True
        info["min_cluster_size_retry"] = _mcs2
        info["min_samples_retry"] = _ms2
        _mcs, _ms, _sel = _mcs2, _ms2, "leaf"

    n_noise = int(np.sum(lab == -1))
    n_cl = len(set(lab)) - (1 if -1 in lab else 0)
    info.update(
        {
            "n_clusters": n_cl,
            "n_noise": n_noise,
            "min_cluster_size": _mcs,
            "min_samples": _ms,
            "selection": _sel,
        }
    )
    if n_cl == 0:
        info["hint"] = (
            "HDBSCAN still found no clusters (all noise). "
            "Try HDBSCAN_MIN_CLUSTER_SIZE=5–10, HDBSCAN_MIN_SAMPLES=2–3, "
            "or CLUSTER_METHOD='kmeans' for a fixed partition."
        )
    return lab, info


df = load_data_csv(DATA)
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
'''.strip()


def main() -> None:
    cells: list = []

    cells.append(
        md(
            """# Phospho analysis — **all-in-one** (`v3/all.ipynb`)

This notebook merges the **v2** pipeline into a **single standalone** file: **no** `sys.path` hacks and **no** imports from `analysis.*` or `v2._lib` — those definitions are **inlined** in the first code cell.

**Contents (run top to bottom):** (1) load + QC summaries and volcano plots; (2) contrast–contrast correlation; (3) multivariate outlier / discordance scores; (4) significant-set overlap; (5) profile clustering + per-cluster multiviz (Fisher strips, PCA, boxplots, merged term CSVs); (6) contrast specificity + tail Fisher tests; (7–14) **extensions**: provenance manifest, protein-centric summaries, direction-stratified term Fisher, S/T/Y parsing, optional UniProt/motif, optional preranked GSEA, optional kinase–substrate map z-scores, co-regulation + optional STRING + optional UpSet.

**Data:** expects **`data.csv`** (tab-separated) next to this notebook or discoverable by walking parents (`find_data_csv()`). Opinionated defaults (**thresholds, clustering, Fisher, plot DPI, …**) are **`UPPER_SNAKE` constants** at the top of the first code cell; **`RUN_CONFIG`** mirrors the significance keys for manifests and downstream cells (overlap uses `q_max_sig` / `abs_lfc_sig`; volcano uses `q_max_volcano` / `abs_lfc_volcano`).

**Outputs:** CSVs and PNGs under **`outputs_all/`** beside `data.csv` (e.g. `v3/outputs_all/...`).

**Dependencies (core):** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, **`hdbscan`** (for default clustering; use `CLUSTER_METHOD = "kmeans"` in section 04 if missing).

**Optional (sections skip gracefully if missing):** `requests` (UniProt + STRING), **`tqdm`** (progress in §11), `logomaker` (sequence logo), `gseapy` + **GMT** (§12: runs **`gseapy.prerank`** once per contrast **per `*.gmt` file** next to `data.csv`), `networkx`, `upsetplot`, **`kinase_substrate_map.csv`** (§13 / §13a — PhosphoSitePlus `Kinase_Substrate_Dataset` or manual). See **`v3/requirements.txt`**."""
        )
    )

    cells.append(
        md(
            """### Embedded libraries + load

**What this does:** Sets **global defaults** (thresholds, clustering, Fisher limits, plot DPI, …) then defines contrast metadata (`CONTRASTS`), Benjamini–Hochberg FDR (`q_*` columns), table loading (`load_data_csv`), term parsing and **Fisher** tests vs the universe, and **HDBSCAN/k-means** clustering on standardized shape profiles. Resolves **`DATA`** and creates **`OUT_ROOT`**, then loads **`df`**.

**Why:** Keeps the notebook copy-pasteable without the rest of the repository.

**Interpret:** After this cell, `df` is the full site table with FDR; downstream cells filter to rows with finite Log2FC in all eight contrasts where noted."""
        )
    )
    cells.append(code(LIBS_AND_LOAD))

    # --- 01 ---
    cells.append(
        md(
            """### 01 — Contrast summary table

**Method:** Keep sites with finite Log2FC in all contrasts; per contrast compute mean/std of Log2FC, median raw **−log10(p)**, and median **BH q**.

**Interpret:** Central tendency and spread of effects; low **q_median** suggests many sites pass FDR for that stimulation. Saved to `outputs_all/01_load_and_qc/contrast_summary.csv`."""
        )
    )
    cells.append(
        code(
            r"""
OUT01 = OUT_ROOT / "01_load_and_qc"
OUT01.mkdir(parents=True, exist_ok=True)

mask = finite_rows_mask(df)
print("Rows with finite Log2FC across all contrasts:", int(mask.sum()), "/", len(df))
bad = df.loc[~mask]
if len(bad):
    print("Example rows with missing/inf:", bad[["id"]].head())

df_q = df.loc[mask].copy()
summary = []
for c in CONTRASTS:
    summary.append({
        "contrast": c.key,
        "label": c.label,
        "log2fc_mean": float(df_q[c.log2fc_col].mean()),
        "log2fc_std": float(df_q[c.log2fc_col].std()),
        "neglog10p_median": float(df_q[c.neglog10_p_col].median()),
        "q_median": float(df_q[f"q_{c.key}"].median()),
    })
sum_df = pd.DataFrame(summary)
display(sum_df)
sum_df.to_csv(OUT01 / "contrast_summary.csv", index=False)
print("Saved", OUT01 / "contrast_summary.csv")
""".strip()
        )
    )

    cells.append(
        md(
            """### 01 — Log2FC histograms

**Method:** Marginal histogram of **Log2FC** per contrast (finite cohort only).

**Interpret:** Bulk shift (peak left/right of 0) and heterogeneity (width). No significance information."""
        )
    )
    cells.append(
        code(
            r"""
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for ax, c in zip(axes.ravel(), CONTRASTS):
    df_q[c.log2fc_col].hist(bins=40, ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(c.label, fontsize=8)
    ax.set_xlabel("Log2FC")
plt.suptitle("Log2FC distributions (finite rows)")
plt.tight_layout()
fig.savefig(OUT01 / "hist_log2fc_by_contrast.png", dpi=FIG_DPI_DEFAULT, bbox_inches="tight")
plt.show()
""".strip()
        )
    )

    cells.append(
        md(
            """### 01 — FDR *q* histograms

**Method:** Distribution of **q** (BH-FDR) per contrast.

**Interpret:** Mass near 1 ⇒ little adjusted significance; mass near 0 ⇒ many strong hits. Pair with Log2FC or volcano plots."""
        )
    )
    cells.append(
        code(
            r"""
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for ax, c in zip(axes.ravel(), CONTRASTS):
    df_q[f"q_{c.key}"].clip(0, 1).hist(bins=40, ax=ax, color="coral", edgecolor="white")
    ax.set_title(c.label, fontsize=8)
    ax.set_xlabel("q (BH)")
plt.suptitle("FDR q distributions")
plt.tight_layout()
fig.savefig(OUT01 / "hist_q_by_contrast.png", dpi=FIG_DPI_DEFAULT, bbox_inches="tight")
plt.show()
""".strip()
        )
    )

    cells.append(
        md(
            """### 01 — Volcano plots (Log2FC vs −log10 *q*)

**Method:** Scatter **x = Log2FC**, **y = −log10(q)**; highlight **q < Q_MAX**; dotted verticals at **±|Log2FC|** from **`Q_MAX_VOLCANO` / `ABS_LFC_VOLCANO`** (see first code cell).

**Interpret:** Upper corners = large effects with strong FDR; points along bottom = not significant after multiple testing."""
        )
    )
    cells.append(
        code(
            r"""
Q_MAX = RUN_CONFIG["q_max_volcano"]
ABS_LFC_MIN = RUN_CONFIG["abs_lfc_volcano"]

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, c in zip(axes.ravel(), CONTRASTS):
    x = df_q[c.log2fc_col].astype(float).to_numpy()
    qv = np.clip(df_q[f"q_{c.key}"].astype(float).to_numpy(), 1e-300, 1.0)
    y = -np.log10(qv)
    sig = qv < Q_MAX
    ax.scatter(x[~sig], y[~sig], s=6, alpha=0.28, c="#94a3b8", edgecolors="none", rasterized=True)
    ax.scatter(x[sig], y[sig], s=8, alpha=0.45, c="#1d4ed8", edgecolors="none", rasterized=True)
    ax.axhline(-np.log10(Q_MAX), color="k", ls="--", lw=0.85, alpha=0.65)
    ax.axvline(ABS_LFC_MIN, color="k", ls=":", lw=0.7, alpha=0.5)
    ax.axvline(-ABS_LFC_MIN, color="k", ls=":", lw=0.7, alpha=0.5)
    ax.set_title(c.label, fontsize=8)
    ax.set_xlabel("Log2FC")
    ax.set_ylabel(r"$-\log_{10}(q)$")
plt.suptitle(f"Volcano (FDR): Log2FC vs −log10(q); q<{Q_MAX} blue; |Log2FC|={ABS_LFC_MIN} dotted")
plt.tight_layout()
fig.savefig(OUT01 / "volcano_log2fc_vs_neglog10q_by_contrast.png", dpi=FIG_DPI_DEFAULT, bbox_inches="tight")
plt.show()
""".strip()
        )
    )

    # --- 02 ---
    cells.append(
        md(
            """### 02 — Contrast similarity

**Method:** Pearson correlation of **Log2FC vectors across sites** between each pair of contrasts; heatmap + hierarchical clustering of contrasts (distance = 1 − r, average linkage).

**Interpret:** High *r* = coordinated regulation across those stimulations; the dendrogram groups similar contrast profiles."""
        )
    )
    cells.append(
        code(
            r"""
OUT02 = OUT_ROOT / "02_contrast_similarity"
OUT02.mkdir(parents=True, exist_ok=True)

mask = finite_rows_mask(df)
X = log2fc_matrix_from_df(df.loc[mask])
labels = [c.label for c in CONTRASTS]
keys = [c.key for c in CONTRASTS]
C = X.corr(method="pearson")
fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(C, xticklabels=labels, yticklabels=labels, cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax)
ax.set_title("Pearson correlation of Log2FC vectors (across sites)")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
C.to_csv(OUT02 / "contrast_pearson_corr.csv")
fig.savefig(OUT02 / "contrast_pearson_heatmap.png", dpi=FIG_DPI_HEATMAP, bbox_inches="tight")
plt.show()

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

d = 1.0 - np.clip(C.values, -1, 1)
np.fill_diagonal(d, 0)
cond = squareform(d, checks=False)
Z = linkage(cond, method="average")
fig, ax = plt.subplots(figsize=(9, 4))
dendrogram(Z, labels=labels, ax=ax, leaf_rotation=45)
ax.set_title("Contrast clustering (1 − Pearson correlation, average linkage)")
plt.tight_layout()
fig.savefig(OUT02 / "contrast_dendrogram.png", dpi=FIG_DPI_HEATMAP, bbox_inches="tight")
plt.show()
""".strip()
        )
    )

    # --- 02 outliers (abbreviated section headers from v2) ---
    cells.append(
        md(
            """### 02b — Multivariate outliers / discordance

**Method:** On the **8 × Log2FC** matrix (finite rows): (1) classical Mahalanobis vs sample covariance; (2) robust MCD Mahalanobis; (3) PCA reconstruction error on standardized data (*k*=2,3); (4) max pairwise OLS residual over all contrast pairs; (5–6) Isolation Forest and LOF; (7) row-wise contrast **specificity** (max *z* − median of other seven).

**Interpret:** Complementary “off bulk pattern” scores — prefer consensus across methods for follow-up; inspect raw profiles for candidates."""
        )
    )
    cells.append(
        code(
            r"""
OUT02b = OUT_ROOT / "02_contrast_similarity_outliers"
OUT02b.mkdir(parents=True, exist_ok=True)

mask = finite_rows_mask(df)
sub_o = df.loc[mask].copy()
X_df = log2fc_matrix_from_df(sub_o)
X = X_df.to_numpy(dtype=float)
n, d = X.shape
keys = [c.key for c in CONTRASTS]
labels = [c.label for c in CONTRASTS]

C = X_df.corr(method="pearson")
fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(C, xticklabels=labels, yticklabels=labels, cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax)
ax.set_title("Pearson r of Log2FC vectors (context)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
C.to_csv(OUT02b / "contrast_pearson_corr_context.csv")
fig.savefig(OUT02b / "heatmap_contrast_pearson_context.png", dpi=FIG_DPI_HEATMAP, bbox_inches="tight")
plt.show()
print("n_sites, n_contrasts =", n, d)
""".strip()
        )
    )

    cells.append(
        md(
            """**Classical Mahalanobis** — squared distance from the mean using pooled covariance (Moore–Penrose inverse); χ² 99% reference on *d*².

**Interpret:** Large = far from the average 8D profile; sensitive to covariance estimate and non-Gaussian tails."""
        )
    )
    cells.append(
        code(
            r"""
from scipy.stats import chi2

mu = X.mean(axis=0, keepdims=True)
Xm = X - mu
cov = np.cov(X, rowvar=False) + np.eye(d) * 1e-8
inv = np.linalg.pinv(cov)
q = Xm @ inv
mahal_class_sq = np.sum(Xm * q, axis=1)
mahal_class_sq = np.clip(mahal_class_sq, 0, None)
mahal_class = np.sqrt(mahal_class_sq)
chi2_thr = float(chi2.ppf(MAHAL_CHI2_QUANTILE, df=d))
exceed_chi = mahal_class_sq > chi2_thr
print("chi2(%.4f, df=%d) threshold on d^2 = %.4f" % (MAHAL_CHI2_QUANTILE, d, chi2_thr))
""".strip()
        )
    )

    cells.append(
        md(
            """**Robust Mahalanobis (MinCovDet)** — down-weight leverage points when estimating center/covariance.

**Interpret:** Can disagree with classical Mahal when a few sites distort Σ̂."""
        )
    )
    cells.append(
        code(
            r"""
from sklearn.covariance import MinCovDet

try:
    mcd = MinCovDet(support_fraction=MINCOV_DET_SUPPORT_FRACTION, random_state=RNG).fit(X)
    mr = mcd.mahalanobis(X)
    mr = np.asarray(mr, dtype=float)
    mahal_rob = np.sqrt(np.maximum(mr, 0))
    print("MinCovDet OK")
except Exception as ex:
    print("MinCovDet failed:", ex)
    mahal_rob = np.full(n, np.nan)
""".strip()
        )
    )

    cells.append(
        md(
            """**PCA reconstruction residual** — column-standardize *X*, fit PCA with *k* components, Euclidean norm of residual.

**Interpret:** Large = site not explained by the leading joint modes of variation across contrasts."""
        )
    )
    cells.append(
        code(
            r"""
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sc = StandardScaler()
Xs = sc.fit_transform(X)
pca_resid = {}
for kcom in (2, 3):
    pca = PCA(n_components=kcom, random_state=RNG).fit(Xs)
    recon = pca.inverse_transform(pca.transform(Xs))
    pca_resid[kcom] = np.linalg.norm(Xs - recon, axis=1)
    print(f"PCA k={kcom}: explained variance ratio sum = {pca.explained_variance_ratio_.sum():.4f}")
""".strip()
        )
    )

    cells.append(
        md(
            """**Pairwise OLS residuals** — for each contrast pair, both regression directions; per site take max |residual|; track winning pair.

**Interpret:** Flags sites that break the local linear coupling between two contrasts."""
        )
    )
    cells.append(
        code(
            r"""
max_abs = np.zeros(n)
pair_a = np.array([""] * n, dtype=object)
pair_b = np.array([""] * n, dtype=object)
for i in range(d):
    for j in range(i + 1, d):
        xi = X[:, i]
        xj = X[:, j]
        A1 = np.column_stack([np.ones(n), xi])
        b1, *_ = np.linalg.lstsq(A1, xj, rcond=None)
        r1 = np.abs(xj - A1 @ b1)
        A2 = np.column_stack([np.ones(n), xj])
        b2, *_ = np.linalg.lstsq(A2, xi, rcond=None)
        r2 = np.abs(xi - A2 @ b2)
        r = np.maximum(r1, r2)
        better = r > max_abs
        max_abs = np.where(better, r, max_abs)
        use_j_on_i = r1 >= r2
        pa = np.where(use_j_on_i, keys[i], keys[j])
        pb = np.where(use_j_on_i, keys[j], keys[i])
        pair_a = np.where(better, pa, pair_a)
        pair_b = np.where(better, pb, pair_b)

pairwise_long_rows = []
for i in range(d):
    for j in range(i + 1, d):
        xi = X[:, i]
        xj = X[:, j]
        A1 = np.column_stack([np.ones(n), xi])
        b1, *_ = np.linalg.lstsq(A1, xj, rcond=None)
        r1 = xj - A1 @ b1
        A2 = np.column_stack([np.ones(n), xj])
        b2, *_ = np.linalg.lstsq(A2, xi, rcond=None)
        r2 = xi - A2 @ b2
        rmax = np.maximum(np.abs(r1), np.abs(r2))
        sgn = np.where(np.abs(r1) >= np.abs(r2), np.sign(r1), np.sign(r2))
        for t in range(n):
            pairwise_long_rows.append(
                {
                    "id": str(sub_o.iloc[t]["id"]),
                    "geneid": str(sub_o.iloc[t].get("geneid", "")),
                    "contrast_a": keys[i],
                    "contrast_b": keys[j],
                    "signed_residual_dom": float(sgn[t] * rmax[t]),
                    "abs_residual_max_dir": float(rmax[t]),
                }
            )
pair_long = pd.DataFrame(pairwise_long_rows)
pair_long.to_csv(OUT02b / "pairwise_ols_residuals_long.csv", index=False)
print("Saved", OUT02b / "pairwise_ols_residuals_long.csv")
""".strip()
        )
    )

    cells.append(
        md(
            """**Isolation Forest & LOF** — unsupervised anomaly scores on standardized *Xs* (fixed contamination).

**Interpret:** Higher LOF score here = more outlier-like (see code for sign convention on NOF)."""
        )
    )
    cells.append(
        code(
            r"""
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

iso = IsolationForest(
    random_state=RNG, contamination=ISOFOREST_CONTAMINATION, n_estimators=ISOFOREST_N_ESTIMATORS
).fit(Xs)
iforest_decision = iso.decision_function(Xs)
iforest_outlier = (iso.predict(Xs) == -1).astype(int)

lof = LocalOutlierFactor(n_neighbors=LOF_N_NEIGHBORS, contamination=LOF_CONTAMINATION, novelty=False)
y_lof = lof.fit_predict(Xs)
nof = np.asarray(lof.negative_outlier_factor_, dtype=float)
lof_anomaly_score = -nof
lof_outlier = (y_lof == -1).astype(int)
""".strip()
        )
    )

    cells.append(
        md(
            """**Contrast specificity (row *z*)** — same construction as section 06: *z* across contrasts per site; specificity = *z*ₘₐₓ − median(rest).

**Interpret:** One contrast driving the site vs a shared multi-contrast shift."""
        )
    )
    cells.append(
        code(
            r"""
Zr = (X - np.nanmean(X, axis=1, keepdims=True)) / (np.nanstd(X, axis=1, keepdims=True) + 1e-9)
spec_rows = []
for t in range(n):
    z = Zr[t]
    j_max = int(np.argmax(z))
    z_other = np.delete(z, j_max)
    spec = float(z[j_max] - np.median(z_other))
    spec_rows.append({"argmax_contrast": keys[j_max], "specificity": spec})
spec_part = pd.DataFrame(spec_rows)
""".strip()
        )
    )

    cells.append(
        md(
            """**Merge scores + quick QC plots** — one wide CSV; histograms and scatter checks.

**Interpret:** Sites extreme on several metrics are stronger candidates than any single score."""
        )
    )
    cells.append(
        code(
            r"""
out = pd.DataFrame(
    {
        "id": sub_o["id"].astype(str).values,
        "geneid": sub_o["geneid"].astype(str).values if "geneid" in sub_o.columns else "",
        "mahal_class_sq": mahal_class_sq,
        "mahal_class": mahal_class,
        "mahal_class_sq_exceeds_chi2_99": exceed_chi.astype(int),
        "mahal_robust": mahal_rob,
        "pca_resid_norm_k2": pca_resid[2],
        "pca_resid_norm_k3": pca_resid[3],
        "pairwise_max_abs_resid": max_abs,
        "pairwise_max_pred": pair_b,
        "pairwise_max_from": pair_a,
        "iforest_decision": iforest_decision,
        "iforest_outlier": iforest_outlier,
        "lof_anomaly_score": lof_anomaly_score,
        "lof_outlier": lof_outlier,
    }
)
out = pd.concat([out.reset_index(drop=True), spec_part.reset_index(drop=True)], axis=1)
out.to_csv(OUT02b / "site_multivariate_outlier_scores.csv", index=False)
print("Saved", OUT02b / "site_multivariate_outlier_scores.csv")

fig, axes = plt.subplots(2, 4, figsize=(14, 6))
cols_plot = [
    "mahal_class",
    "mahal_robust",
    "pca_resid_norm_k2",
    "pca_resid_norm_k3",
    "pairwise_max_abs_resid",
    "iforest_decision",
    "lof_anomaly_score",
    "specificity",
]
titles = [
    "Mahalanobis (classical)",
    "Robust MCD",
    "PCA resid k=2",
    "PCA resid k=3",
    "Max |pairwise OLS res.|",
    "IF decision (high=inlier)",
    "LOF −NOF (high=outlier)",
    "Specificity",
]
for ax, col, tt in zip(axes.ravel(), cols_plot, titles):
    s = out[col].replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        ax.set_title(tt + " (empty)")
        continue
    hi = s.quantile(OUTLIER_SCORE_DISPLAY_Q)
    s2 = s.clip(upper=float(hi))
    s2.hist(bins=40, ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(tt, fontsize=8)
    ax.set_ylabel("count")
plt.suptitle("Distributions of outlier / discordance scores (winsor 99.5% for display)")
plt.tight_layout()
fig.savefig(OUT02b / "hist_all_scores.png", dpi=FIG_DPI_DEFAULT, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
a = out["mahal_class"].values
b = out["mahal_robust"].values
m = np.isfinite(a) & np.isfinite(b)
axes[0].scatter(a[m], b[m], s=10, alpha=0.35, edgecolors="none")
axes[0].set_xlabel("Mahalanobis classical")
axes[0].set_ylabel("Mahalanobis robust (MCD)")
axes[0].set_title("Classical vs robust")

axes[1].scatter(out["mahal_class"].values, out["pca_resid_norm_k2"].values, s=10, alpha=0.35, edgecolors="none")
axes[1].set_xlabel("Mahalanobis classical")
axes[1].set_ylabel("PCA recon error (k=2)")
axes[1].set_title("Shape vs low-rank deviation")
plt.tight_layout()
fig.savefig(OUT02b / "scatter_mahal_vs_robust_pca.png", dpi=FIG_DPI_DEFAULT, bbox_inches="tight")
plt.show()

top = out.sort_values("mahal_class", ascending=False).head(OUTLIER_TOP_TABLE_N)
print(f"Top {OUTLIER_TOP_TABLE_N} by classical Mahalanobis:")
display(top[["id", "geneid", "mahal_class", "mahal_robust", "pca_resid_norm_k2", "pairwise_max_abs_resid", "specificity", "argmax_contrast"]])
""".strip()
        )
    )

    # --- 03 ---
    cells.append(
        md(
            """### 03 — Significant overlap across contrasts

**Method:** Define significant sites per contrast (**q < Q_MAX** and **|Log2FC| ≥ ABS_LFC_MIN**). Jaccard between sets; bar chart of how many contrasts each site hits.

**Interpret:** Jaccard shows shared regulation programs; the count histogram shows pleiotropic vs contrast-specific hits."""
        )
    )
    cells.append(
        code(
            r"""
OUT03 = OUT_ROOT / "03_significant_overlap"
OUT03.mkdir(parents=True, exist_ok=True)

Q_MAX = RUN_CONFIG["q_max_sig"]
ABS_LFC_MIN = RUN_CONFIG["abs_lfc_sig"]

sets = {}
for c in CONTRASTS:
    sig = (df[f"q_{c.key}"] < Q_MAX) & (df[c.log2fc_col].abs() >= ABS_LFC_MIN)
    sets[c.key] = set(df.loc[sig, "id"].astype(str))
keys = [c.key for c in CONTRASTS]
labels = [c.label for c in CONTRASTS]
m = np.zeros((len(keys), len(keys)))
for i, a in enumerate(keys):
    for j, b in enumerate(keys):
        A, B = sets[a], sets[b]
        u = len(A | B)
        m[i, j] = len(A & B) / u if u else 0.0
jac = pd.DataFrame(m, index=labels, columns=labels)
jac.to_csv(OUT03 / "jaccard_significant_site_sets.csv")
fig, ax = plt.subplots(figsize=(7.5, 6.5))
sns.heatmap(jac, vmin=0, vmax=1, cmap="mako", ax=ax)
ax.set_title(f"Jaccard overlap of significant sites (q<{Q_MAX}, |LFC|≥{ABS_LFC_MIN})")
plt.tight_layout()
fig.savefig(OUT03 / "jaccard_significant_site_sets.png", dpi=FIG_DPI_HEATMAP, bbox_inches="tight")
plt.show()

id_to_n = {}
for sid in df["id"].astype(str):
    nn = sum(1 for k in keys if sid in sets[k])
    if nn > 0:
        id_to_n[sid] = nn
cnt = Counter(id_to_n.values())
xs = sorted(cnt.keys())
ys = [cnt[k] for k in xs]
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar([str(k) for k in xs], ys, color="teal")
ax.set_xlabel("Number of contrasts significant")
ax.set_ylabel("Number of sites")
ax.set_title("Multi-contrast significance count")
plt.tight_layout()
fig.savefig(OUT03 / "sig_count_per_site_bar.png", dpi=FIG_DPI_DEFAULT, bbox_inches="tight")
plt.show()
pd.Series({k: len(sets[k]) for k in keys}, name="n_sig_sites").to_csv(OUT03 / "n_significant_per_contrast.csv")
print("Saved outputs to", OUT03)
""".strip()
        )
    )

    # --- 04 clustering + 05 per-cluster multiviz ---
    cells.append(
        md(
            """### 04 — Profile groups (clustering + heatmap)

**Method:** Row-mean center **Log2FC** across contrasts, then column **`StandardScaler`**; **HDBSCAN** (or **k-means**) on that matrix. Heatmap of **median raw Log2FC** per cluster × contrast; save **`site_cluster_assignments.csv`**.

**Interpret:** Clusters = sites with similar *shape* of response across the eight contrasts (not magnitude-only). Noise cluster (−1) may appear for HDBSCAN.

**Downstream:** Section **05** reuses **`sub`** and **`cluster`** from this cell — do not re-run clustering there."""
        )
    )
    cells.append(
        code(
            r"""
OUT04 = OUT_ROOT / "04_profile_groups"
OUT04.mkdir(parents=True, exist_ok=True)

from sklearn.preprocessing import StandardScaler
"""
            + "\n"
            + V2_CLUSTER_FIT.strip()
            + "\n"
            + r"""

sub[["id", "geneid", "cluster"]].to_csv(OUT04 / "site_cluster_assignments.csv", index=False)

cl_pos = sorted(int(x) for x in sub["cluster"].unique() if int(x) >= 0)
cl_order = list(cl_pos)
if (sub["cluster"] == -1).any():
    cl_order.append(-1)
if not cl_pos:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.text(0.5, 0.55, "HDBSCAN: no dense clusters (all noise).", ha="center", fontsize=12, transform=ax.transAxes)
    ax.text(
        0.5,
        0.38,
        "Try CLUSTER_METHOD='kmeans' or set HDBSCAN_MIN_CLUSTER_SIZE / HDBSCAN_MIN_SAMPLES.",
        ha="center",
        fontsize=10,
        transform=ax.transAxes,
    )
    ax.axis("off")
    ax.set_title("Median Log2FC per cluster (no clusters extracted)")
else:
    mat = sub.groupby("cluster")[[c.log2fc_col for c in CONTRASTS]].median().reindex(cl_order)
    mat.index = [("noise" if int(i) == -1 else f"c{int(i)}") for i in mat.index]
    mat.columns = [c.label for c in CONTRASTS]
    fig_h = max(4, 0.45 * len(mat))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    sns.heatmap(mat, cmap="vlag", center=0, ax=ax)
    ax.set_title("Median raw Log2FC per cluster × contrast")
plt.tight_layout()
fig.savefig(OUT04 / "cluster_contrast_median_heatmap.png", dpi=FIG_DPI_HEATMAP, bbox_inches="tight")
plt.show()
print("Saved", OUT04 / "site_cluster_assignments.csv")
""".strip()
        )
    )

    cells.append(
        md(
            """### 05 — Per-cluster **multiviz** (inline figures)

**Method:** For each cluster: **Fisher’s exact test** (term in cluster vs rest of finite-row sites) for **GOBP / GOMF / GOCC / KEGG** shown as four strip panels (−log10 *p*); **median Log2FC ± IQR** across contrasts; **PCA** on raw 8×Log2FC within cluster; **boxplots**; text listing **site ids** and **genes** (lists omitted for large noise cluster). Writes **`fisher_merged_<cluster>.csv`** plus **`cluster_<cluster>_multiviz.png`** under `outputs_all/05_terms_by_cluster_details/`.

**Interpret:** One-stop view of cluster phenotype, dispersion, and annotation enrichment."""
        )
    )
    cells.append(
        code(
            r"""
from matplotlib import patheffects as pe
from matplotlib.gridspec import GridSpecFromSubplotSpec
from sklearn.decomposition import PCA

OUT05b = OUT_ROOT / "05_terms_by_cluster_details"
OUT05b.mkdir(parents=True, exist_ok=True)

terms_long = parse_terms_long(sub)
universe_ids = set(sub["id"].astype(str))

MIN_SUPPORT = FISHER_MIN_SUPPORT_MULTIVIZ
TOP_TERMS_PER_LAYER = FISHER_TOP_TERMS_MULTIVIZ
MAX_SITES_LISTED = MULTIVIZ_MAX_SITES_LISTED
MAX_GENE_CHARS = MULTIVIZ_MAX_GENE_CHARS


def _cl_title_d(k):
    return "noise" if k == -1 else f"c{k}"


def _draw_fisher_layer_axes(ax, layer_name, res_df, xmax_global, top_n):
    ax.set_facecolor("#eef2f6")
    ax.tick_params(axis="both", labelsize=7)
    if res_df is None or len(res_df) == 0:
        ax.text(0.5, 0.5, f"{layer_name}: no terms", ha="center", va="center", transform=ax.transAxes, fontsize=9, color="#2d3748")
        ax.set_axis_off()
        return
    take = res_df.head(min(top_n, len(res_df))).reset_index(drop=True)
    vals = -np.log10(take["p_fisher"].clip(1e-300).astype(float)).values
    terms = take["term"].astype(str).tolist()
    y = np.arange(len(take))
    bar_color = "#2c5282"
    ax.barh(y, vals, height=0.68, color=bar_color, edgecolor="#ffffff", linewidth=0.45, zorder=1)
    xmax = float(xmax_global)
    ax.set_xlim(0, xmax)
    ax.set_yticks([])
    ax.set_xlabel("-log10 Fisher p", fontsize=8)
    ax.set_title(f"{layer_name}  (Fisher vs rest)", fontsize=9, fontweight="bold", color="#1a365d")
    thr = 0.20 * xmax
    stroke = [pe.withStroke(linewidth=2.25, foreground="#1a365d", alpha=0.85)]
    for yi, (w, raw_t) in enumerate(zip(vals, terms)):
        disp = raw_t if len(raw_t) <= 160 else raw_t[: 157] + "…"
        if w >= thr:
            tx = max(w * 0.5, 0.035 * xmax)
            t = ax.text(tx, yi, disp, ha="center", va="center", fontsize=6.5, color="#f8fafc", fontweight="medium", zorder=4, clip_on=True)
            t.set_path_effects(stroke)
        else:
            t = ax.text(min(w + 0.018 * xmax, 0.97 * xmax), yi, disp, ha="left", va="center", fontsize=6.5, color="#1a202c", fontweight="normal", zorder=4)
            t.set_path_effects([pe.withStroke(linewidth=2.0, foreground="#f8fafc", alpha=0.95)])
    ax.invert_yaxis()


def _build_cluster_figure(k, sub_k, terms_long, universe_ids, top_n, min_support, rng_seed):
    n_sites = len(sub_k)
    qids = set(sub_k["id"].astype(str))
    genes = sub_k["geneid"].astype(str)
    uniq = sorted(genes.unique())
    ids_sorted = sorted(sub_k["id"].astype(str).tolist())

    if k == -1:
        text_body = (
            f"HDBSCAN noise cluster (label −1). Site id and gene symbol lists are omitted here because this group "
            f"often contains very many rows and would dominate the figure.\n\n"
            f"n_sites = {n_sites}  |  n_unique_genes = {len(uniq)}"
        )
        bottom_ratio = 0.12
    else:
        if len(ids_sorted) > MAX_SITES_LISTED:
            sites_block = ", ".join(ids_sorted[:MAX_SITES_LISTED]) + f"\n… (+{len(ids_sorted) - MAX_SITES_LISTED} more site ids)"
        else:
            sites_block = ", ".join(ids_sorted)
        gene_joined = ", ".join(uniq)
        if len(gene_joined) > MAX_GENE_CHARS:
            gene_joined = gene_joined[: MAX_GENE_CHARS - 40] + f"\n… (truncated; {len(uniq)} unique gene symbols total)"
        text_body = f"SITE IDS (n={n_sites})\n\nGENE SYMBOLS (n_unique={len(uniq)})\n\n" + sites_block + "\n\n" + gene_joined
        bottom_ratio = 0.36

    Xk = log2fc_matrix_from_df(sub_k).to_numpy(dtype=float)
    max_abs = np.nanmax(np.abs(Xk), axis=1)

    fig = plt.figure(figsize=(16, 18))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.65, 1.12, bottom_ratio], hspace=0.36, wspace=0.26)
    fig.suptitle(
        f"Cluster {_cl_title_d(k)}  |  n_sites={n_sites}  |  n_genes_unique={genes.nunique()}",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    ax_med = fig.add_subplot(gs[0, 0])
    cols = [c.log2fc_col for c in CONTRASTS]
    med = sub_k[cols].median()
    q25 = sub_k[cols].quantile(0.25)
    q75 = sub_k[cols].quantile(0.75)
    x = np.arange(len(CONTRASTS))
    ax_med.bar(x, med.values, yerr=[(med - q25).values, (q75 - med).values], capsize=3, color="steelblue", alpha=0.88)
    ax_med.set_xticks(x)
    ax_med.set_xticklabels([c.label for c in CONTRASTS], rotation=45, ha="right", fontsize=8)
    ax_med.axhline(0, color="k", lw=0.6)
    ax_med.set_ylabel("Log2FC")
    ax_med.set_title("Median Log2FC ± IQR (within cluster)")

    layer_order = list(ANNOTATION_LAYERS.keys())
    xmax_global = 0.35
    pre_res = {}
    for layer in layer_order:
        ms = max(3, min(min_support, max(3, n_sites // 4)))
        r0 = fisher_terms_vs_rest(qids, universe_ids, terms_long, layer, min_support=ms, top_n=top_n)
        pre_res[layer] = r0
        if len(r0) > 0:
            xmax_global = max(xmax_global, float((-np.log10(r0["p_fisher"].clip(1e-300))).max()) * 1.12)
    xmax_global = max(xmax_global, 0.35)

    subgs = GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0, 1], hspace=0.52, height_ratios=[1, 1, 1, 1])
    for li, layer in enumerate(layer_order):
        axf = fig.add_subplot(subgs[li, 0])
        r0 = pre_res[layer]
        _draw_fisher_layer_axes(axf, layer, r0 if not r0.empty else None, xmax_global, top_n)

    ax_pca = fig.add_subplot(gs[1, 0])
    if n_sites >= 3:
        pc = PCA(n_components=2, random_state=rng_seed).fit_transform(Xk)
        sz = int(min(120, max(25, 800 // max(n_sites, 1))))
        sc = ax_pca.scatter(pc[:, 0], pc[:, 1], c=max_abs, cmap="magma", s=sz, alpha=0.88, edgecolors="k", linewidths=0.25)
        plt.colorbar(sc, ax=ax_pca, fraction=0.046, label="max |Log2FC|")
        ax_pca.set_xlabel("PC1 (8 contrasts)")
        ax_pca.set_ylabel("PC2")
        ax_pca.set_title("Sites — PCA on raw 8× Log2FC")
    elif n_sites == 2:
        pc1 = PCA(n_components=1, random_state=rng_seed).fit_transform(Xk).ravel()
        ax_pca.scatter(pc1, [0, 1], c=max_abs, cmap="magma", s=80, edgecolors="k")
        ax_pca.set_yticks([])
        ax_pca.set_xlabel("PC1 (8 contrasts)")
        ax_pca.set_title("Two sites — 1D PCA scores")
    else:
        ax_pca.text(0.5, 0.5, "Single site (no PCA cloud)", ha="center", va="center", transform=ax_pca.transAxes)

    ax_box = fig.add_subplot(gs[1, 1])
    plot_data = [sub_k[c.log2fc_col].astype(float).values for c in CONTRASTS]
    bp = ax_box.boxplot(plot_data, patch_artist=True)
    ax_box.set_xticks(np.arange(1, len(CONTRASTS) + 1))
    for p in bp["boxes"]:
        p.set_facecolor("lightsteelblue")
    ax_box.axhline(0, color="k", lw=0.6)
    ax_box.set_xticklabels([c.label for c in CONTRASTS], rotation=45, ha="right", fontsize=7)
    ax_box.set_ylabel("Log2FC")
    ax_box.set_title("Per-contrast spread (boxplot)")

    ax_txt = fig.add_subplot(gs[2, :])
    ax_txt.axis("off")
    ax_txt.text(0.01, 0.98, text_body, transform=ax_txt.transAxes, fontsize=7 if k != -1 else 9, va="top", family="monospace" if k != -1 else "sans-serif")

    return fig


plot_ids = sorted([int(c) for c in np.unique(sub["cluster"]) if c >= 0])
if -1 in sub["cluster"].values and (sub["cluster"] == -1).sum() >= 3:
    plot_ids.append(-1)

for k in plot_ids:
    m = sub["cluster"] == k
    sub_k = sub.loc[m]
    n_sites = len(sub_k)
    tag = "noise" if k == -1 else str(k)
    if n_sites < 1:
        continue
    qids = set(sub_k["id"].astype(str))
    fisher_rows = []
    for layer in ANNOTATION_LAYERS:
        ms = max(3, min(MIN_SUPPORT, max(3, n_sites // 4)))
        res = fisher_terms_vs_rest(qids, universe_ids, terms_long, layer, min_support=ms, top_n=TOP_TERMS_PER_LAYER)
        if res.empty:
            continue
        for _, r in res.iterrows():
            fisher_rows.append(
                {
                    "layer": layer,
                    "term": r["term"],
                    "p_fisher": float(r["p_fisher"]),
                    "neglog10": float(-np.log10(max(float(r["p_fisher"]), 1e-300))),
                }
            )
    fisher_df = pd.DataFrame(fisher_rows)
    if len(fisher_df) > 0:
        fisher_df = fisher_df.sort_values("p_fisher").head(MULTIVIZ_FISHER_MERGED_CAP)
        fisher_df.to_csv(OUT05b / f"fisher_merged_{tag}.csv", index=False)
    else:
        fisher_df = pd.DataFrame(columns=["layer", "term", "p_fisher", "neglog10"])

    display(Markdown(f"### Cluster {_cl_title_d(k)}  (n_sites={n_sites})"))
    fig_c = _build_cluster_figure(k, sub_k, terms_long, universe_ids, TOP_TERMS_PER_LAYER, MIN_SUPPORT, RNG)
    png_path = OUT05b / f"cluster_{tag}_multiviz.png"
    fig_c.savefig(png_path, dpi=FIG_DPI_CLUSTER_MULTIVIZ, bbox_inches="tight")
    display(fig_c)
    plt.close(fig_c)
    print("Saved", png_path)

print("Done. Outputs in", OUT05b)
""".strip()
        )
    )

    # --- 06 ---
    cells.append(
        md(
            """### 06 — Contrast specificity and tail terms

**Method:** Row-wise *z* of Log2FC across contrasts; **specificity** = max *z* − median(other seven). Top-decile tail per **argmax** contrast; **Fisher** on **GOBP** and **KEGG** for that tail vs universe.

**Interpret:** High specificity = driven mainly by one contrast; tail enrichment suggests pathways/terms overrepresented among contrast-specific changers."""
        )
    )
    cells.append(
        code(
            r"""
OUT06 = OUT_ROOT / "06_contrast_specificity"
OUT06.mkdir(parents=True, exist_ok=True)

mask = finite_rows_mask(df)
sub6 = df.loc[mask].copy()
X6 = log2fc_matrix_from_df(sub6).to_numpy(dtype=float)
Z = (X6 - np.nanmean(X6, axis=1, keepdims=True)) / (np.nanstd(X6, axis=1, keepdims=True) + 1e-9)
spec_rows = []
for i in range(len(sub6)):
    z = Z[i]
    j_max = int(np.argmax(z))
    z_other = np.delete(z, j_max)
    spec = float(z[j_max] - np.median(z_other))
    spec_rows.append({
        "id": sub6.iloc[i]["id"],
        "geneid": sub6.iloc[i]["geneid"],
        "argmax_contrast": CONTRASTS[j_max].key,
        "specificity": spec,
    })
spec_df = pd.DataFrame(spec_rows)
spec_df.to_csv(OUT06 / "site_specificity_scores.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4))
spec_df["specificity"].hist(bins=50, ax=ax, color="purple", edgecolor="white")
ax.set_xlabel("Specificity score (row-z: max − median others)")
ax.set_title("Distribution of per-site contrast specificity")
plt.tight_layout()
fig.savefig(OUT06 / "hist_specificity.png", dpi=FIG_DPI_DEFAULT, bbox_inches="tight")
plt.show()

terms_long6 = parse_terms_long(sub6)
universe_ids6 = set(sub6["id"].astype(str))

cut = spec_df["specificity"].quantile(SPECIFICITY_TAIL_QUANTILE)
for c in CONTRASTS:
    sel = spec_df[(spec_df["argmax_contrast"] == c.key) & (spec_df["specificity"] >= cut)]
    qids = set(sel["id"].astype(str))
    if len(qids) < SPECIFICITY_TAIL_MIN_SITES:
        print(c.key, "tail too small:", len(qids))
        continue
    for layer in ["GOBP", "KEGG"]:
        res = fisher_terms_vs_rest(
            qids, universe_ids6, terms_long6, layer, min_support=FISHER_MIN_SUPPORT_SPECIFICITY, top_n=FISHER_TOP_TERMS_SPECIFICITY
        )
        if not res.empty:
            res.to_csv(OUT06 / f"fisher_specific_tail_{c.key}_{layer}.csv", index=False)
print("Saved", OUT06)
""".strip()
        )
    )

    # --- 07 provenance ---
    cells.append(
        md(
            """### 07 — Run manifest (provenance)

**Method:** Write **`run_manifest.json`**: timestamp, Python version, package versions, row counts, **`RUN_CONFIG`** thresholds, and (after section 04) clustering summary **`cinfo`** / **`CLUSTER_METHOD`**.

**Interpret:** One file for reproducibility and support questions about “what ran.”"""
        )
    )
    cells.append(
        code(
            r"""
import json
import sys
from datetime import datetime, timezone

OUT07 = OUT_ROOT / "00_provenance"
OUT07.mkdir(parents=True, exist_ok=True)
manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "data_path": str(DATA),
    "n_rows_df": int(len(df)),
    "n_finite_log2fc": int(finite_rows_mask(df).sum()),
    "python": sys.version,
    "run_config": dict(RUN_CONFIG),
    "packages": {
        "pandas": pd.__version__,
        "numpy": np.__version__,
    },
}
try:
    manifest["packages"]["sklearn"] = __import__("sklearn").__version__
except Exception:
    manifest["packages"]["sklearn"] = None
try:
    manifest["packages"]["scipy"] = __import__("scipy").__version__
except Exception:
    manifest["packages"]["scipy"] = None
try:
    import hdbscan as _hdb

    manifest["packages"]["hdbscan"] = getattr(_hdb, "__version__", "?")
except Exception:
    manifest["packages"]["hdbscan"] = None
try:
    manifest["cluster_method"] = CLUSTER_METHOD
    manifest["clustering_info"] = {str(k): (v if isinstance(v, (int, float, str, bool, type(None))) else repr(v)) for k, v in cinfo.items()}
except NameError:
    manifest["clustering_info"] = "undefined (run section 04 first)"
(OUT07 / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("Wrote", OUT07 / "run_manifest.json")
""".strip()
        )
    )

    # --- 08 protein-centric ---
    cells.append(
        md(
            """### 08 — Protein-centric summaries

**Method:** Aggregate **finite-Log2FC** sites by **`geneid`**: site counts, per-contrast significance counts (same rule as overlap: `q < q_max_sig`, `|LFC| ≥ abs_lfc_sig`), median Log2FC per contrast, min *q* per contrast, and best site id by `|sign(LFC)·(−log10 q)|`.

**Interpret:** Highlights multi-phospho proteins and genes repeatedly significant; heatmap shows contrast-wise median shift for top multi-site genes."""
        )
    )
    cells.append(
        code(
            r"""
import re

OUT08 = OUT_ROOT / "08_protein_centric"
OUT08.mkdir(parents=True, exist_ok=True)

mask = finite_rows_mask(df)
df_q = df.loc[mask].copy()
Q_MAX = RUN_CONFIG["q_max_sig"]
ABS_LFC = RUN_CONFIG["abs_lfc_sig"]

gdf = df_q.copy()
gdf["geneid"] = gdf["geneid"].astype(str).str.strip()
rows = []
for gene, grp in gdf.groupby("geneid"):
    if not gene or gene.upper() == "NAN":
        continue
    n_sites = len(grp)
    rec = {"geneid": gene, "n_sites": n_sites}
    med_lfc = {}
    n_sig = {}
    best_q = {}
    for c in CONTRASTS:
        sig = (grp[f"q_{c.key}"] < Q_MAX) & (grp[c.log2fc_col].abs() >= ABS_LFC)
        n_sig[c.key] = int(sig.sum())
        med_lfc[c.key] = float(grp[c.log2fc_col].median())
        best_q[c.key] = float(grp[f"q_{c.key}"].min())
    qmat = np.clip(grp[[f"q_{c.key}" for c in CONTRASTS]].astype(float), 1e-300, 1.0)
    lmat = grp[[c.log2fc_col for c in CONTRASTS]].astype(float)
    score_mat = np.sign(lmat.values) * (-np.log10(qmat.values))
    abs_s = np.nanmax(np.abs(score_mat), axis=1)
    j = int(np.argmax(abs_s))
    rec.update({f"n_sig_{c.key}": n_sig[c.key] for c in CONTRASTS})
    rec.update({f"median_lfc_{c.key}": med_lfc[c.key] for c in CONTRASTS})
    rec.update({f"min_q_{c.key}": best_q[c.key] for c in CONTRASTS})
    rec["best_site_id"] = str(grp.iloc[j]["id"])
    rec["max_abs_signed_neglog10q"] = float(abs_s[j])
    rows.append(rec)
protein_summary = pd.DataFrame(rows).sort_values("n_sites", ascending=False)
protein_summary.to_csv(OUT08 / "protein_summary.csv", index=False)

med_cols = [f"median_lfc_{c.key}" for c in CONTRASTS]
wide_med = protein_summary[["geneid", "n_sites"] + med_cols].copy()
wide_med.columns = ["geneid", "n_sites"] + [c.label for c in CONTRASTS]
wide_med.to_csv(OUT08 / "protein_contrast_median_lfc.csv", index=False)

top_n = min(25, len(protein_summary))
fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * top_n)))
subp = protein_summary.head(top_n).iloc[::-1]
ax.barh(subp["geneid"], subp["n_sites"], color="steelblue")
ax.set_xlabel("Number of phospho sites (finite cohort)")
ax.set_title("Top genes by site count")
plt.tight_layout()
fig.savefig(OUT08 / "bar_top_genes_by_site_count.png", dpi=FIG_DPI_DEFAULT, bbox_inches="tight")
plt.show()

multi = protein_summary[protein_summary["n_sites"] >= 2].head(30)
if len(multi) > 0:
    mat = multi.set_index("geneid")[[f"median_lfc_{c.key}" for c in CONTRASTS]]
    mat.columns = [c.label for c in CONTRASTS]
    fig_h = max(5, 0.35 * len(mat))
    fig, ax = plt.subplots(figsize=(11, fig_h))
    sns.heatmap(mat, cmap="vlag", center=0, ax=ax)
    ax.set_title("Median Log2FC per gene × contrast (top multi-site genes)")
    plt.tight_layout()
    fig.savefig(OUT08 / "heatmap_protein_median_lfc.png", dpi=FIG_DPI_DEFAULT, bbox_inches="tight")
    plt.show()
else:
    print("No genes with >=2 sites for heatmap.")

try:
    mrg = protein_summary.merge(spec_df.groupby("geneid")["specificity"].max().reset_index(), on="geneid", how="left")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(mrg["n_sites"], mrg["specificity"].fillna(0), s=18, alpha=0.35, edgecolors="none")
    ax.set_xlabel("n_sites per gene")
    ax.set_ylabel("max site specificity (within gene)")
    ax.set_title("Multi-site breadth vs contrast specificity")
    plt.tight_layout()
    fig.savefig(OUT08 / "scatter_nsites_vs_max_specificity.png", dpi=FIG_DPI_DEFAULT, bbox_inches="tight")
    plt.show()
except NameError:
    print("spec_df missing — run section 06 first for specificity scatter.")
print("Saved", OUT08)
""".strip()
        )
    )

    # --- 09 direction Fisher ---
    cells.append(
        md(
            """### 09 — Direction-stratified term enrichment (GOBP / KEGG)

**Method:** For each contrast, split **significant** sites (`RUN_CONFIG` overlap rule) into **up** (Log2FC > 0) and **down** (Log2FC < 0). **Fisher** vs all finite-row sites for **GOBP** and **KEGG** only (`min_support` relaxed for small sets).

**Interpret:** Up and down often implicate different biology; compare paired bars."""
        )
    )
    cells.append(
        code(
            r"""
OUT09 = OUT_ROOT / "09_direction_terms"
OUT09.mkdir(parents=True, exist_ok=True)

mask = finite_rows_mask(df)
df_q = df.loc[mask].copy()
terms_long9 = parse_terms_long(df_q)
universe_ids9 = set(df_q["id"].astype(str))
Q_MAX = RUN_CONFIG["q_max_sig"]
ABS_LFC = RUN_CONFIG["abs_lfc_sig"]
MIN_SUPPORT_DIR = FISHER_MIN_SUPPORT_DIRECTION
TOPN = FISHER_TOP_TERMS_DIRECTION

for c in CONTRASTS:
    base = (df_q[f"q_{c.key}"] < Q_MAX) & (df_q[c.log2fc_col].abs() >= ABS_LFC)
    up_ids = set(df_q.loc[base & (df_q[c.log2fc_col] > 0), "id"].astype(str))
    down_ids = set(df_q.loc[base & (df_q[c.log2fc_col] < 0), "id"].astype(str))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for j, layer in enumerate(["GOBP", "KEGG"]):
        for i, (tag, qset) in enumerate([("up", up_ids), ("down", down_ids)]):
            ax = axes[j, i]
            if len(qset) < MIN_SUPPORT_DIR:
                ax.text(0.5, 0.5, f"{c.key} {tag}: n={len(qset)} (too small)", ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
                continue
            res = fisher_terms_vs_rest(qset, universe_ids9, terms_long9, layer, min_support=MIN_SUPPORT_DIR, top_n=TOPN)
            fn = OUT09 / f"fisher_direction_{c.key}_{tag}_{layer}.csv"
            if res.empty:
                ax.text(0.5, 0.5, "no terms", ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
                res.to_csv(fn, index=False)
                continue
            take = res.head(min(FISHER_BARS_PER_CLUSTER_MAX, len(res)))
            ax.barh(take["term"].astype(str), -np.log10(take["p_fisher"].clip(1e-300)), color="darkslateblue" if tag == "up" else "firebrick")
            ax.invert_yaxis()
            ax.set_title(f"{c.label} · {layer} · {tag} (n={len(qset)})", fontsize=8)
            ax.set_xlabel("-log10 Fisher p")
            res.to_csv(fn, index=False)
    fig.suptitle(f"Direction Fisher: {c.key}", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT09 / f"direction_fisher_{c.key}.png", dpi=FIG_DPI_DIRECTION_FISHER, bbox_inches="tight")
    plt.show()
print("Saved under", OUT09)
""".strip()
        )
    )

    # --- 10 S/T/Y ---
    cells.append(
        md(
            """### 10 — Residue class (S / T / Y) from site `id`

**Method:** Regex on **`id`** (expects `…-S123` / `T` / `Y`). Count global and among **significant** sites per contrast (same overlap thresholds).

**Interpret:** Basophilic vs proline-directed programs often differ by S/T/Y usage; inspect **`unparsed_ids`** in the CSV if patterns fail."""
        )
    )
    cells.append(
        code(
            r"""
import re

OUT10 = OUT_ROOT / "10_residue_class"
OUT10.mkdir(parents=True, exist_ok=True)

_pat = re.compile(r"-(?P<res>[STY])(?P<pos>\d+)\s*$", re.IGNORECASE)

def parse_residue(site_id):
    m = _pat.search(str(site_id).strip())
    if not m:
        return None, None
    return m.group("res").upper(), int(m.group("pos"))

rows = []
for sid in df["id"].astype(str):
    res, pos = parse_residue(sid)
    rows.append({"id": sid, "residue": res, "position": pos})
res_df = pd.DataFrame(rows)
res_df.to_csv(OUT10 / "site_residue_class.csv", index=False)
unparsed = res_df[res_df["residue"].isna()]
unparsed.head(20).to_csv(OUT10 / "unparsed_ids_sample.csv", index=False)
print("Unparsed ids:", int(res_df["residue"].isna().sum()), "/", len(res_df))

vc = res_df["residue"].value_counts(dropna=True)
fig, ax = plt.subplots(figsize=(5, 3.5))
vc.plot(kind="bar", ax=ax, color=["#3182bd", "#9ecae1", "#fdae6b"])
ax.set_title("Global S / T / Y counts (parsed)")
ax.set_ylabel("sites")
plt.tight_layout()
fig.savefig(OUT10 / "bar_residue_global.png", dpi=FIG_DPI_DEFAULT, bbox_inches="tight")
plt.show()

mask = finite_rows_mask(df)
df_q = df.loc[mask].copy()
Q_MAX = RUN_CONFIG["q_max_sig"]
ABS_LFC = RUN_CONFIG["abs_lfc_sig"]
rmap = res_df.set_index("id")["residue"]

counts = {c.key: {"S": 0, "T": 0, "Y": 0} for c in CONTRASTS}
for c in CONTRASTS:
    sig = (df_q[f"q_{c.key}"] < Q_MAX) & (df_q[c.log2fc_col].abs() >= ABS_LFC)
    for sid in df_q.loc[sig, "id"].astype(str):
        rr = rmap.get(sid, None)
        if rr in counts[c.key]:
            counts[c.key][rr] += 1

fig, ax = plt.subplots(figsize=(10, 4))
labels = [c.label for c in CONTRASTS]
x = np.arange(len(CONTRASTS))
w = 0.25
ax.bar(x - w, [counts[c.key]["S"] for c in CONTRASTS], width=w, label="S", color="#3182bd")
ax.bar(x, [counts[c.key]["T"] for c in CONTRASTS], width=w, label="T", color="#6baed6")
ax.bar(x + w, [counts[c.key]["Y"] for c in CONTRASTS], width=w, label="Y", color="#fdae6b")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
ax.set_ylabel("significant sites")
ax.set_title("S/T/Y among significant sites per contrast")
ax.legend()
plt.tight_layout()
fig.savefig(OUT10 / "bar_residue_sig_per_contrast.png", dpi=FIG_DPI_DEFAULT, bbox_inches="tight")
plt.show()
print("Saved", OUT10)
""".strip()
        )
    )

    # --- 11 UniProt + motif ---
    cells.append(
        md(
            """### 11 — Sequence windows (UniProt) and motif-style plots

**Method:** Optional **`requests`** + **`tqdm`**: fetch FASTA per unique **`proteinid`** (UniProt accession) with a **progress bar**, cache under **`outputs_all/11_sequence_motif/fasta_cache/`**. Parse site position from **`id`**; extract ±7 aa windows (second progress bar). **AA frequency heatmap** across positions; **logomaker** logo if installed. **Binomial** test for **Pro at +1** vs pooled P rate at other window columns.

**Interpret:** Motifs support mechanism hypotheses; Pro+1 enrichment is a coarse CDK-like signal, not kinase ID. Skips on HTTP errors or missing package."""
        )
    )
    cells.append(
        code(
            r"""
OUT11 = OUT_ROOT / "11_sequence_motif"
OUT11.mkdir(parents=True, exist_ok=True)
CACHE = OUT11 / "fasta_cache"
CACHE.mkdir(parents=True, exist_ok=True)

windows = []
try:
    import requests
except ImportError:
    display(Markdown("**Skipped §11:** install `requests` for UniProt fetch."))
else:
    import re
    import time

    from tqdm.auto import tqdm

    _pat_site = re.compile(r"-(?P<res>[STY])(?P<pos>\d+)\s*$", re.IGNORECASE)

    def parse_site(site_id):
        m = _pat_site.search(str(site_id).strip())
        if not m:
            return None, None
        return m.group("res").upper(), int(m.group("pos"))

    def fetch_fasta(acc):
        p = CACHE / f"{acc}.fasta"
        if p.is_file():
            return p.read_text(encoding="utf-8", errors="replace")
        url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        txt = r.text
        p.write_text(txt, encoding="utf-8")
        time.sleep(0.34)
        return txt

    def fasta_to_seq(fa):
        lines = [ln.strip() for ln in fa.splitlines() if ln and not ln.startswith(">")]
        return "".join(lines).replace("\n", "")

    mask = finite_rows_mask(df)
    subw = df.loc[mask, ["id", "proteinid", "geneid"]].dropna(subset=["proteinid"]).copy()
    subw["proteinid"] = subw["proteinid"].astype(str).str.strip()
    accs = subw["proteinid"].unique().tolist()
    if len(accs) > UNIPROT_MAX_ACCESSIONS:
        print("Warning: truncating to", UNIPROT_MAX_ACCESSIONS, "accessions for UniProt fetch")
        accs = accs[:UNIPROT_MAX_ACCESSIONS]
    seq_by_acc = {}
    failed = []
    for acc in tqdm(accs, desc="UniProt FASTA", unit="acc"):
        try:
            fa = fetch_fasta(acc)
            seq_by_acc[acc] = fasta_to_seq(fa)
        except Exception as ex:
            failed.append((acc, str(ex)[:120]))
    pd.DataFrame(failed, columns=["accession", "error"]).to_csv(OUT11 / "uniprot_fetch_errors.csv", index=False)

    flank = SEQUENCE_FLANK_AA
    for _, row in tqdm(subw.iterrows(), total=len(subw), desc="Build ±7 windows", unit="site"):
        acc = row["proteinid"]
        res, pos = parse_site(row["id"])
        if res is None or acc not in seq_by_acc:
            continue
        s = seq_by_acc[acc]
        if pos < 1 or pos > len(s):
            continue
        i0 = pos - 1
        lo = max(0, i0 - flank)
        hi = min(len(s), i0 + flank + 1)
        wseq = s[lo:hi]
        center_local = i0 - lo
        if len(wseq) < flank * 2 + 1:
            continue
        windows.append({"id": row["id"], "accession": acc, "window": wseq, "center_local": center_local, "residue": res})

win_df = pd.DataFrame(windows)
win_df.to_csv(OUT11 / "phospho_windows_15mer.csv", index=False)
print("Windows built:", len(win_df))

if len(win_df) > 0:
    from collections import defaultdict

    AA = "ACDEFGHIKLMNPQRSTVWY"
    pos_counts = [defaultdict(int) for _ in range(2 * flank + 1)]
    plus1_p = 0
    plus1_total = 0
    for w in win_df["window"].astype(str):
        if len(w) != 2 * flank + 1:
            continue
        c = flank
        for j, ch in enumerate(w):
            if j == c:
                continue
            if ch in AA:
                pos_counts[j][ch] += 1
        if len(w) > c + 1:
            plus1_total += 1
            if w[c + 1] == "P":
                plus1_p += 1
    freq = np.zeros((len(AA), 2 * flank + 1))
    for j in range(2 * flank + 1):
        tot = sum(pos_counts[j].values())
        if tot == 0:
            continue
        for ai, a in enumerate(AA):
            freq[ai, j] = pos_counts[j][a] / tot
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(freq, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_yticks(range(len(AA)))
    ax.set_yticklabels(list(AA))
    ax.set_xticks(range(2 * flank + 1))
    ax.set_xticklabels([str(i - flank) for i in range(2 * flank + 1)])
    ax.set_xlabel("position relative to site (0 = phospho residue column skipped in matrix)")
    ax.set_title("AA frequency at each window column (excluding fixed center column in counts)")
    plt.colorbar(im, ax=ax, fraction=0.02)
    plt.tight_layout()
    fig.savefig(OUT11 / "heatmap_position_aa_freq.png", dpi=FIG_DPI_DEFAULT, bbox_inches="tight")
    plt.show()

    try:
        import logomaker

        counts_logo = pd.DataFrame([{a: float(pos_counts[j].get(a, 0)) for a in AA} for j in range(2 * flank + 1) if j != flank])
        prob_logo = counts_logo.div(counts_logo.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        fig_logo, ax_logo = plt.subplots(figsize=(10, 2.5))
        logomaker.Logo(prob_logo, ax=ax_logo, color_scheme="chemistry")
        ax_logo.set_title("Sequence logo (excluding fixed phospho column)")
        plt.tight_layout()
        fig_logo.savefig(OUT11 / "logo_window.png", dpi=FIG_DPI_HEATMAP, bbox_inches="tight")
        plt.show()
        plt.close(fig_logo)
    except Exception as ex:
        display(Markdown(f"**Logo skipped:** {ex}"))

    if plus1_total > 0:
        from scipy.stats import binomtest

        pooled = []
        for j in range(2 * flank + 1):
            if j in (flank, flank + 1):
                continue
            tot_j = sum(pos_counts[j].values())
            if tot_j == 0:
                continue
            pooled.append(pos_counts[j].get("P", 0) / tot_j)
        p0 = float(np.mean(pooled)) if pooled else (1.0 / 20.0)
        bt = binomtest(plus1_p, n=plus1_total, p=max(p0, 1e-6), alternative="greater")
        pd.DataFrame(
            [
                {
                    "test": "Pro_at_plus1_vs_pooled_other_positions",
                    "n_windows": plus1_total,
                    "n_pro_at_plus1": plus1_p,
                    "p_null_mean_other_cols": p0,
                    "binom_p_greater": float(bt.pvalue),
                }
            ]
        ).to_csv(OUT11 / "motif_proline_plus1_test.csv", index=False)
        print("Pro+1 binomial p=", float(bt.pvalue))

if len(win_df) == 0:
    display(Markdown("**No sequence windows** — see UniProt error log or install `requests`."))
print("Done", OUT11)
""".strip()
        )
    )
    # --- 12 preranked GSEA ---
    cells.append(
        md(
            """### 12 — Gene-level preranked GSEA (optional)

**Method:** Collapse sites to genes: per gene pick the site with largest **|sign(Log2FC) * (-log10 q)|**; score = **sign(Log2FC) * (-log10 q)** (same rule as `scripts/19_preranked_gsea_per_contrast.py`). When **`gseapy`** is installed, the notebook finds **every `*.gmt` file** in the same folder as **`data.csv`**, sorts them by name, and runs **`gseapy.prerank`** for **each GMT × each contrast**. Outputs: `outputs_all/12_gsea_prerank/<gmt_stem>/<contrast_key>/` (`ranked_genes.tsv`, `gsea_results.csv`, `bar_top_nes.png`).

**GMT files:** Use MSigDB or any GMT whose gene IDs match **`geneid`** (typically HGNC symbols). Examples: Hallmarks `h.all.*.gmt`, canonical pathways `c2.cp.*.gmt`, CGP `c2.cgp.*.gmt` — download from [MSigDB human collections](https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp). Drop multiple `.gmt` files next to `data.csv`; all are run.

**Interpret:** Enrichment on a **gene-level** ranking from phospho scores; it is not site-level kinase substrate mapping (see §13 for that style of table)."""
        )
    )
    cells.append(
        code(
            r"""
OUT12 = OUT_ROOT / "12_gsea_prerank"
OUT12.mkdir(parents=True, exist_ok=True)


def _discover_gmt_files():
    base = DATA.parent
    return sorted(base.glob("*.gmt"), key=lambda p: p.name.lower())


GMT_PATHS = _discover_gmt_files()


def gene_rnk(df0, c):
    qv = df0[f"q_{c.key}"].clip(lower=1e-300)
    lfc = df0[c.log2fc_col].astype(float)
    score = np.sign(lfc) * (-np.log10(qv))
    tmp = df0.assign(_score=score)
    tmp["_abs"] = tmp["_score"].abs()
    tmp["_gene"] = tmp["geneid"].astype(str).str.strip().str.upper()
    tmp = tmp[tmp["_gene"].notna() & (tmp["_gene"] != "NAN")]
    tmp = tmp[tmp["_abs"].notna()]
    idx = tmp.groupby("_gene", sort=False)["_abs"].idxmax().dropna()
    rnk = tmp.loc[idx, ["_gene", "_score"]].rename(columns={"_gene": "gene", "_score": "score"})
    return rnk.dropna(subset=["gene"]).sort_values("score", ascending=False)

try:
    import gseapy as gp
except ImportError:
    display(Markdown("**Skipped 12:** install `gseapy`."))
else:
    if not GMT_PATHS:
        display(Markdown("**Skipped 12:** no `*.gmt` files next to `data.csv`. Add one or more GMT files in that folder (see §12 markdown)."))
    else:
        print("GSEA §12 GMT files:", len(GMT_PATHS), "→", [p.name for p in GMT_PATHS])
        mask = finite_rows_mask(df)
        df_q = df.loc[mask].copy()
        for gmt_path in GMT_PATHS:
            gmt_key = gmt_path.stem
            print("  →", gmt_path.resolve())
            for c in CONTRASTS:
                rnk = gene_rnk(df_q, c)
                out_sub = OUT12 / gmt_key / c.key
                out_sub.mkdir(parents=True, exist_ok=True)
                rnk.to_csv(out_sub / "ranked_genes.tsv", sep="\t", index=False)
                rnk_series = pd.Series(rnk["score"].to_numpy(), index=rnk["gene"].astype(str).to_numpy())
                rnk_series = rnk_series.groupby(level=0).max().sort_values(ascending=False)
                pre_res = gp.prerank(
                    rnk=rnk_series,
                    gene_sets=str(gmt_path),
                    outdir=str(out_sub),
                    permutation_num=GSEA_PERMUTATIONS,
                    seed=1,
                    threads=GSEA_THREADS,
                    verbose=False,
                    no_plot=True,
                    min_size=GSEA_MIN_GENESET,
                    max_size=GSEA_MAX_GENESET,
                )
                res_df = getattr(pre_res, "res2d", None)
                if res_df is None:
                    res_df = getattr(pre_res, "results", None)
                if res_df is None or "NES" not in res_df.columns:
                    continue
                res_df.to_csv(out_sub / "gsea_results.csv", index=False)
                top = res_df.reindex(res_df["NES"].abs().sort_values(ascending=False).head(GSEA_TOP_PATHWAYS_BAR).index)
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.barh(top["Term"].astype(str), top["NES"].astype(float), color=np.where(top["NES"].astype(float) >= 0, "steelblue", "coral"))
                ax.axvline(0, color="k", lw=0.6)
                ax.set_title(f"Top |NES| — {gmt_key} — {c.label}")
                ax.invert_yaxis()
                plt.tight_layout()
                fig.savefig(out_sub / "bar_top_nes.png", dpi=FIG_DPI_DEFAULT, bbox_inches="tight")
                plt.show()
        print("Saved under", OUT12)
""".strip()
        )
    )

    # --- 13 kinase map ---
    cells.append(
        md(
            """### 13 — Kinase–substrate map (overview)

**What is `kinase_substrate_map.csv`?** A two-column table: **`kinase_gene`** (HGNC symbol of the kinase) and **`substrate_gene`** (HGNC symbol of the substrate protein). One row per kinase–substrate pair (duplicates OK; §13 aggregates). The notebook matches **`substrate_gene`** to your **`geneid`** values in `data.csv`.

**How to create it (pick one path):**

1. **PhosphoSitePlus (recommended curated source)**  
   - Go to **[PhosphoSitePlus → static downloads](https://www.phosphosite.org/staticDownloads)** (requires accepting their **license**).  
   - Download the bulk file usually named **`Kinase_Substrate_Dataset`** (sometimes offered as `.gz`).  
   - Put it in **`v3/`** with that basename (e.g. `v3/Kinase_Substrate_Dataset` or `v3/Kinase_Substrate_Dataset.gz`).  
   - Run **§13a** below: it reads PSP’s **`GENE`** (= kinase gene) and **`SUB_GENE`** (= substrate gene) columns and writes **`v3/kinase_substrate_map.csv`**.

2. **Manual / Excel**  
   - Export any table you trust to CSV with exactly two headers: `kinase_gene,substrate_gene` (or `kinase,substrate`). Save as **`v3/kinase_substrate_map.csv`**.

3. **Phospho.ELM or paper supplements**  
   - Reformat to the same two columns; mind each resource’s **terms of use**.

**Where to put the file:** **`v3/kinase_substrate_map.csv`** next to `data.csv`. If missing and §13a did not build it, §13 is skipped.

**Interpret:** Mean phospho change of substrates of kinase *K* vs a permutation null — exploratory. For **gene-set** enrichment without this file, §12 + a GMT (e.g. **C2 all** or Hallmarks) is the right tool.

**Next:** run **§13a** to auto-build the CSV from PhosphoSitePlus (if you placed their file in `v3/`), then **§13b** for the z-score plot."""
        )
    )
    cells.append(
        md(
            """### 13a — Build `kinase_substrate_map.csv` from PhosphoSitePlus (optional)

**When to use:** You downloaded **`Kinase_Substrate_Dataset`** from PhosphoSitePlus into **`v3/`** but do not have `kinase_substrate_map.csv` yet.

**What it does:** If a PSP file is found and `kinase_substrate_map.csv` is still missing, reads tab-separated rows, keeps human rows if an organism column exists, maps **`GENE` → kinase**, **`SUB_GENE` → substrate**, deduplicates, and writes **`v3/kinase_substrate_map.csv`**. If no PSP file is present, this cell prints a short hint and does nothing."""
        )
    )
    cells.append(
        code(
            r"""
MAP_OUT = DATA.parent / "kinase_substrate_map.csv"
candidates = []
for name in (
    "Kinase_Substrate_Dataset",
    "Kinase_Substrate_Dataset.txt",
    "Kinase_Substrate_Dataset.tsv",
    "Kinase_Substrate_Dataset.gz",
):
    p = DATA.parent / name
    if p.is_file():
        candidates.append(p)
if not candidates:
    for p in sorted(DATA.parent.glob("Kinase_Substrate_Dataset*")):
        if p.is_file():
            candidates.append(p)

if MAP_OUT.is_file():
    print("Already exists:", MAP_OUT)
elif not candidates:
    print("No PhosphoSitePlus Kinase_Substrate_Dataset file in v3/. Download from https://www.phosphosite.org/staticDownloads then re-run §13a.")
else:
    src = candidates[0]
    raw = pd.read_csv(src, sep="\t", low_memory=False, compression="infer")
    cols = {str(c).strip(): c for c in raw.columns}
    U = {k.upper().replace(" ", "_"): v for k, v in cols.items()}

    def pick(*names):
        for n in names:
            if n.upper() in U:
                return raw[U[n.upper()]]
        for n in names:
            for k, v in U.items():
                if n.upper() in k.replace(" ", "_"):
                    return raw[v]
        return None

    kin_col = pick("GENE", "KINASE_GENE", "KIN_GENE")
    sub_col = pick("SUB_GENE", "SUBSTRATE_GENE", "SUB_GENE_NAME")
    if kin_col is None or sub_col is None:
        print("Could not find GENE/SUB_GENE columns. Columns are:", list(raw.columns)[:25], "...")
    else:
        tmp = pd.DataFrame({"kinase_gene": kin_col.astype(str), "substrate_gene": sub_col.astype(str)})
        tmp["kinase_gene"] = tmp["kinase_gene"].str.strip().str.upper()
        tmp["substrate_gene"] = tmp["substrate_gene"].str.strip().str.upper()
        tmp = tmp[(tmp["kinase_gene"] != "") & (tmp["substrate_gene"] != "") & (tmp["kinase_gene"] != "NAN")]
        org = pick("SUB_ORGANISM", "KIN_ORGANISM", "ORGANISM")
        if org is not None:
            o = org.astype(str).str.lower()
            tmp = tmp.loc[o.str.contains("human", na=False) | o.str.contains("homo sapiens", na=False) | (o == "human")]
        tmp = tmp.drop_duplicates()
        tmp.to_csv(MAP_OUT, index=False)
        print("Wrote", MAP_OUT, "rows", len(tmp), "from", src.name)
""".strip()
        )
    )
    cells.append(
        md(
            """### 13b — Kinase proxy z-scores (needs `kinase_substrate_map.csv`)

**Method:** If **`kinase_substrate_map.csv`** exists, map each site’s **`geneid`** to kinases that phosphorylate that gene. For **each contrast** and each kinase with ≥ **`KINASE_MIN_SUBSTRATES`** mapped rows, **mean Log2FC** on those rows is compared to **`KINASE_N_PERM`** **label-shuffle** nulls → **`z_perm`** (see first code cell: `KINASE_*`, `KINASE_VERBOSE`).

**Outputs:** **`kinase_z_long.csv`** (one row per kinase × contrast; includes `null_mean`, `null_std`), **`kinase_z_wide.csv`**, **`kinase_mean_lfc_wide.csv`**. Figures: heatmap of Z across contrasts, **hierarchical clustermap** (exploratory reordering; not a hypothesis test), **2×4** horizontal bar grid (one panel per contrast), bubble matrix (color = Z, size = substrate count), and line profiles for top kinases by `max(|z|)` across contrasts.

**Interpret:** Exploratory. **Separate permutation null per contrast** — comparing `z_perm` across contrasts is descriptive only. Completeness of the PSP/map dominates which kinases appear."""
        )
    )
    cells.append(
        code(
            r"""
OUT13 = OUT_ROOT / "13_kinase_substrate_proxy"
OUT13.mkdir(parents=True, exist_ok=True)
MAP_PATH = DATA.parent / "kinase_substrate_map.csv"
if not MAP_PATH.is_file():
    display(Markdown("**Skipped 13:** add `kinase_substrate_map.csv` (columns `kinase_gene,substrate_gene`)."))
else:
    ks = pd.read_csv(MAP_PATH)
    low = {str(c).strip().lower().replace(" ", "_"): c for c in ks.columns}
    kg = low.get("kinase_gene", low.get("kinase", ks.columns[0]))
    sg = low.get("substrate_gene", low.get("substrate", ks.columns[min(1, len(ks.columns) - 1)]))
    sub_to_kin = {}
    for _, r in ks.iterrows():
        kin = str(r[kg]).strip().upper()
        sub = str(r[sg]).strip().upper()
        if kin and sub:
            sub_to_kin.setdefault(sub, set()).add(kin)
    mask = finite_rows_mask(df)
    df_q = df.loc[mask].copy()
    df_q["_geneu"] = df_q["geneid"].astype(str).str.strip().str.upper()
    genes = df_q["_geneu"].values
    kin_to_idx = {}
    for i, g in enumerate(genes):
        for kin in sub_to_kin.get(g, []):
            kin_to_idx.setdefault(kin, []).append(i)
    n_kin_ok = sum(1 for ix in kin_to_idx.values() if len(ix) >= KINASE_MIN_SUBSTRATES)
    if KINASE_VERBOSE:
        print(
            "[13b] map: substrates_with_kinase=",
            len(sub_to_kin),
            "| kinases with n>=" + str(KINASE_MIN_SUBSTRATES) + " mapped rows:",
            n_kin_ok,
            "| permutations:",
            KINASE_N_PERM,
        )
    rng = np.random.default_rng(RNG)
    rows_all = []
    for c in CONTRASTS:
        lfc = df_q[c.log2fc_col].astype(float).values
        for kin, idxs in kin_to_idx.items():
            if len(idxs) < KINASE_MIN_SUBSTRATES:
                continue
            obs = float(np.mean(lfc[idxs]))
            null = [
                float(np.mean(lfc[rng.choice(len(lfc), size=len(idxs), replace=False)])) for _ in range(KINASE_N_PERM)
            ]
            mu, sd = float(np.mean(null)), float(np.std(null) + 1e-9)
            z = (obs - mu) / sd
            rows_all.append(
                {
                    "contrast_key": c.key,
                    "contrast_label": c.label,
                    "kinase": kin,
                    "n_substrates": len(idxs),
                    "mean_lfc": obs,
                    "z_perm": z,
                    "null_mean": mu,
                    "null_std": sd,
                }
            )
        if KINASE_VERBOSE:
            tdf = pd.DataFrame([r for r in rows_all if r["contrast_key"] == c.key])
            if len(tdf) > 0:
                tdf = tdf.assign(_az=tdf["z_perm"].abs()).sort_values("_az", ascending=False).drop(columns=["_az"])
                print("[13b]", c.label, "| kinases tested:", len(tdf), "| top |z|:", tdf.head(3)[["kinase", "z_perm", "n_substrates"]].to_string(index=False))
    long_df = pd.DataFrame(rows_all)
    if len(long_df) == 0:
        display(Markdown("**Skipped 13b:** no kinases passed the minimum substrate count."))
    else:
        long_df.to_csv(OUT13 / "kinase_z_long.csv", index=False)
        z_wide = long_df.pivot_table(index="kinase", columns="contrast_key", values="z_perm", aggfunc="first")
        m_wide = long_df.pivot_table(index="kinase", columns="contrast_key", values="mean_lfc", aggfunc="first")
        col_order = [c.key for c in CONTRASTS if c.key in z_wide.columns]
        z_wide = z_wide.reindex(columns=col_order)
        m_wide = m_wide.reindex(columns=col_order)
        z_wide.to_csv(OUT13 / "kinase_z_wide.csv")
        m_wide.to_csv(OUT13 / "kinase_mean_lfc_wide.csv")
        label_map = {c.key: c.label for c in CONTRASTS}
        Z = z_wide.astype(float)
        max_abs = Z.abs().max(axis=1)

        n_show = min(KINASE_HEATMAP_MAX_KINASES, len(Z))
        top_heat = max_abs.nlargest(n_show).index
        Z_heat = Z.loc[top_heat]
        Z_heat_lbl = Z_heat.rename(columns=label_map)
        fig_h, ax_h = plt.subplots(figsize=(max(8, 0.65 * len(CONTRASTS)), max(5, 0.22 * len(Z_heat_lbl))))
        sns.heatmap(Z_heat_lbl, ax=ax_h, center=0, cmap="vlag", linewidths=0.3, cbar_kws={"label": "z_perm"})
        ax_h.set_title("Kinase z vs permutation null (subset by max |z| across contrasts)")
        plt.tight_layout()
        fig_h.savefig(OUT13 / "heatmap_kinase_z_across_contrasts.png", dpi=FIG_DPI_HEATMAP, bbox_inches="tight")
        plt.show()
        plt.close(fig_h)

        n_cl = min(KINASE_CLUSTERMAP_MAX_KINASES, len(Z))
        top_cl = max_abs.nlargest(n_cl).index
        Z_cl = Z.loc[top_cl].astype(float).fillna(0.0)
        Z_cl_lbl = Z_cl.rename(columns=label_map)
        if Z_cl_lbl.shape[0] >= 2 and Z_cl_lbl.shape[1] >= 2:
            cg = sns.clustermap(
                Z_cl_lbl,
                cmap="vlag",
                center=0,
                method=KINASE_CLUSTERMAP_METHOD,
                metric=KINASE_CLUSTERMAP_METRIC,
                figsize=(max(9, 0.7 * len(CONTRASTS)), max(7, 0.28 * len(Z_cl_lbl))),
            )
            cg.ax_heatmap.set_xlabel("")
            cg.savefig(OUT13 / "clustermap_kinase_z.png", dpi=FIG_DPI_HEATMAP, bbox_inches="tight")
            plt.close(cg.figure)

        else:
            print("[13b] clustermap skipped (need >=2 kinases and >=2 contrasts in matrix).")

        zlim = float(max(1e-6, long_df["z_perm"].abs().max()))
        fig_g, axes_g = plt.subplots(2, 4, figsize=(14, 10))
        for ax, c in zip(axes_g.ravel(), CONTRASTS):
            sub = long_df[long_df["contrast_key"] == c.key].copy()
            sub = sub.assign(_az=sub["z_perm"].abs()).sort_values("_az", ascending=False).drop(columns=["_az"])
            sub = sub.head(KINASE_BAR_TOP_N).iloc[::-1]
            if sub.empty:
                ax.axis("off")
                continue
            ax.barh(sub["kinase"].astype(str), sub["z_perm"].astype(float), color="darkmagenta")
            ax.axvline(0, color="k", lw=0.5)
            ax.set_xlim(-zlim * 1.05, zlim * 1.05)
            ax.set_title(c.label, fontsize=8)
            ax.set_xlabel("z_perm", fontsize=7)
        fig_g.suptitle("Kinase proxy z (top " + str(KINASE_BAR_TOP_N) + " per contrast by |z|)", fontsize=11)
        plt.tight_layout()
        fig_g.savefig(OUT13 / "bar_kinase_z_grid.png", dpi=FIG_DPI_DEFAULT, bbox_inches="tight")
        plt.show()
        plt.close(fig_g)

        n_dot = min(KINASE_DOT_MAX_KINASES, len(Z))
        dot_kin = max_abs.nlargest(n_dot).index.tolist()
        dot_kin_rev = list(reversed(dot_kin))
        ki_y = {k: i for i, k in enumerate(dot_kin_rev)}
        xs, ys, cs, ss = [], [], [], []
        n_by_k = long_df.drop_duplicates(subset=["kinase"]).set_index("kinase")["n_substrates"].astype(float)
        for j, c in enumerate(CONTRASTS):
            for kin in dot_kin:
                row = long_df[(long_df["contrast_key"] == c.key) & (long_df["kinase"] == kin)]
                if row.empty:
                    continue
                xs.append(float(j))
                ys.append(float(ki_y[kin]))
                cs.append(float(row["z_perm"].iloc[0]))
                ss.append(float(n_by_k.get(kin, row["n_substrates"].iloc[0])))
        fig_d, ax_d = plt.subplots(figsize=(max(9, 0.95 * len(CONTRASTS)), max(6, 0.24 * n_dot)))
        if xs:
            vmax = max(1e-6, max(abs(v) for v in cs))
            sz = 18.0 * np.sqrt(np.clip(ss, 1.0, None))
            sc = ax_d.scatter(xs, ys, c=cs, s=sz, cmap="vlag", vmin=-vmax, vmax=vmax, alpha=0.85, edgecolors="k", linewidths=0.2)
            plt.colorbar(sc, ax=ax_d, label="z_perm")
            ax_d.set_xticks(range(len(CONTRASTS)))
            ax_d.set_xticklabels([c.label for c in CONTRASTS], rotation=45, ha="right", fontsize=7)
            ax_d.set_yticks(range(len(dot_kin_rev)))
            ax_d.set_yticklabels(dot_kin_rev, fontsize=7)
            ax_d.set_xlabel("Contrast")
            ax_d.set_ylabel("Kinase (top " + str(n_dot) + " by max |z|)")
            ax_d.set_title("Bubble: color=z_perm, size ~ sqrt(n_substrates)")
            ax_d.grid(True, axis="x", alpha=0.25)
        plt.tight_layout()
        fig_d.savefig(OUT13 / "dot_kinase_z_matrix.png", dpi=FIG_DPI_DEFAULT, bbox_inches="tight")
        plt.show()
        plt.close(fig_d)

        line_kin = max_abs.nlargest(min(KINASE_LINEPLOT_TOP_N, len(Z))).index.tolist()
        fig_l, ax_l = plt.subplots(figsize=(max(8, 0.55 * len(CONTRASTS)), 5))
        xv = np.arange(len(CONTRASTS))
        for kin in line_kin:
            ys_l = [float(Z.loc[kin, c.key]) if c.key in Z.columns and kin in Z.index else np.nan for c in CONTRASTS]
            ax_l.plot(xv, ys_l, marker="o", ms=4, lw=1.0, label=str(kin))
        ax_l.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax_l.set_xticks(xv)
        ax_l.set_xticklabels([c.label for c in CONTRASTS], rotation=45, ha="right", fontsize=7)
        ax_l.set_ylabel("z_perm")
        ax_l.set_title("Top kinases by max |z| across contrasts")
        ax_l.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6, ncol=1)
        plt.tight_layout()
        fig_l.savefig(OUT13 / "line_kinase_z_profiles.png", dpi=FIG_DPI_DEFAULT, bbox_inches="tight")
        plt.show()
        plt.close(fig_l)

        if KINASE_VERBOSE:
            print("[13b] wrote:", OUT13 / "kinase_z_long.csv", "| figures: heatmap, clustermap (if ok), bar grid, dot matrix, lines")
    print("Saved", OUT13)
""".strip()
        )
    )

    # --- 14 network STRING upset ---
    cells.append(
        md(
            """### 14 — Co-regulation graph, STRING, UpSet (optional)

**Method:** (1) Genes with **`GENE_CORR_MIN_SITES`**+ sites: median Log2FC per gene; gene–gene Pearson on median vectors; edges **|r| ≥ `GENE_CORR_EDGE_THRESHOLD`**. **Network (14a):** the full edge list is still saved to CSV; the **spring** figure uses at most **`GENE_CORR_MAX_NODES_DRAW`** high-degree genes (reduces hairballs), longer **`GENE_CORR_SPRING_ITERATIONS`**, degree-sized nodes, and labels only for the **`GENE_CORR_LABEL_TOP_NODES`** highest-degree nodes. **Alternative A — embedding:** 2D layout from the weighted adjacency (try **`umap-learn`** on connectivity rows; if missing, **spectral embedding** / Laplacian eigenmaps from sklearn). **Alternative B — hub subgraph:** top **`GENE_CORR_HUB_SUBGRAPH_N`** genes by degree, induced subgraph, spring layout. (2) **STRING** API (**`STRING_TOP_N_GENES`**, **`STRING_REQUIRED_SCORE`**, **`requests`**). (3) **UpSet** of eight significant site sets (same rule as §03 — `Q_MAX_SIG` / `ABS_LFC_SIG`).

**Interpret:** Co-moving targets; STRING adds PPI context; UpSet shows overlap sizes."""
        )
    )
    cells.append(
        code(
            r"""
OUT14 = OUT_ROOT / "14_network_context"
OUT14.mkdir(parents=True, exist_ok=True)
mask = finite_rows_mask(df)
df_q = df.loc[mask].copy()
Q_MAX = RUN_CONFIG["q_max_sig"]
ABS_LFC = RUN_CONFIG["abs_lfc_sig"]

try:
    import networkx as nx
except ImportError:
    display(Markdown("**Skipped 14a:** install `networkx`."))
else:
    gdf = df_q.copy()
    gdf["geneid"] = gdf["geneid"].astype(str).str.strip()
    med = gdf.groupby("geneid")[[c.log2fc_col for c in CONTRASTS]].median()
    med = med[(med.index.str.len() > 0) & (med.index.str.upper() != "NAN")]
    cnt = gdf.groupby("geneid").size()
    keep = cnt[cnt >= GENE_CORR_MIN_SITES].index
    M = med.loc[med.index.isin(keep)]
    if len(M) >= 3:
        Cg = M.T.corr(method="pearson")
        thr = GENE_CORR_EDGE_THRESHOLD
        edges = []
        idx = list(Cg.index)
        for ii in range(len(idx)):
            for jj in range(ii + 1, len(idx)):
                gi, gj = idx[ii], idx[jj]
                v = Cg.iloc[ii, jj]
                if pd.notna(v) and abs(float(v)) >= thr:
                    edges.append((gi, gj, float(v)))
        pd.DataFrame(edges, columns=["gene_a", "gene_b", "pearson_r"]).to_csv(OUT14 / "gene_coregulation_edges.csv", index=False)
        G = nx.Graph()
        G.add_weighted_edges_from([(a, b, abs(w)) for a, b, w in edges])
        if G.number_of_nodes() == 0:
            print("No edges above correlation threshold.")
        else:
            deg_full = dict(G.degree())
            nodes_by_deg = sorted(G.nodes(), key=lambda n: deg_full.get(n, 0), reverse=True)

            def _top_subgraph(G0, ranked_nodes, cap):
                if len(ranked_nodes) <= cap:
                    return G0
                return G0.subgraph(set(ranked_nodes[:cap])).copy()

            # --- Primary: capped nodes, tuned spring, degree-sized nodes, labels for top degree only ---
            G_draw = _top_subgraph(G, nodes_by_deg, GENE_CORR_MAX_NODES_DRAW)
            n_draw = G_draw.number_of_nodes()
            k_spring = STRING_SPRING_K * max(0.12, 1.05 / max(1, float(np.sqrt(n_draw))))
            pos = nx.spring_layout(
                G_draw,
                seed=STRING_NETWORK_LAYOUT_SEED,
                k=k_spring,
                iterations=GENE_CORR_SPRING_ITERATIONS,
                weight="weight",
            )
            deg_d = dict(G_draw.degree())
            node_list = list(G_draw.nodes())
            sizes = [22.0 + 48.0 * float(deg_d.get(n, 0)) for n in node_list]
            label_set = set(nodes_by_deg[: min(GENE_CORR_LABEL_TOP_NODES, len(nodes_by_deg))])
            el = list(G_draw.edges(data=True))
            wlist = [0.35 + 1.1 * float(d.get("weight", 0.0)) for _, _, d in el]
            ebunch = [(u, v) for u, v, _ in el]
            fig, ax = plt.subplots(figsize=(10, 8))
            nx.draw_networkx_nodes(G_draw, pos, nodelist=node_list, node_size=sizes, ax=ax, alpha=0.86, node_color="#4c72b0")
            if ebunch:
                nx.draw_networkx_edges(G_draw, pos, edgelist=ebunch, width=wlist, ax=ax, alpha=0.22, edge_color="0.55")
            for n in node_list:
                if n not in label_set:
                    continue
                x, y = pos[n]
                ax.text(
                    float(x),
                    float(y),
                    str(n),
                    fontsize=5.5,
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="0.7", lw=0.2, alpha=0.82),
                )
            ax.axis("off")
            ax.set_title(
                f"Gene co-regulation (spring) | |r|≥{GENE_CORR_EDGE_THRESHOLD}, ≥{GENE_CORR_MIN_SITES} sites | "
                f"draw n={n_draw} of |V|={G.number_of_nodes()} (top-degree cap)"
            )
            plt.tight_layout()
            fig.savefig(OUT14 / "graph_gene_coregulation.png", dpi=FIG_DPI_HEATMAP, bbox_inches="tight")
            plt.show()
            plt.close(fig)

            # --- Alt B: hub-only induced subgraph (more readable labels) ---
            Gh = _top_subgraph(G, nodes_by_deg, GENE_CORR_HUB_SUBGRAPH_N)
            if Gh.number_of_nodes() >= 2:
                nh = Gh.number_of_nodes()
                pos_h = nx.spring_layout(
                    Gh,
                    seed=STRING_NETWORK_LAYOUT_SEED + 7,
                    k=STRING_SPRING_K * max(0.18, 1.15 / max(1, float(np.sqrt(nh)))),
                    iterations=GENE_CORR_SPRING_ITERATIONS,
                    weight="weight",
                )
                deg_h = dict(Gh.degree())
                fig_h, ax_h = plt.subplots(figsize=(10, 8))
                nx.draw_networkx_nodes(
                    Gh,
                    pos_h,
                    node_size=[30 + 48 * deg_h[n] for n in Gh.nodes()],
                    ax=ax_h,
                    alpha=0.88,
                    node_color="#2ca02c",
                )
                nx.draw_networkx_edges(Gh, pos_h, ax=ax_h, alpha=0.35, width=1.0, edge_color="0.45")
                nx.draw_networkx_labels(Gh, pos_h, font_size=6.5, ax=ax_h)
                ax_h.axis("off")
                ax_h.set_title(f"Hub subgraph (top {GENE_CORR_HUB_SUBGRAPH_N} genes by degree, |r|≥{GENE_CORR_EDGE_THRESHOLD})")
                plt.tight_layout()
                fig_h.savefig(OUT14 / "graph_gene_coregulation_hub_subgraph.png", dpi=FIG_DPI_HEATMAP, bbox_inches="tight")
                plt.show()
                plt.close(fig_h)

            # --- Alt A: 2D embedding from weighted adjacency (UMAP if installed, else spectral) ---
            nodes_emb = nodes_by_deg[: min(GENE_CORR_EMBED_MAX_NODES, len(nodes_by_deg))]
            G_emb = G.subgraph(nodes_emb).copy()
            ccs = list(nx.connected_components(G_emb))
            if ccs:
                giant = max(ccs, key=len)
                if len(giant) >= 4:
                    G_emb = G_emb.subgraph(giant).copy()
            if G_emb.number_of_nodes() >= 4:
                order = sorted(G_emb.nodes())
                ix = {n: i for i, n in enumerate(order)}
                n_e = len(order)
                A = np.zeros((n_e, n_e), dtype=float)
                for u, v, d in G_emb.edges(data=True):
                    iu, iv = ix[u], ix[v]
                    w0 = float(d.get("weight", 0.0))
                    A[iu, iv] = w0
                    A[iv, iu] = w0
                np.fill_diagonal(A, A.diagonal() + 1e-5)
                X2 = None
                emb_title = ""
                try:
                    import umap

                    nn_u = max(2, min(20, n_e - 1))
                    red = umap.UMAP(
                        n_components=2,
                        random_state=int(RNG) & 0x7FFFFFFF,
                        n_neighbors=nn_u,
                        min_dist=0.12,
                        metric="cosine",
                    )
                    X2 = red.fit_transform(A)
                    emb_title = "UMAP (cosine) on adjacency rows"
                except Exception:
                    X2 = None
                if X2 is None:
                    from sklearn.manifold import SpectralEmbedding

                    se = SpectralEmbedding(
                        n_components=2,
                        affinity="precomputed",
                        random_state=int(RNG) & 0x7FFFFFFF,
                    )
                    X2 = se.fit_transform(np.clip(A, 1e-8, None))
                    emb_title = "Spectral embedding on adjacency (install umap-learn for UMAP)"
                fig_e, ax_e = plt.subplots(figsize=(9, 7))
                de = dict(G_emb.degree())
                sz_e = np.array([20.0 + 42.0 * float(de[n]) for n in order], dtype=float)
                ax_e.scatter(X2[:, 0], X2[:, 1], s=sz_e, alpha=0.82, c="#9467bd", edgecolors="k", linewidths=0.2)
                lab_nodes = set(nodes_by_deg[: min(20, len(order))])
                for i, n in enumerate(order):
                    if n in lab_nodes:
                        ax_e.text(X2[i, 0], X2[i, 1], str(n), fontsize=6, ha="center", va="bottom", alpha=0.92)
                ax_e.set_title(emb_title + f" | giant component n={n_e}")
                ax_e.set_xlabel("Embedding 1")
                ax_e.set_ylabel("Embedding 2")
                plt.tight_layout()
                fig_e.savefig(OUT14 / "graph_gene_coregulation_embedding.png", dpi=FIG_DPI_HEATMAP, bbox_inches="tight")
                plt.show()
                plt.close(fig_e)
    else:
        print("Too few multi-site genes for graph.")

try:
    import requests
except ImportError:
    display(Markdown("**Skipped 14b:** install `requests` for STRING."))
else:
    c0 = CONTRASTS[0]
    gdf = df_q.copy()
    gdf["geneid"] = gdf["geneid"].astype(str).str.strip()
    med0 = gdf.groupby("geneid")[c0.log2fc_col].median().abs().sort_values(ascending=False)
    top_genes = med0.head(STRING_TOP_N_GENES).index.astype(str).tolist()
    if top_genes:
        url = "https://string-db.org/api/tsv/network"
        params = {
            "identifiers": "\n".join(top_genes),
            "species": 9606,
            "required_score": STRING_REQUIRED_SCORE,
            "network_type": "functional",
        }
        try:
            r = requests.get(url, params=params, timeout=STRING_TIMEOUT_SEC)
            r.raise_for_status()
            (OUT14 / "string_network.tsv").write_text(r.text, encoding="utf-8")
            lines = [ln.split("\t") for ln in r.text.strip().splitlines()]
            if len(lines) > 1:
                hdr, body = lines[0], lines[1:]
                pd.DataFrame(body, columns=hdr).to_csv(OUT14 / "string_network_parsed.csv", index=False)
                print("STRING rows:", len(body))
        except Exception as ex:
            print("STRING fetch failed:", ex)

sets_upset = {}
for c in CONTRASTS:
    sig = (df[f"q_{c.key}"] < Q_MAX) & (df[c.log2fc_col].abs() >= ABS_LFC)
    sets_upset[c.label] = set(df.loc[sig, "id"].astype(str))
try:
    from upsetplot import from_contents, UpSet

    up = from_contents(sets_upset)
    fig = plt.figure(figsize=(10, 5))
    UpSet(up, subset_size="count", show_counts=True).plot(fig=fig)
    plt.suptitle("UpSet: significant sites per contrast")
    plt.tight_layout()
    fig.savefig(OUT14 / "upset_significant_sites.png", dpi=FIG_DPI_DEFAULT, bbox_inches="tight")
    plt.show()
except ImportError:
    display(Markdown("**Skipped 14c:** install `upsetplot`."))
print("Saved", OUT14)
""".strip()
        )
    )

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT_NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("Wrote", OUT_NB)


if __name__ == "__main__":
    main()
