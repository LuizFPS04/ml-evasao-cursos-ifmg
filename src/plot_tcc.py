"""
Gráficos individuais do TCC, no estilo de Teodoro e Kappel (2020): barras em vermelho/azul,
fundo branco, grade leve, uma figura por arquivo. Largura = \\textwidth (6,3 in) nas figuras
inteiras e 0.49\\textwidth (3,1 in) nos painéis que ficam lado a lado (cursos, campus, importância,
correlação e grupos temáticos).

Escopo da base: só o IFMG (uma IES, federal, MG, cursos presenciais). Por isso não há gráficos por
região, UF, IES, tipo administrativo ou modalidade (seriam constantes); o recorte espacial é o
CAMPUS (município de oferta do curso).

Entradas (geradas por src.models.ml_models_train, em data/results/csv/):
  predictions_oof_cv{sufixo}.csv, metrics_cv_all_models{sufixo}.csv,
  importancia_consenso_modelos_*.csv, eda_correlacao_com_alvo.csv (opcional),
  avaliacao_recortes_subgrupos.csv (opcional)

Os gráficos de campus usam sempre o modelo MODELO_CAMPUS (Random Forest), independentemente
do modelo de referência usado nos demais gráficos.

Uso:  python -m src.plot_tcc --design D0
      python -m src.plot_tcc --design D1
Saída: data/results/figures_tcc/<D0|D1>/<grupo>/<figura>.pdf e .png
"""
import argparse
import inspect
import os
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

from src.utils import GRAU, MISSING, campus_of

# ------------------------------------------------------------------------------ configuração
CSV = "data/results/csv"
OUT = "data/results/figures_tcc"
SCENARIO = "Completo - Curso + histórico + IES/contexto"
KEYS = ["id_ies", "id_curso", "ano"]
FW = 6.3                                   # largura da figura em polegadas (= \textwidth SBC)
FW_MEIA = 3.1                              # painel de subfigura (= 0.49\textwidth): o LaTeX não reduz a fonte
MODELO_CAMPUS = "Random Forest"            # modelo fixo dos gráficos de campus

RED, BLUE, GRAY = "#E8312F", "#1F7FB0", "#BDBDBD"
CLASSES = ["baixo", "médio", "alto"]
PROB = {"baixo": "prob_baixo", "médio": "prob_medio", "alto": "prob_alto"}
MODEL_ORDER = ["Naive Bayes", "kNN", "Árvore de Decisão", "Random Forest", "MLP"]
MODEL_COLORS = {"Naive Bayes": "#1f77b4", "kNN": "#ff7f0e", "Árvore de Decisão": "#2ca02c",
                "Random Forest": "#d62728", "MLP": "#9467bd"}
AREAS_CINE = {0: "Programas básicos", 1: "Educação", 2: "Artes e humanidades",
              3: "Ciências sociais e info.", 4: "Negócios e direito", 5: "Ciências naturais e mat.",
              6: "Computação e TIC", 7: "Engenharia e construção", 8: "Agricultura e veterinária",
              9: "Saúde e bem-estar", 10: "Serviços", -1: "Não informado"}
# Palavras que ficam em minúscula no meio do nome do curso; siglas que ficam em maiúscula
MINUSCULAS = {"a", "à", "ao", "aos", "as", "às", "com", "da", "das", "de", "do", "dos", "e", "em",
              "na", "nas", "no", "nos", "o", "os", "ou", "para", "por", "sem"}
SIGLAS = {"TI", "TIC", "RH", "EAD", "SUS", "LIBRAS"}


def set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888888",
        "axes.axisbelow": True, "grid.color": "#E4E4E4", "grid.linewidth": 0.8,
        "legend.frameon": True, "legend.framealpha": 1, "legend.edgecolor": "#CCCCCC",
        "legend.fontsize": 7.5, "savefig.dpi": 300, "savefig.bbox": "tight"})


def save(fig, group, name):
    folder = os.path.join(OUT, group)
    os.makedirs(folder, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(folder, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  salvo: {group}/{name}")


def short(text, n=42):
    text = str(text)
    return text if len(text) <= n else text[: n - 1] + "…"


def nome_curso(texto, largura=24, max_linhas=2):
    """
    Nome do curso legível: preposições em minúscula ("Gestão da Produção Industrial"),
    siglas em maiúscula e quebra em até `max_linhas` linhas de `largura` caracteres
    (só corta com "…" se ainda assim não couber).
    """
    palavras = str(texto).strip().split()
    out = []
    for i, p in enumerate(palavras):
        pl = p.lower()
        if p.upper().strip("()") in SIGLAS:
            out.append(p.upper())
        elif i > 0 and pl in MINUSCULAS:
            out.append(pl)
        else:
            out.append("-".join(s[:1].upper() + s[1:] for s in pl.split("-")))
    linhas = textwrap.wrap(" ".join(out), largura)
    if len(linhas) > max_linhas:
        linhas = linhas[:max_linhas]
        linhas[-1] = short(linhas[-1] + " …", largura) if len(linhas[-1]) < largura - 1 \
            else linhas[-1][: largura - 1].rstrip() + "…"
    return "\n".join(linhas)


def _barh_duplo(ax, labels, v1, v2, l1, l2):
    y = np.arange(len(labels))
    h = 0.38
    b1 = ax.barh(y - h / 2, v1, h, color=RED, label=l1)
    b2 = ax.barh(y + h / 2, v2, h, color=BLUE, label=l2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.bar_label(b1, fmt="%.1f", padding=2, fontsize=7)
    ax.bar_label(b2, fmt="%.1f", padding=2, fontsize=7)
    ax.grid(axis="x")
    ax.set_xlim(0, max(max(v1), max(v2)) * 1.18)


# ------------------------------------------------------------------------------ dados
def load_predictions(suffix=""):
    p = pd.read_csv(f"{CSV}/predictions_oof_cv{suffix}.csv")
    p = p.rename(columns={"model": "modelo", "taxa_evasao_prox_real": "taxa_evasao_real"})
    return p


def build_meta(d, X):
    """Metadados por (id_ies, id_curso, ano) para cruzar com as previsões fora da dobra."""
    meta = pd.DataFrame({
        "id_ies": d["id_ies"], "id_curso": d["id_curso"], "ano": d["ano"],
        "campus": campus_of(d),
        "grau": d["tipo_grau_academico"].map(GRAU).fillna("Não informado"),
        "area": d["id_area_geral"].fillna(-1).astype(int)})
    for c in ["p_idade_0_24", "p_idade_35_49", "p_idade_50_mais"]:
        if c in X.columns:
            meta[c] = X[c].replace(MISSING, np.nan)
    return meta


def best_model(metrics, scenario):
    m = metrics[metrics["scenario"] == scenario]
    return m.groupby("modelo")["F1-macro"].mean().idxmax()


def preds_modelo(pred, meta, scenario, modelo):
    """Previsões fora da dobra de um modelo/cenário, cruzadas com os metadados."""
    return pred[(pred["scenario"] == scenario) & (pred["modelo"] == modelo)].merge(
        meta, on=KEYS, how="left", validate="many_to_one")


# ------------------------------------------------------------------------------ 1. desempenho
def plot_roc(pred, classe, scenario, name):
    """Curva ROC média (one-vs-rest) por modelo: média das 5 dobras, AUC ± desvio padrão."""
    p = pred[pred["scenario"] == scenario]
    grid = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(FW * 0.78, FW * 0.78))
    ax.plot([0, 1], [0, 1], "--", color=RED, lw=1.1, label="Previsão aleatória")
    ordem = [m for m in MODEL_ORDER if m in p["modelo"].unique()]
    for modelo in ordem:
        tprs, aucs = [], []
        for _, gf in p[p["modelo"] == modelo].groupby("fold"):
            y = (gf["risco_real"] == classe).astype(int)
            if y.nunique() < 2:
                continue
            fpr, tpr, _ = roc_curve(y, gf[PROB[classe]])
            tprs.append(np.interp(grid, fpr, tpr))
            aucs.append(auc(fpr, tpr))
        mt = np.mean(tprs, axis=0)
        mt[0] = 0.0
        ax.plot(grid, mt, color=MODEL_COLORS.get(modelo), lw=1.5,
                label=f"{modelo} (AUC = {np.mean(aucs):.2f} ± {np.std(aucs):.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos")
    ax.grid(True)
    ax.legend(loc="lower right")
    save(fig, "01_desempenho", name)


def plot_metricas(metrics, scenario):
    m = metrics[metrics["scenario"] == scenario]
    cols = ["acurácia", "F1-macro", "AUC OvR", "kappa quad."]
    media = m.groupby("modelo")[cols].mean()
    dp = m.groupby("modelo")[cols].std()
    ordem = [x for x in MODEL_ORDER if x in media.index]
    fig, ax = plt.subplots(figsize=(FW, FW * 0.55))
    x = np.arange(len(cols))
    w = 0.8 / len(ordem)
    for i, modelo in enumerate(ordem):
        ax.bar(x + (i - (len(ordem) - 1) / 2) * w, media.loc[modelo, cols], w,
               yerr=dp.loc[modelo, cols], capsize=2, color=MODEL_COLORS.get(modelo), label=modelo,
               error_kw={"lw": 0.8})
    ax.axhline(1 / 3, color="gray", ls=":", lw=1)
    ax.text(len(cols) - 0.55, 1 / 3 + 0.01, "F1 do acaso (0,33)", fontsize=7, color="gray", ha="right")
    ax.set_xticks(x, ["Acurácia", "F1-macro", "AUC OvR", "Kappa quadrático"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Valor médio (5 dobras)")
    ax.grid(axis="y")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    save(fig, "01_desempenho", "metricas_por_modelo")


# ------------------------------------------------------------------------------ 2. cursos
def plot_top_cursos(pm, nivel="nome", k=5, min_obs=30):
    """
    Top-k cursos com maior e menor risco identificado pelo modelo.
    Maior risco: ordena pelo % de observações PREVISTAS como risco alto; menor risco: pelo % previstas
    como risco baixo. Em vermelho, o % real; em azul, o % previsto (mesma unidade, comparáveis).
    """
    chave = ["nome_curso"] if nivel == "nome" else ["campus", "id_curso", "nome_curso"]
    pm = pm.assign(real_alto=pm["risco_real"] == "alto", prev_alto=pm["risco_previsto"] == "alto",
                   real_baixo=pm["risco_real"] == "baixo", prev_baixo=pm["risco_previsto"] == "baixo")
    g = pm.groupby(chave, dropna=False).agg(
        n=("taxa_evasao_real", "size"), evasao=("taxa_evasao_real", "mean"),
        real_alto=("real_alto", "mean"), prev_alto=("prev_alto", "mean"),
        real_baixo=("real_baixo", "mean"), prev_baixo=("prev_baixo", "mean")).reset_index()
    g = g[g["n"] >= min_obs]
    if g.empty:
        print(f"  (top-{k} por {nivel}: nenhum curso com n >= {min_obs})")
        return
    for tag, col, real, prev, rot in [
        ("maior_risco", "prev_alto", "real_alto", "prev_alto", "risco alto"),
        ("menor_risco", "prev_baixo", "real_baixo", "prev_baixo", "risco baixo")]:
        top = g.nlargest(k, [col, real])
        labels = []
        for r in top.itertuples():
            info = f"n={r.n}; evasão {r.evasao * 100:.1f}%"
            if nivel != "nome":
                info = f"{r.campus}; " + info
            labels.append(f"{nome_curso(r.nome_curso)}\n({info})")
        n_linhas = max(lbl.count("\n") + 1 for lbl in labels)
        fig, ax = plt.subplots(figsize=(FW_MEIA, (0.2 + 0.14 * n_linhas) * len(top) + 1.3))
        _barh_duplo(ax, labels, top[real] * 100, top[prev] * 100,
                    f"% real em {rot}", f"% previsto em {rot}")
        ax.tick_params(axis="y", labelsize=7, length=0)
        ax.set_xlabel(f"Curso × ano em {rot} (%)")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, borderaxespad=0.3)
        sufixo = "" if nivel == "nome" else "_curso_campus"
        save(fig, "02_cursos", f"top{k}_{tag}{sufixo}")


# ------------------------------------------------------------------------------ 3. campus
def _ic95(s):
    return 1.96 * s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0


def _ordem_campus(pm, min_obs):
    """Campi com pelo menos `min_obs` cursos × ano, do maior para o menor índice de evasão média."""
    g = pm.groupby("campus")["taxa_evasao_real"].agg(["mean", "size"])
    return g[g["size"] >= min_obs].sort_values("mean", ascending=False)


def plot_campus(pm, rot, min_obs=5):
    """
    Gráficos por campus (município de oferta). `pm` deve conter as previsões do MODELO_CAMPUS.
    Campi com menos de `min_obs` cursos × ano ficam de fora (média e F1 instáveis).
    """
    pm = pm[pm["campus"] != "ND"]
    g = _ordem_campus(pm, min_obs)
    if g.empty:
        print(f"  (campus: nenhum com n >= {min_obs})")
        return
    ordem = g.index.tolist()
    pm = pm[pm["campus"].isin(ordem)]
    labels = [f"{c} (n={n})" for c, n in zip(ordem, g["size"])]
    media_geral = pm["taxa_evasao_real"].mean() * 100
    altura = 0.24 * len(ordem) + 1.0

    # Taxa de evasão média por campus (IC 95%), vermelho = acima da média do IFMG
    ic = pm.groupby("campus")["taxa_evasao_real"].apply(_ic95).reindex(ordem) * 100
    media = g["mean"] * 100
    fig, ax = plt.subplots(figsize=(FW * 0.8, altura))
    cores = [RED if v >= media_geral else BLUE for v in media]
    b = ax.barh(labels, media, xerr=ic, color=cores, error_kw={"lw": 0.8, "capsize": 2})
    ax.invert_yaxis()
    ax.bar_label(b, fmt="%.1f", padding=3, fontsize=7)
    ax.axvline(media_geral, color="#555", ls="--", lw=0.9)
    ax.set_xlabel(f"Taxa de evasão média {rot} (%)\n(barra: IC 95%; tracejado: média do IFMG; "
                  "vermelho = acima)")
    ax.grid(axis="x")
    save(fig, "03_campus", "campus_taxa_evasao")

    # Distribuição das classes de risco reais por campus
    dist = pd.crosstab(pm["campus"], pm["risco_real"], normalize="index")
    dist = dist.reindex(index=ordem, columns=CLASSES, fill_value=0) * 100
    fig, ax = plt.subplots(figsize=(FW * 0.8, altura + 0.3))
    base = np.zeros(len(ordem))
    for c, cor in zip(CLASSES, [BLUE, GRAY, RED]):
        ax.barh(labels, dist[c], left=base, color=cor, label=f"Risco {c}")
        for i, v in enumerate(dist[c]):
            if v >= 8:
                ax.text(base[i] + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=6.5,
                        color="white" if cor != GRAY else "#333")
        base += dist[c].values
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Cursos × ano (%)")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), borderaxespad=0.3)
    ax.grid(axis="x")
    save(fig, "03_campus", "campus_distribuicao_risco")

    # Painel de meia largura: % em risco alto, real vs. previsto (Random Forest)
    real = pm.assign(x=pm["risco_real"] == "alto").groupby("campus")["x"].mean().reindex(ordem) * 100
    prev = pm.assign(x=pm["risco_previsto"] == "alto").groupby("campus")["x"].mean().reindex(ordem) * 100
    fig, ax = plt.subplots(figsize=(FW_MEIA, 0.3 * len(ordem) + 1.1))
    _barh_duplo(ax, labels, real.values, prev.values, "Real", "Previsto")
    ax.tick_params(axis="y", labelsize=6.8, length=0)
    ax.set_xlabel("Cursos × ano em risco alto (%)")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), borderaxespad=0.3)
    save(fig, "03_campus", "campus_risco_alto_real_vs_previsto")

    plot_heatmap_campus_area(pm, ordem)


def plot_heatmap_campus_area(pm, ordem, min_obs=3):
    """Taxa de evasão média por campus × área CINE (células com menos de `min_obs` ficam vazias)."""
    piv = pm.pivot_table(index="campus", columns="area", values="taxa_evasao_real", aggfunc="mean") * 100
    cnt = pm.pivot_table(index="campus", columns="area", values="taxa_evasao_real", aggfunc="size")
    piv = piv.where(cnt >= min_obs).reindex(index=[c for c in ordem if c in piv.index])
    piv = piv.dropna(how="all").dropna(axis=1, how="all")
    if piv.empty:
        return
    fig, ax = plt.subplots(figsize=(FW, 0.3 * len(piv) + 1.6))
    im = ax.imshow(piv.values, cmap="Reds", aspect="auto")
    ax.set_xticks(range(piv.shape[1]), [textwrap.fill(AREAS_CINE.get(a, str(a)), 11) for a in piv.columns],
                  fontsize=7)
    ax.set_yticks(range(piv.shape[0]), piv.index)
    vmax = np.nanmax(piv.values)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v > vmax * 0.7 else "#222")
    ax.grid(False)
    ax.spines[:].set_visible(False)
    fig.colorbar(im, label="Taxa de evasão média (%)")
    save(fig, "03_campus", "heatmap_campus_area_curso")


# ------------------------------------------------------------------------------ 4. perfil do curso
def _barras_media(pm, col, rotulos, nome, rot, largura):
    """Taxa de evasão média (IC 95%) por categoria de `col`, ordenada da maior para a menor."""
    g = pm.groupby(col)["taxa_evasao_real"].agg(["mean", "size", _ic95]).sort_values("mean", ascending=False)
    if len(g) < 2:
        return
    labels = [f"{textwrap.fill(rotulos(i), 12)}\n(n={n})" for i, n in zip(g.index, g["size"])]
    media_geral = pm["taxa_evasao_real"].mean() * 100
    fig, ax = plt.subplots(figsize=(largura, FW * 0.5))
    cores = [RED if v * 100 >= media_geral else BLUE for v in g["mean"]]
    b = ax.bar(labels, g["mean"] * 100, yerr=g["_ic95"] * 100, capsize=3, color=cores, error_kw={"lw": 0.9})
    ax.bar_label(b, labels=[f"{v * 100:.1f}%" for v in g["mean"]], padding=6, fontsize=7.5)
    ax.axhline(media_geral, color="#555", ls="--", lw=0.9)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_ylabel(f"Taxa de evasão média {rot} (%)\n(barra: IC 95%; tracejado: média do IFMG)")
    ax.grid(axis="y")
    save(fig, "04_perfil_curso", nome)


def plot_perfil_curso(pm, rot):
    _barras_media(pm, "grau", str, "grau_academico", rot, FW * 0.6)
    _barras_media(pm, "area", lambda a: AREAS_CINE.get(a, str(a)), "area_cine", rot, FW)


# ------------------------------------------------------------------------------ 5. idade
def plot_idade(pm, rot):
    faixas = [("p_idade_0_24", "Proporção de matriculados até 24 anos", "idade_ate_24"),
              ("p_idade_35_49", "Proporção de matriculados de 35 a 49 anos", "idade_35_49"),
              ("p_idade_50_mais", "Proporção de matriculados com 50 anos ou mais", "idade_50_mais")]
    for col, rotulo, nome in faixas:
        if col not in pm.columns:
            continue
        s = pm[[col, "taxa_evasao_real"]].dropna()
        if s[col].nunique() < 4:
            print(f"  (idade: {col} sem variação suficiente)")
            continue
        # Muitos cursos têm proporção 0 (ex.: sem matriculados 50+): os quartis repetem bordas e
        # duplicates="drop" reduz o nº de grupos; por isso os rótulos são montados depois.
        s["q"] = pd.qcut(s[col], 4, labels=False, duplicates="drop")
        k = int(s["q"].max()) + 1
        nomes_q = (["Q1 (menor)"] + [f"Q{i}" for i in range(2, k)] + [f"Q{k} (maior)"]) if k > 1 else ["Q1"]
        cats = nomes_q
        dados = [s.loc[s["q"] == i, "taxa_evasao_real"] * 100 for i in range(k)]
        fig, ax = plt.subplots(figsize=(FW * 0.75, FW * 0.5))
        bp = ax.boxplot(dados, patch_artist=True, widths=0.55, showfliers=False,
                        medianprops={"color": "black", "lw": 1.2},
                        whiskerprops={"color": "#666"}, capprops={"color": "#666"})
        for patch in bp["boxes"]:
            patch.set(facecolor=RED, edgecolor="#666")
        ax.plot(range(1, len(cats) + 1), [d.mean() for d in dados], "D", color="white",
                mec="black", ms=4, label="Média")
        ax.set_xticks(range(1, len(cats) + 1), [f"{c}\n(n={len(d)})" for c, d in zip(cats, dados)])
        ax.set_xlabel(rotulo + " (quartis)")
        ax.set_ylabel(f"Taxa de evasão {rot} (%)")
        ax.grid(axis="y")
        ax.legend(loc="upper left")
        save(fig, "05_idade", f"boxplot_{nome}")


# ------------------------------------------------------------------------------ 6. atributos
def plot_importancia(top=15):
    import glob
    for f in glob.glob(f"{CSV}/importancia_consenso_modelos_*.csv"):
        if f.endswith("_artigo.csv"):
            continue
        c = pd.read_csv(f, index_col="atributo").head(top)
        c = c.sort_values("perm_relativa_media")
        fig, ax = plt.subplots(figsize=(FW_MEIA, 0.22 * len(c) + 0.9))
        b = ax.barh(c.index, c["perm_relativa_media"], color="black")
        ax.bar_label(b, fmt="%.2f", padding=2, fontsize=6.5, fontweight="bold")
        ax.set_xlim(min(0, c["perm_relativa_media"].min() * 1.2), c["perm_relativa_media"].max() * 1.22)
        ax.set_xlabel("Importância relativa por permutação\n(média dos modelos; máx. = 1)")
        ax.grid(axis="x")
        nome = os.path.basename(f)[len("importancia_consenso_modelos_"):-4]
        save(fig, "06_atributos", f"importancia_{nome}")


def plot_correlacao(top=20):
    f = f"{CSV}/eda_correlacao_com_alvo.csv"
    if not os.path.exists(f):
        return
    c = pd.read_csv(f, index_col="atributo").head(top).sort_values("spearman_alvo")
    cores = [RED if v > 0 else BLUE for v in c["spearman_alvo"]]
    fig, ax = plt.subplots(figsize=(FW_MEIA, 0.22 * len(c) + 1.0))
    b = ax.barh(c.index, c["spearman_alvo"], color=cores)
    ax.bar_label(b, fmt="%.2f", padding=2, fontsize=6.5)
    lim = c["spearman_alvo"].abs().max() * 1.3
    ax.set_xlim(-lim, lim)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Correlação de Spearman com a taxa\nde evasão (vermelho: maior; azul: menor)")
    ax.grid(axis="x")
    save(fig, "06_atributos", "correlacao_spearman_alvo")


# ------------------------------------------------------------------------------ 7. subgrupos
def plot_subgrupos(modelo="Random Forest", f1_geral=None):
    """
    Dois painéis com o mesmo eixo de subgrupos:
      esquerda: taxa de evasão real média (%) -> diz ONDE se evade mais;
      direita : F1-macro do modelo            -> diz ONDE o modelo acerta melhor.
    Linhas de referência: média geral de evasão (esq.), F1 geral do modelo e F1 do acaso (dir.).
    """
    f = f"{CSV}/avaliacao_recortes_subgrupos.csv"
    if not os.path.exists(f):
        return
    df = pd.read_csv(f)
    aliases = {"modelo": ["modelo", "model", "Modelo", "Model"],
               "Dimensao": ["Dimensao", "dimensao", "Dimensão"],
               "Subgrupo": ["Subgrupo", "subgrupo"],
               "N_Cursos_Ano": ["N_Cursos_Ano", "n_cursos_ano", "n"],
               "Taxa_Evasao_Real_Media": ["Taxa_Evasao_Real_Media", "taxa_evasao_real_media"],
               "F1-macro": ["F1-macro", "f1_macro", "F1_macro", "F1_Macro"]}
    ren = {}
    for padrao, opcoes in aliases.items():
        for o in opcoes:
            if o in df.columns and padrao not in df.columns:
                ren[o] = padrao
                break
    df = df.rename(columns=ren)
    faltam = [c for c in aliases if c != "modelo" and c not in df.columns]
    if faltam:
        print(f"  (subgrupos: colunas ausentes {faltam}; colunas do arquivo: {list(df.columns)})")
        return
    rot_modelo = modelo
    if "modelo" in df.columns:
        if modelo not in df["modelo"].unique():
            rot_modelo = df["modelo"].iloc[0]
        df = df[df["modelo"] == rot_modelo]

    dims = {"Campus": "campus", "Grau_Academico": "grau_academico", "Perfil_Idade": "idade",
            "Perfil_Ingresso_Cotas": "ingresso_cotas", "Perfil_Cor_Raca": "cor_raca",
            "Area_Curso": "area_curso"}
    for dim, nome in dims.items():
        s = df[df["Dimensao"] == dim].sort_values("Taxa_Evasao_Real_Media", ascending=False).copy()
        if s.empty:
            continue
        if dim == "Area_Curso":
            s["Subgrupo"] = s["Subgrupo"].map(lambda v: AREAS_CINE.get(int(str(v).split("_")[-1]), v)
                                              if str(v).startswith("Área_") else v)
        labels = [f"{short(a, 30)} (n={n})" for a, n in zip(s["Subgrupo"], s["N_Cursos_Ano"])]
        evas = s["Taxa_Evasao_Real_Media"].values * 100
        f1 = s["F1-macro"].values
        y = np.arange(len(s))
        # média geral ponderada pelo nº de cursos × ano (= média de todas as observações)
        evas_geral = np.average(evas, weights=s["N_Cursos_Ano"])

        fig, (a1, a2) = plt.subplots(1, 2, sharey=True, figsize=(FW, 0.36 * len(s) + 1.3),
                                     gridspec_kw={"wspace": 0.08})
        b1 = a1.barh(y, evas, 0.65, color=RED)
        a1.bar_label(b1, fmt="%.1f", padding=2, fontsize=7)
        a1.axvline(evas_geral, color="#555", ls="--", lw=0.9)
        a1.set_yticks(y, labels)
        a1.invert_yaxis()
        a1.set_xlim(0, evas.max() * 1.22)
        a1.set_xlabel("Taxa de evasão real média (%)\n(tracejado: média geral)")
        a1.set_title("Onde se evade mais", loc="left", pad=26)
        a1.grid(axis="x")

        b2 = a2.barh(y, f1, 0.65, color=BLUE)
        a2.bar_label(b2, fmt="%.2f", padding=2, fontsize=7)
        a2.axvline(1 / 3, color="gray", ls=":", lw=1, label="Acaso (0,33)")
        if f1_geral is not None:
            a2.axvline(f1_geral, color="#555", ls="--", lw=0.9, label=f"F1 geral ({f1_geral:.2f})")
        a2.set_xlim(0, 1)
        a2.set_xlabel(f"F1-macro – {rot_modelo}")
        a2.set_title("Onde o modelo acerta mais", loc="left", pad=26)
        a2.tick_params(axis="y", left=False)
        a2.grid(axis="x")
        a2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), borderaxespad=0.3, ncol=2, fontsize=6.5)
        save(fig, "07_subgrupos", f"subgrupo_{nome}")


# ------------------------------------------------------------------------------ 8. grupos temáticos
def plot_grupos_tematicos(suffix=""):
    """F1-macro de cada grupo temático isolado, por modelo (painel de meia largura)."""
    f = f"{CSV}/desempenho_por_grupo_tematico{suffix}.csv"
    if not os.path.exists(f):
        return
    df = pd.read_csv(f)
    pivot = df.pivot(index="dimensao_tematica", columns="modelo", values="F1-macro")
    pivot = pivot[[m for m in MODEL_ORDER if m in pivot.columns]]
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=True).index]

    fig, ax = plt.subplots(figsize=(FW_MEIA, 0.45 * len(pivot.index) + 1.1))
    y = np.arange(len(pivot.index))
    n_models = len(pivot.columns)
    h = 0.8 / n_models
    for i, modelo in enumerate(pivot.columns):
        off = (i - n_models / 2) * h + h / 2
        b = ax.barh(y + off, pivot[modelo], h, color=MODEL_COLORS.get(modelo), label=modelo)
        ax.bar_label(b, fmt="%.2f", padding=1.5, fontsize=6)
    ax.axvline(1 / 3, color="gray", ls=":", lw=1, label="Acaso (0,33)")
    ax.set_yticks(y, [textwrap.fill(str(g), 16) for g in pivot.index])
    ax.set_xlabel("F1-macro médio (5 partições)")
    ax.set_xlim(0, min(1.0, pivot.max().max() + 0.12))
    ax.grid(axis="x")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), borderaxespad=0.3, fontsize=6.5)
    save(fig, "08_grupos_tematicos", "poder_preditivo_grupos_tematicos")


# ------------------------------------------------------------------------------ orquestração
def run_all(meta, design="D0", scenario=SCENARIO, suffix="", modelo=None):
    pred = load_predictions(suffix)
    mpath = f"{CSV}/metrics_cv_all_models{suffix}.csv"
    metrics = pd.read_csv(mpath) if os.path.exists(mpath) else None
    sc = scenario if scenario in pred["scenario"].unique() else pred["scenario"].unique()[-1]
    modelo = modelo or (best_model(metrics, sc) if metrics is not None else "Random Forest")
    modelos_sc = pred.loc[pred["scenario"] == sc, "modelo"].unique()
    modelo_campus = MODELO_CAMPUS if MODELO_CAMPUS in modelos_sc else modelo
    print(f"Predições carregadas: {len(pred)} observações; cenários: {pred['scenario'].unique()}; "
          f"Modelos: {pred['modelo'].unique()} - usando {modelo}")
    print(f"Desenho {design} | cenário: {sc} | modelo de referência: {modelo} | campus: {modelo_campus}")
    rot = "no ano" if design == "D0" else "no ano seguinte (t+1)"

    for classe in ("alto", "médio", "baixo"):
        plot_roc(pred, classe, sc, f"roc_risco_{classe.replace('é', 'e')}")
    if metrics is not None:
        plot_metricas(metrics, sc)

    pm = preds_modelo(pred, meta, sc, modelo)
    pm_campus = pm if modelo_campus == modelo else preds_modelo(pred, meta, sc, modelo_campus)

    # Base pequena (~240 cursos × ano): o mesmo nome de curso aparece em poucos campi
    plot_top_cursos(pm, nivel="nome", min_obs=10)
    plot_top_cursos(pm, nivel="curso_campus", min_obs=3)
    plot_campus(pm_campus, rot)                   # sempre Random Forest
    plot_perfil_curso(pm, rot)
    plot_idade(pm, rot)
    plot_importancia()
    plot_correlacao()
    plot_subgrupos(modelo)
    plot_grupos_tematicos(suffix)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="D0", choices=["D0", "D1"])
    ap.add_argument("--suffix", default="")
    ap.add_argument("--scenario", default=SCENARIO)
    ap.add_argument("--modelo", default=None,
                    help='modelo de referência dos demais gráficos (use aspas: --modelo "Random Forest")')
    a = ap.parse_args()
    OUT = os.path.join("data/results/figures_tcc", a.design)
    set_style()
    from src.cleaning import cleaning_data
    from src.utils import assemble_attributes
    kw = {"target_include": True} if "target_include" in inspect.signature(cleaning_data).parameters else {}
    d = cleaning_data(**kw).reset_index(drop=True)
    X = assemble_attributes(d)
    run_all(build_meta(d, X), a.design, a.scenario, a.suffix, a.modelo)


if __name__ == "__main__":
    main()