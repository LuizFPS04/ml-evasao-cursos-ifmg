import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import ALVO, MISSING

os.makedirs("data/results/figures", exist_ok=True)

# Mapeamento oficial CINE Brasil (INEP) para deixar o gráfico de Cursos legível
AREAS_CINE = {
    "Área_0": "0 - Programas Básicos",
    "Área_1": "1 - Educação",
    "Área_2": "2 - Artes e Humanidades",
    "Área_3": "3 - Ciências Sociais, Comunicação e Info.",
    "Área_4": "4 - Negócios, Administração e Direito",
    "Área_5": "5 - Ciências Naturais, Matemática e Estat.",
    "Área_6": "6 - Computação e TIC",
    "Área_7": "7 - Engenharia, Produção e Construção",
    "Área_8": "8 - Agricultura, Silvicultura, Pesca e Vet.",
    "Área_9": "9 - Saúde e Bem-Estar",
    "Área_10": "10 - Serviços",
    "Área_-1": "Não Informado"
}


def _boxplot(ax, data, labels, color):
    """Boxplot compatível com Matplotlib < 3.9 (labels) e >= 3.9 (tick_labels)."""
    try:
        ax.boxplot(data, tick_labels=labels, patch_artist=True, boxprops=dict(facecolor=color))
    except TypeError:
        ax.boxplot(data, labels=labels, patch_artist=True, boxprops=dict(facecolor=color))


# ==============================================================================
# GRÁFICO 1: CORRELAÇÃO DAS VARIÁVEIS DO EDA COM O ALVO (BARRAS HORIZONTAIS)
# ==============================================================================
def plot_eda_target_correlation(corr_target: pd.DataFrame, top_n: int = 20):
    df_plot = corr_target.head(top_n).sort_values(by="spearman_alvo", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 0.35 * len(df_plot) + 1.5))
    colors = ["#d95f02" if v > 0 else "#1b9e77" for v in df_plot["spearman_alvo"]]

    bars = ax.barh(df_plot.index, df_plot["spearman_alvo"], color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Correlação de Spearman com a Taxa de Evasão (mesmo ano)")
    ax.set_title(f"Top {len(df_plot)} Variáveis (EDA) Mais Correlacionadas com a Evasão")

    for bar in bars:
        w = bar.get_width()
        offset = 0.01 if w >= 0 else -0.01
        ha = "left" if w >= 0 else "right"
        ax.text(w + offset, bar.get_y() + bar.get_height()/2, f"{w:.2f}", va="center", ha=ha, fontsize=8)

    ax.set_xlim(min(df_plot["spearman_alvo"].min() - 0.1, -0.1), max(df_plot["spearman_alvo"].max() + 0.1, 0.1))
    fig.tight_layout()
    fig.savefig("data/results/figures/1_eda_correlacao_alvo_barras.png", dpi=300)
    plt.close(fig)
    print("Gráfico salvo: 1_eda_correlacao_alvo_barras.png")


# ==============================================================================
# GRÁFICO 2: COMPARAÇÃO DE MÉTODOS DE SELEÇÃO DE ATRIBUTOS (EDA vs OUTROS)
# ==============================================================================
def plot_selection_methods_comparison():
    df = pd.read_csv("data/results/csv/comparacao_metodos_selecao.csv")
    pivot_f1 = df.groupby(["modelo", "metodo_selecao"])["F1-macro"].mean().unstack()
    pivot_std = df.groupby(["modelo", "metodo_selecao"])["F1-macro"].std().unstack()

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(pivot_f1.index))
    n_methods = len(pivot_f1.columns)
    width = 0.8 / n_methods

    cmap = plt.get_cmap("Set2")
    for i, col in enumerate(pivot_f1.columns):
        offset = (i - n_methods / 2) * width + width / 2
        vals = pivot_f1[col].values
        errs = pivot_std[col].values
        rects = ax.bar(x + offset, vals, width, yerr=errs, capsize=3, label=col, color=cmap(i), edgecolor="black", linewidth=0.5)
        for r in rects:
            h = r.get_height()
            ax.text(r.get_x() + r.get_width()/2, h + 0.015, f"{h:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_ylabel("F1-Macro (Média ± DP nos 5 Folds)")
    ax.set_title("Desempenho dos Modelos: Variáveis EDA vs Outros Métodos de Seleção")
    ax.set_xticks(x, pivot_f1.index)
    ax.set_ylim(0, min(1.05, pivot_f1.max().max() + 0.2))
    ax.legend(title="Método de Seleção", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig("data/results/figures/2_comparacao_metodos_selecao.png", dpi=300)
    plt.close(fig)
    print("Gráfico salvo: 2_comparacao_metodos_selecao.png")


# ==============================================================================
# GRÁFICO 3: PODER PREDITIVO ISOLADO POR GRUPO TEMÁTICO (6 DIMENSÕES)
# ==============================================================================
def plot_thematic_groups():
    df = pd.read_csv("data/results/csv/desempenho_por_grupo_tematico.csv")
    pivot = df.pivot(index="dimensao_tematica", columns="modelo", values="F1-macro")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=True).index]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = np.arange(len(pivot.index))
    n_models = len(pivot.columns)
    height = 0.75 / n_models

    for i, model in enumerate(pivot.columns):
        offset = (i - n_models / 2) * height + height / 2
        bars = ax.barh(y + offset, pivot[model], height, label=model)
        for b in bars:
            w = b.get_width()
            ax.text(w + 0.005, b.get_y() + b.get_height()/2, f"{w:.3f}", va="center", ha="left", fontsize=7.5)

    ax.axvline(0.333, color="red", linestyle=":", linewidth=1, label="Aleatório (3 classes = 0.33)")
    ax.set_yticks(y, pivot.index)
    ax.set_xlabel("F1-Macro Médio (5 Folds)")
    ax.set_title("Poder Preditivo Isolado por Grupo Temático de Variáveis")
    ax.set_xlim(0, min(1.0, pivot.max().max() + 0.1))
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig("data/results/figures/3_poder_preditivo_grupos_tematicos.png", dpi=300)
    plt.close(fig)
    print("Gráfico salvo: 3_poder_preditivo_grupos_tematicos.png")


# ==============================================================================
# GRÁFICO 4: PAINEL DAS 6 DIMENSÕES (TAXA DE EVASÃO REAL vs F1-MACRO DO MODELO)
# ==============================================================================
def plot_subgroups_panel(best_model_name: str = "Random Forest"):
    df = pd.read_csv("data/results/csv/avaliacao_recortes_subgrupos.csv")
    df["Subgrupo"] = df["Subgrupo"].replace(AREAS_CINE)

    # Filtra pelo modelo de referência (ou tira a média entre os modelos se não encontrar)
    if best_model_name in df["modelo"].unique():
        df_mod = df[df["modelo"] == best_model_name].copy()
        titulo_mod = best_model_name
    else:
        df_mod = df.groupby(["Dimensao", "Subgrupo"], as_index=False)[
            ["N_Cursos_Ano", "Taxa_Evasao_Real_Media", "acurácia", "F1-macro", "kappa"]
        ].mean()
        titulo_mod = "Média dos Modelos"

    dimensoes = [
        ("Regiao", "Região Geográfica"),
        ("Modalidade", "Modalidade de Ensino"),
        ("Perfil_Idade", "Faixa Etária Predominante"),
        ("Perfil_Ingresso_Cotas", "Forma de Ingresso (Reserva de Vagas)"),
        ("Perfil_Cor_Raca", "Perfil de Cor/Raça"),
        ("Area_Curso", "Área Geral do Curso (CINE Brasil)")
    ]

    fig, axes = plt.subplots(3, 2, figsize=(15, 13))
    axes = axes.flatten()

    for idx, (dim_key, dim_title) in enumerate(dimensoes):
        ax = axes[idx]
        sub = df_mod[df_mod["Dimensao"] == dim_key].sort_values("Taxa_Evasao_Real_Media", ascending=True)
        if sub.empty:
            ax.set_visible(False)
            continue

        y_pos = np.arange(len(sub))
        # Barras: Taxa de Evasão Real Média (%)
        evasao_pct = sub["Taxa_Evasao_Real_Media"] * 100
        bars = ax.barh(y_pos - 0.18, evasao_pct, height=0.35, color="#e41a1c", alpha=0.8, label="Evasão Real Média (%)")

        # Segundo eixo X para o F1-Macro (0 a 100%)
        ax2 = ax.twiny()
        f1_pct = sub["F1-macro"] * 100
        bars2 = ax2.barh(y_pos + 0.18, f1_pct, height=0.35, color="#377eb8", alpha=0.8, label=f"F1-Macro {titulo_mod} (%)")

        ax.set_yticks(y_pos, [f"{s} (n={n})" for s, n in zip(sub["Subgrupo"], sub["N_Cursos_Ano"])], fontsize=8.5)
        ax.set_xlabel("Taxa de Evasão Real Média (%)", color="#b30000", fontsize=8.5)
        ax2.set_xlabel(f"F1-Macro (%) - {titulo_mod}", color="#08519c", fontsize=8.5)
        ax.set_title(dim_title, fontsize=10, fontweight="bold", pad=12)

        ax.set_xlim(0, max(evasao_pct.max() * 1.25, 35))
        ax2.set_xlim(0, 105)

        for b in bars:
            w = b.get_width()
            ax.text(w + 0.5, b.get_y() + b.get_height()/2, f"{w:.1f}%", va="center", fontsize=7.5, color="#b30000")
        for b in bars2:
            w = b.get_width()
            ax2.text(w + 0.8, b.get_y() + b.get_height()/2, f"{w:.1f}%", va="center", fontsize=7.5, color="#08519c")

        ax.grid(axis="x", linestyle=":", alpha=0.4)

    fig.suptitle(f"Análise Estratificada por Dimensões: Taxa de Evasão Real vs Desempenho ({titulo_mod})", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("data/results/figures/4_painel_6_dimensoes_evasao_e_f1.png", dpi=300)
    plt.close(fig)
    print("Gráfico salvo: 4_painel_6_dimensoes_evasao_e_f1.png")


# ==============================================================================
# GRÁFICO 5: DISTRIBUIÇÃO (BOXPLOT) DA EVASÃO CONTÍNUA POR DIMENSÃO NA BASE BRUTA
# ==============================================================================
def plot_continuous_distributions(d: pd.DataFrame, X: pd.DataFrame, regiao_map: dict):
    """
    Mostra a distribuição real (mediana e quartis) da taxa de evasão do ano
    e a relação com Idade, Ingresso (Cotas) e Cor/Raça.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    y_evasao = d[ALVO] * 100  # em %

    # 1. Boxplot por Região
    reg_series = d["sigla_uf_ies"].map(regiao_map).fillna("ND")
    reg_order = ["N", "NE", "CO", "SE", "S"]
    data_reg = [y_evasao[reg_series == r].dropna() for r in reg_order if (reg_series == r).any()]
    labels_reg = [r for r in reg_order if (reg_series == r).any()]
    _boxplot(axes[0, 0], data_reg, labels_reg, "#a6cee3")
    axes[0, 0].set_title("Evasão por Região")
    axes[0, 0].set_ylabel("Taxa de Evasão (%)")

    # 2. Boxplot por Modalidade
    mod_series = np.where(d["tipo_modalidade_ensino"] == 2, "EAD", "Presencial")
    labels_mod = ["Presencial", "EAD"]
    data_mod = [y_evasao[mod_series == m].dropna() for m in labels_mod]
    _boxplot(axes[0, 1], data_mod, labels_mod, "#b2df8a")
    axes[0, 1].set_title("Evasão por Modalidade")

    # 3. Boxplot por Área do Curso (CINE)
    area_series = d["id_area_geral"].fillna(-1).astype(int)
    areas_validas = sorted([a for a in area_series.unique() if a >= 0])
    data_area = [y_evasao[area_series == a].dropna() for a in areas_validas]
    _boxplot(axes[0, 2], data_area, [str(a) for a in areas_validas], "#fb9a99")
    axes[0, 2].set_title("Evasão por Código da Área CINE (0 a 10)")
    axes[0, 2].set_xlabel("Código id_area_geral (INEP)")

    # 4, 5, 6: Evasão por quartis de Idade, Reserva de Vagas (Ingresso) e Cor/Raça
    continuous_vars = [
        ("p_idade_0_24", "Proporção de Alunos até 24 Anos (Idade)", axes[1, 0], "#fdbf6f"),
        ("p_reserva_vaga", "Proporção de Ingresso por Cotas (Reserva)", axes[1, 1], "#cab2d6"),
        ("p_preta_parda", "Proporção de Alunos Pretos/Pardos (Cor/Raça)", axes[1, 2], "#ffff99"),
    ]

    for col_name, title, ax, color in continuous_vars:
        if col_name in X.columns:
            s = X[col_name].replace(MISSING, np.nan)
            valid = s.notna() & y_evasao.notna()
            quartis = pd.qcut(s[valid], q=4, labels=["Q1 (Menor)", "Q2", "Q3", "Q4 (Maior)"], duplicates="drop")
            grupos = [y_evasao[valid][quartis == q] for q in quartis.cat.categories]
            _boxplot(ax, grupos, list(quartis.cat.categories), color)
            ax.set_title(f"Evasão vs {title}")
            ax.set_ylabel("Taxa de Evasão (%)")
            ax.grid(axis="y", linestyle=":", alpha=0.5)

    for ax in axes[0, :]:
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle("Distribuição da Taxa de Evasão Real pelas 6 Dimensões Analisadas", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("data/results/figures/5_boxplots_distribuicao_6_dimensoes.png", dpi=300)
    plt.close(fig)
    print("Gráfico salvo: 5_boxplots_distribuicao_6_dimensoes.png")