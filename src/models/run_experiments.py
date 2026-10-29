import os
from functools import partial

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

from src.plot_results import (
    plot_eda_target_correlation,
    plot_selection_methods_comparison,
    plot_thematic_groups,
    plot_subgroups_panel,
    plot_continuous_distributions,
)
from src.cleaning import cleaning_data
from src.utils import assemble_attributes, ALVO, MISSING, REGIAO, SEED
from src.models.ml_models_train import define_models, apply_selection, validate_model

os.makedirs("data/results/csv", exist_ok=True)
os.makedirs("data/results/figures", exist_ok=True)

MIN_N_SUBGRUPO = 50   # subgrupos menores que isso são ignorados (F1-macro instável)


# ==============================================================================
# 1. ANÁLISE DE CORRELAÇÃO DAS VARIÁVEIS DO EDA
# ==============================================================================
def analyze_correlations(d: pd.DataFrame, X: pd.DataFrame, eda_cols: list):
    """
    Análise EXPLORATÓRIA (usa a base inteira). Não derive `eda_cols` destas correlações: escolher
    atributos olhando a correlação com o alvo em todos os dados é seleção com vazamento e
    favoreceria a estratégia "EDA" na comparação com os demais métodos.
    """
    print("\n--- 1. Analisando Correlações (EDA) ---")
    # Substitui o marcador de ausência por NaN para não enviesar a correlação
    X_clean = X[eda_cols].replace(MISSING, np.nan).copy()
    y_target = d[ALVO]

    corr_target = pd.DataFrame({
        "pearson_alvo": X_clean.apply(lambda col: col.corr(y_target, method="pearson")),
        "spearman_alvo": X_clean.apply(lambda col: col.corr(y_target, method="spearman")),
        "pct_preenchido": X_clean.notna().mean() * 100
    }).sort_values(by="spearman_alvo", key=abs, ascending=False)

    corr_target.to_csv("data/results/csv/eda_correlacao_com_alvo.csv", index_label="atributo")
    print("Top 10 maiores correlações (Spearman) com a evasão:")
    print(corr_target.head(10).round(3).to_string())

    # Matriz de correlação entre as próprias variáveis do EDA (Spearman)
    corr_matrix = X_clean.corr(method="spearman")
    corr_matrix.to_csv("data/results/csv/eda_matriz_correlacao_spearman.csv")

    # Gráfico da matriz de correlação das top 15 variáveis mais correlacionadas com o alvo
    top_cols = corr_target.head(15).index.tolist()
    top_corr = X_clean[top_cols].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(top_corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(top_cols)), top_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(top_cols)), top_cols)
    for i in range(len(top_cols)):
        for j in range(len(top_cols)):
            ax.text(j, i, f"{top_corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, label="Correlação de Spearman")
    ax.set_title("Matriz de Correlação - Top 15 Variáveis EDA")
    fig.tight_layout()
    fig.savefig("data/results/figures/eda_matriz_correlacao.png", dpi=150)
    plt.close(fig)

    return corr_target


# ==============================================================================
# 2 & 3. RODAR MODELOS COM EDA E OUTROS MÉTODOS DE SELEÇÃO DE ATRIBUTOS
# ==============================================================================
def _apply_selector(models, make_selector):
    """
    Injeta um seletor de atributos (criado por `make_selector()`) antes do classificador de cada modelo.
    O seletor fica dentro do pipeline, então é refeito em cada fold do CV (sem vazamento).
    """
    def wrap(factory):
        def new():
            est = factory()
            sel = make_selector()
            if isinstance(est, Pipeline):
                return Pipeline(est.steps[:-1] + [("sel", sel)] + est.steps[-1:])
            return Pipeline([("sel", sel), ("clf", est)])
        return new

    output = {}
    for name, (factory, grid) in models.items():
        if not isinstance(factory(), Pipeline):
            grid = {f"clf__{k}": v for k, v in grid.items()}
        output[name] = (wrap(factory), grid)
    return output


def apply_mutual_info_selection(models, topk=12):
    """Método alternativo 2: seleção por Informação Mútua (SelectKBest)."""
    return _apply_selector(models, lambda: SelectKBest(
        score_func=partial(mutual_info_classif, discrete_features=False, random_state=SEED),
        k=topk))


def apply_rfe_selection(models, topk=12):
    """Método alternativo 3: Recursive Feature Elimination (RFE) com Árvore de Decisão."""
    return _apply_selector(models, lambda: RFE(
        estimator=DecisionTreeClassifier(max_depth=6, random_state=SEED),
        n_features_to_select=topk,
        step=0.2))


def compare_selection_methods(d: pd.DataFrame, X: pd.DataFrame, eda_cols: list):
    print("\n--- 2 & 3. Comparando EDA vs Outros Métodos de Seleção de Atributos ---")
    all_cols = list(X.columns)
    base_models = define_models()
    topk = 12

    # (nome_estrategia, colunas_usadas, dicionario_modelos)
    strategies = [
        ("1_Selecionadas_EDA", eda_cols, base_models),
        ("2_Todas_Sem_Selecao", all_cols, base_models),
        ("3_SelectFromModel_RF", all_cols, apply_selection(base_models, topk=topk)),
        ("4_Mutual_Information", all_cols, apply_mutual_info_selection(base_models, topk=topk)),
        ("5_RFE_DecisionTree", all_cols, apply_rfe_selection(base_models, topk=topk)),
    ]

    results, oofs_eda = [], []
    for strat_name, cols, models_dict in strategies:
        for model_name, (factory, grid) in models_dict.items():
            m, _, oof = validate_model(d, X, cols, model_name, factory, grid, with_importance=False)
            m.insert(0, "metodo_selecao", strat_name)
            results.append(m)
            if strat_name == "1_Selecionadas_EDA":
                oof.insert(0, "modelo", model_name)
                oofs_eda.append(oof)
            print(f"[{strat_name}] {model_name} -> F1-macro: {m['F1-macro'].mean():.3f}")

    df_res = pd.concat(results, ignore_index=True)
    df_res.to_csv("data/results/csv/comparacao_metodos_selecao.csv", index=False)

    summary = df_res.groupby(["metodo_selecao", "modelo"])[["acurácia", "F1-macro", "AUC OvR", "kappa quad."]].mean().round(3)
    summary.to_csv("data/results/csv/resumo_metodos_selecao.csv")
    print("\nResumo Comparativo (Média dos 5 Folds):")
    print(summary.to_string())

    return pd.concat(oofs_eda, ignore_index=True)


# ==============================================================================
# 4. SEPARAÇÃO POR DIMENSÕES (REGIÃO, IDADE, INGRESSO, CURSO, MODALIDADE, COR)
# ==============================================================================
def _perfil_binario(X: pd.DataFrame, col: str, rotulo_sim: str, rotulo_nao: str, limiar: float = 0.5) -> pd.Series:
    """Classifica o curso pelo limiar de uma proporção; ausentes (coluna inexistente ou MISSING) viram 'ND'."""
    if col not in X.columns:
        return pd.Series("ND", index=X.index)
    s = X[col].replace(MISSING, np.nan)
    return pd.Series(np.where(s.isna(), "ND", np.where(s >= limiar, rotulo_sim, rotulo_nao)), index=X.index)


def thematic_and_subgroup_analysis(d: pd.DataFrame, X: pd.DataFrame, oof_predictions: pd.DataFrame):
    print("\n--- 4A. Modelagem por Grupos Temáticos de Variáveis ---")
    thematic_groups = {
        "Regiao": [c for c in X.columns if c.startswith("reg_")],
        "Idade": [c for c in X.columns if "idade" in c],
        "Formas_Ingresso_Apoio": [c for c in X.columns if any(k in c for k in ["reserva_", "financiamento", "apoio_social", "ingress_", "ocupacao_", "concorrencia"])],
        "Cursos_Area_Grau": [c for c in X.columns if c.startswith(("area_", "grau_")) or c in ["log_matriculas", "log_vagas", "concl_sobre_matric", "p_noturno", "p_feminino"]],
        "Modalidade_Ensino": [c for c in X.columns if c in ["ead", "n_locais_oferta"]],
        "Cor_Raca": [c for c in X.columns if any(k in c for k in ["branca", "preta_parda", "amarela", "indigena"]) and c.startswith("p_")],
    }

    modelos_tematicos = {k: v for k, v in define_models().items() if k in ["Random Forest", "Árvore de Decisão", "MLP"]}
    theme_metrics = []
    for theme, cols in thematic_groups.items():
        if not cols:
            continue
        for model_name, (factory, grid) in modelos_tematicos.items():
            m, _, _ = validate_model(d, X, cols, model_name, factory, grid, with_importance=False)
            m.insert(0, "dimensao_tematica", theme)
            m.insert(1, "n_atributos", len(cols))
            theme_metrics.append(m)

    df_theme = pd.concat(theme_metrics, ignore_index=True)
    resumo_tema = df_theme.groupby(["dimensao_tematica", "modelo"])[["n_atributos", "acurácia", "F1-macro", "AUC OvR"]].mean().round(3)
    resumo_tema.to_csv("data/results/csv/desempenho_por_grupo_tematico.csv")
    print("\nPoder preditivo isolado por Dimensão Temática:")
    print(resumo_tema.to_string())

    # --------------------------------------------------------------------------
    print("\n--- 4B. Desempenho Desagregado por Recortes (Subgrupos na Validação) ---")
    # Metadados de 'd' e 'X' (uma linha por id_ies x id_curso x ano), unidos às previsões OOF
    meta = pd.DataFrame({
        "id_ies": d["id_ies"],
        "id_curso": d["id_curso"],
        "ano": d["ano"],
        "Regiao": d["sigla_uf_ies"].map(REGIAO).fillna("ND"),
        "Modalidade": np.where(d["tipo_modalidade_ensino"] == 2, "EAD", "Presencial"),
        "Area_Curso": "Área_" + d["id_area_geral"].fillna(-1).astype(int).astype(str),
        "Perfil_Idade": _perfil_binario(X, "p_idade_0_24", "Jovem (≥50% até 24a)", "Adulto (<50% até 24a)"),
        "Perfil_Ingresso_Cotas": _perfil_binario(X, "p_reserva_vaga", "Maioria Cotas (≥50%)", "Ampla/Misto (<50%)"),
        "Perfil_Cor_Raca": _perfil_binario(X, "p_preta_parda", "Maioria Preta/Parda (≥50%)", "Outras/Misto (<50%)"),
    })

    # As previsões OOF repetem cada curso x ano uma vez por modelo; a chave em 'meta' é única.
    oof_merged = oof_predictions.merge(meta, on=["id_ies", "id_curso", "ano"], how="left", validate="many_to_one")

    slice_cols = ["Regiao", "Modalidade", "Area_Curso", "Perfil_Idade", "Perfil_Ingresso_Cotas", "Perfil_Cor_Raca"]
    slice_rows = []

    for sc in slice_cols:
        for (modelo, categoria), grp in oof_merged.groupby(["modelo", sc]):
            if len(grp) < MIN_N_SUBGRUPO:
                continue
            slice_rows.append({
                "Dimensao": sc,
                "Subgrupo": categoria,
                "modelo": modelo,
                "N_Cursos_Ano": len(grp),
                "Taxa_Evasao_Real_Media": grp["taxa_evasao_real"].mean(),
                "acurácia": accuracy_score(grp["risco_real"], grp["risco_previsto"]),
                "F1-macro": f1_score(grp["risco_real"], grp["risco_previsto"], average="macro", zero_division=0),
                "kappa": cohen_kappa_score(grp["risco_real"], grp["risco_previsto"])
            })

    df_slices = pd.DataFrame(slice_rows).round(3)
    df_slices.to_csv("data/results/csv/avaliacao_recortes_subgrupos.csv", index=False)
    print("\nArquivo 'avaliacao_recortes_subgrupos.csv' salvo com sucesso!")
    return df_slices


# Atributos escolhidos no EDA. NÃO incluir taxa_evasao_t, taxa_trancado_t nem taxa_desvinc_t:
# são o próprio alvo (ou seus componentes) no desenho t -> t. O histórico do curso entra via
# taxa_evasao_t1 (ano anterior).
EDA_CANDIDATAS = [
    "taxa_evasao_t1", "diff_taxa_curso_ies",
    "log_matriculas", "log_vagas", "concorrencia", "ocupacao_vagas",
    "ingress_sobre_matric", "concl_sobre_matric", "ead", "n_locais_oferta",
    "p_feminino", "p_noturno", "p_idade_0_24", "p_idade_35_49", "p_idade_50_mais",
    "p_branca", "p_preta_parda", "p_reserva_vaga", "p_reserva_rede_publica",
    "p_reserva_renda", "p_apoio_social", "p_extracurricular",
]

if __name__ == "__main__":
    # Carrega a base (todos os anos, alvo no mesmo ano) e monta os atributos
    d = cleaning_data().reset_index(drop=True)
    X = assemble_attributes(d)

    ausentes = [c for c in EDA_CANDIDATAS if c not in X.columns]
    if ausentes:
        print(f"AVISO: {len(ausentes)} atributo(s) do EDA não existem em X e foram descartados: {ausentes}")
    eda_selected_cols = [c for c in EDA_CANDIDATAS if c in X.columns]

    # Etapa 1: Correlação (exploratória)
    corr_target = analyze_correlations(d, X, eda_selected_cols)

    # Etapas 2 e 3: Modelos com EDA vs Outros Métodos de Seleção
    oof_eda = compare_selection_methods(d, X, eda_selected_cols)

    # Etapa 4: Recortes por Região, Idade, Ingresso, Cursos, Modalidade e Cor
    thematic_and_subgroup_analysis(d, X, oof_eda)

    # Gráficos em data/results/figures/
    plot_eda_target_correlation(corr_target, top_n=20)
    plot_selection_methods_comparison()
    plot_thematic_groups()
    plot_subgroups_panel(best_model_name="Random Forest")
    plot_continuous_distributions(d, X, REGIAO)