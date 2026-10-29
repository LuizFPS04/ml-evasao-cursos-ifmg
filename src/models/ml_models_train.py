"""
Os 5 algoritmos (Árvore de Decisão, Random Forest, kNN, Naive Bayes e MLP) sob o MESMO protocolo (t -> t):
  - StratifiedGroupKFold externo (5 folds) estratificado por risco e agrupado por IES;
  - cortes de risco (quartis 25/75) calculados só no treino de cada fold;
  - ajuste de hiperparâmetros com CV interna (também estratificada e agrupada por IES);
  - métricas por fold + baseline de persistência (taxa do ano anterior), calculada SOMENTE nas
    linhas com histórico válido (ano anterior consecutivo);
  - importância por PERMUTAÇÃO no teste (serve para qualquer algoritmo).

Como os folds são idênticos para todos os modelos, a comparação entre algoritmos é pareada.
"""
import os
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
"""
Importações de modelos e funções do scikit-learn para classificação, seleção de atributos, pré-processamento e avaliação.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.cleaning import cleaning_data
from src.utils import (
    ALVO,
    K_EXTERNO,
    K_INTERNO,
    MISSING,
    SEED,
    TOP_N,
    assemble_attributes,
    attribute_group,
    cut_quartis,
    for_risk,
)

# ------------------------------------------------------------------------------
# Configuração do experimento
# ------------------------------------------------------------------------------
ONLY_COURSE_IMPORTANCE = True   # True = importância só nos cenários de IMPORTANCE_SCENARIOS | False = em todos
IMPORTANCE_SCENARIOS = {"Somente curso", "Completo - Curso + histórico + IES/contexto"}
N_REPEATS = 5                   # repetições do embaralhamento na importância por permutação
N_AMOSTRA_PERM = 3000           # linhas do fold de teste usadas na permutação (limita o custo)

USE_ARTICLE = False             # True = configuração fixa do artigo | False = ajuste de hiperparâmetros
CLASS_WEIGHT = None             # None ou "balanced" (aplicado à Árvore e à Random Forest)

SUFIXO = "_artigo" if USE_ARTICLE else ""
RISK_NAMES = {0: "baixo", 1: "médio", 2: "alto"}


def convert_nan(a):
    """Converte o marcador MISSING (-999) de volta em NaN, para os modelos que precisam de imputação."""
    return np.where(np.asarray(a, dtype=float) == MISSING, np.nan, a)


def not_tree_preprocessing():
    """
    Pré-processamento para os modelos que não são baseados em árvore.
    - MISSING (-999) vira NaN;
    - imputação pela mediana, com indicador de ausência;
    - padronização (StandardScaler), importante para kNN e MLP.
    """
    return [
        ("na", FunctionTransformer(convert_nan)),
        ("imp", SimpleImputer(strategy="median", add_indicator=True)),
        ("esc", StandardScaler()),
    ]


def define_models():
    """
    Define os modelos e suas grades de hiperparâmetros: {nome: (factory, grade)}.

    - Configuração fixa (USE_ARTICLE=True): melhor configuração de Teodoro e Kappel (2020).
      Naive Bayes: sem parâmetros | kNN: K=10 | Árvore: profundidade 18 | RF: profundidade 22,
      70 árvores e todas as características (os dados têm menos de 70 atributos) |
      MLP: 1 camada oculta de 40 neurônios.
    - Ajuste (USE_ARTICLE=False): grades para GridSearchCV com CV interna.

    O paralelismo (n_jobs=-1) fica só no GridSearchCV/permutação; os estimadores usam n_jobs=1
    para não oversubscrever a CPU com paralelismo aninhado.
    """
    if USE_ARTICLE:
        return {
            "Árvore de Decisão": (
                lambda: DecisionTreeClassifier(
                    max_depth=18, class_weight=CLASS_WEIGHT, random_state=SEED), {}),
            "Random Forest": (
                lambda: RandomForestClassifier(
                    n_estimators=70, max_depth=22, max_features=None,
                    class_weight=CLASS_WEIGHT, random_state=SEED, n_jobs=-1), {}),
            "kNN": (lambda: Pipeline(
                not_tree_preprocessing() + [("clf", KNeighborsClassifier(n_neighbors=10))]), {}),
            "Naive Bayes": (lambda: Pipeline(
                not_tree_preprocessing() + [("clf", GaussianNB())]), {}),
            "MLP": (lambda: Pipeline(not_tree_preprocessing() + [
                ("clf", MLPClassifier(
                    hidden_layer_sizes=(40,), early_stopping=True,
                    max_iter=300, random_state=SEED))]), {}),
        }

    return {
        "Árvore de Decisão": (
            lambda: DecisionTreeClassifier(class_weight=CLASS_WEIGHT, random_state=SEED),
            {
                "max_depth": [3, 4, 5, 6, 8],
                "min_samples_leaf": [30, 60, 100],
                "criterion": ["gini", "entropy"],
            }),
        "Random Forest": (
            lambda: RandomForestClassifier(
                n_estimators=150, class_weight=CLASS_WEIGHT, random_state=SEED, n_jobs=1),
            {
                # A grade anterior tinha o ótimo na borda (profundidade 18, folha 5): ampliada.
                "max_depth": [10, 14, 18, 24],
                "min_samples_leaf": [2, 5, 20, 50],
            }),
        "kNN": (
            lambda: Pipeline(not_tree_preprocessing() + [("clf", KNeighborsClassifier())]),
            {
                # k=31 era o menor valor da grade anterior: acrescentado k=15.
                "clf__n_neighbors": [15, 31, 61, 101],
                "clf__weights": ["uniform", "distance"],
                "clf__p": [1, 2],  # 1 = Manhattan, 2 = Euclidiana
            }),
        "Naive Bayes": (
            lambda: Pipeline(not_tree_preprocessing() + [("clf", GaussianNB())]),
            {"clf__var_smoothing": [1e-9, 1e-6, 1e-3]}),
        "MLP": (
            lambda: Pipeline(not_tree_preprocessing() + [
                ("clf", MLPClassifier(early_stopping=True, max_iter=300, random_state=SEED))]),
            {"clf__hidden_layer_sizes": [(32,), (64, 32)], "clf__alpha": [1e-3, 1e-2]}),
    }


def apply_selection(models, topk=10):
    """
    Injeta uma etapa de seleção de atributos baseada em Random Forest (SelectFromModel) nos modelos.

    - models: dict {nome: (factory_fn, grade)}.
    - topk: para kNN, Naive Bayes e MLP (Pipeline) mantém os `topk` atributos mais importantes;
      para Árvore e RF mantém os atributos com importância acima de 0.5 * mediana.
    A seleção fica dentro do pipeline, logo é refeita em cada fold do CV (sem vazamento).
    """
    def wrap(factory, is_pipeline):
        def new():
            sel = SelectFromModel(
                estimator=RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,          # evita que o seletor decore as IES de treino
                    min_samples_leaf=30,  # importância baseada em padrões gerais
                    random_state=SEED,
                    n_jobs=1,
                ),
                threshold=-np.inf if is_pipeline else "0.5*median",
                max_features=topk if is_pipeline else None,
            )
            est = factory()
            if isinstance(est, Pipeline):
                return Pipeline(est.steps[:-1] + [("sel", sel)] + est.steps[-1:])
            return Pipeline([("sel", sel), ("clf", est)])
        return new

    output = {}
    for name, (factory, grid) in models.items():
        is_pipe = isinstance(factory(), Pipeline)
        if not is_pipe:
            grid = {f"clf__{k}": v for k, v in grid.items()}
        output[name] = (wrap(factory, is_pipe), grid)
    return output


def persistence_prediction(d_te, cuts):
    """
    Baseline de persistência: a taxa do ANO ANTERIOR do curso, discretizada com os cortes do treino.

    Só vale para linhas com histórico válido (ano anterior consecutivo e taxa conhecida). As demais
    ficam NaN e NÃO entram no F1 da baseline (antes recebiam a mediana do treino, o que
    distorcia a comparação).

    Returns:
        (pd.Series com o risco previsto ou NaN, pd.Series booleana com as linhas válidas)
    """
    valido = (d_te["ano_ant"] == d_te["ano"] - 1) & d_te["taxa_evasao_ant"].notna()
    pers = pd.Series(np.nan, index=d_te.index)
    if valido.any():
        pers.loc[valido] = for_risk(d_te.loc[valido, "taxa_evasao_ant"], cuts).astype(float)
    return pers, valido


def validate_model(d, X, columns, name, factory, grid, with_importance):
    """
    Validação cruzada estratificada e agrupada por IES, com métricas e importância por permutação.

    - d: DataFrame original (inclui o alvo `taxa_evasao_alvo`, id_ies, id_curso, ano, ano_ant...).
    - X: DataFrame de atributos (mesmo índice de d).
    - columns: colunas de X usadas neste experimento.
    - name: nome do modelo | factory: função que retorna o estimador | grid: grade ({} = sem ajuste).
    - with_importance: calcula importância por permutação em cada fold.

    Returns:
        (métricas por fold, importâncias ou None, previsões fora da amostra)
    """
    groups = d["id_ies"].values
    # A estratificação da divisão usa cortes globais (afeta só a divisão); os rótulos de
    # treino/teste usam cortes calculados no treino de cada fold.
    y_strat = for_risk(d[ALVO], cut_quartis(d[ALVO]))
    external_cv = StratifiedGroupKFold(n_splits=K_EXTERNO, shuffle=True, random_state=SEED)
    rng = np.random.RandomState(SEED)

    rows, perms, oofs = [], [], []
    for k, (tr, te) in enumerate(external_cv.split(X, y_strat, groups=groups), start=1):
        cuts = cut_quartis(d[ALVO].iloc[tr])
        y = for_risk(d[ALVO], cuts)
        Xtr, Xte = X.iloc[tr][columns], X.iloc[te][columns]
        ytr, yte = y.iloc[tr], y.iloc[te]

        if grid:  # Ajuste de hiperparâmetros por CV interna
            internal_cv = StratifiedGroupKFold(n_splits=K_INTERNO, shuffle=True, random_state=SEED)
            gs = GridSearchCV(factory(), grid, cv=internal_cv, scoring="f1_macro", n_jobs=-1)
            gs.fit(Xtr, ytr, groups=groups[tr])
            model, params = gs.best_estimator_, str(gs.best_params_)
        else:     # Configuração fixa (artigo)
            model, params = factory().fit(Xtr, ytr), "fixos (artigo)"

        assert list(model.classes_) == [0, 1, 2], f"Fold {k}: classes inesperadas {model.classes_}"
        pred = model.predict(Xte)
        proba = model.predict_proba(Xte)

        # Baseline de persistência apenas em linhas com histórico válido
        d_te = d.iloc[te]
        pers, valido = persistence_prediction(d_te, cuts)
        if valido.any():
            f1_pers = f1_score(yte[valido], pers[valido].astype(int), average="macro")
            f1_modelo_valido = f1_score(yte[valido], pred[valido.values], average="macro")
        else:
            f1_pers = f1_modelo_valido = np.nan

        rows.append({
            "modelo": name,
            "fold": k,
            "acurácia": accuracy_score(yte, pred),
            "F1-macro": f1_score(yte, pred, average="macro"),
            "AUC OvR": roc_auc_score(yte, proba, multi_class="ovr"),
            "kappa quad.": cohen_kappa_score(yte, pred, weights="quadratic"),
            "F1-macro (hist. válido)": f1_modelo_valido,   # modelo nas MESMAS linhas da persistência
            "F1 persistência": f1_pers,
            "n persistência": int(valido.sum()),
            "parâmetros": params,
        })

        # Previsões FORA DA AMOSTRA: cada linha é predita por um modelo que não viu aquela IES no treino
        oofs.append(pd.DataFrame({
            "id_ies": d_te["id_ies"].values,
            "id_curso": d_te["id_curso"].values,
            "nome_curso": d_te["nome_curso"].values,
            "ano": d_te["ano"].values,
            "fold": k,
            "ano_ant": d_te["ano_ant"].values,
            "taxa_evasao_ant": d_te["taxa_evasao_ant"].where(valido).values,  # NaN se não há ano anterior consecutivo
            "hist_valido": valido.values,
            "taxa_evasao_real": d_te[ALVO].values,
            "corte_baixo_medio": cuts[1],
            "corte_medio_alto": cuts[2],  # quartis do TREINO deste fold
            "risco_real": yte.map(RISK_NAMES).values,
            "risco_previsto": pd.Series(pred).map(RISK_NAMES).values,
            "risco_persistencia": pers.map(lambda v: RISK_NAMES.get(int(v)) if pd.notna(v) else None).values,
            "prob_baixo": proba[:, 0],
            "prob_medio": proba[:, 1],
            "prob_alto": proba[:, 2],
        }))

        if with_importance:
            idx = rng.choice(len(Xte), size=min(N_AMOSTRA_PERM, len(Xte)), replace=False)
            p = permutation_importance(
                model, Xte.iloc[idx], yte.iloc[idx],
                n_repeats=N_REPEATS, random_state=SEED,
                scoring="f1_macro", n_jobs=-1)
            perms.append(pd.Series(p.importances_mean, index=columns, name=f"fold{k}"))

    imp = None
    if with_importance:
        P = pd.concat(perms, axis=1)
        imp = pd.DataFrame({"perm_media": P.mean(axis=1), "perm_dp": P.std(axis=1)})

    print(f"  → {name}: importância por permutação calculada = {imp is not None}")
    return pd.DataFrame(rows), imp, pd.concat(oofs, ignore_index=True)


def summarize_by_year(predictions):
    """
    Métricas por cenário x modelo x ano a partir das previsões fora da amostra.

    Necessário porque a persistência (e o efeito do histórico) varia muito entre anos: em 2020 a
    taxa de 2019 praticamente não explica o alvo, enquanto em 2021 e 2022 explica bem.
    """
    rows = []
    for (sc, mod, ano), g in predictions.groupby(["scenario", "model", "ano"], sort=False):
        v = g[g["hist_valido"]]
        rows.append({
            "scenario": sc,
            "modelo": mod,
            "ano": ano,
            "n": len(g),
            "acurácia": accuracy_score(g["risco_real"], g["risco_previsto"]),
            "F1-macro": f1_score(g["risco_real"], g["risco_previsto"], average="macro"),
            "n hist. válido": len(v),
            "F1-macro (hist. válido)": (
                f1_score(v["risco_real"], v["risco_previsto"], average="macro") if len(v) else np.nan),
            "F1 persistência (hist. válido)": (
                f1_score(v["risco_real"], v["risco_persistencia"], average="macro") if len(v) else np.nan),
        })
    return pd.DataFrame(rows).round(3)


def save_consensus(imps, scenario):
    """
    Cruza a importância por permutação dos modelos em um cenário.

    A importância de cada algoritmo é normalizada para que o maior valor seja 1. Calcula a
    importância relativa média, a posição média (rank) e em quantos modelos o atributo está no top 10.
    """
    match = "_" + scenario.replace(" ", "_").replace("+", "mais").replace("/", "_").replace("-", "")
    suffix = f"{match}{SUFIXO}"
    rel = pd.DataFrame({n: i["perm_media"] / max(i["perm_media"].max(), 1e-9) for n, i in imps.items()})
    rank = pd.DataFrame({n: i["perm_media"].rank(ascending=False) for n, i in imps.items()})
    cons = pd.DataFrame({
        "perm_relativa_media": rel.mean(axis=1),
        "rank_medio": rank.mean(axis=1),
        "modelos_no_top10": (rank <= 10).sum(axis=1),
    }).join(rel.add_prefix("rel_"))
    cons = cons.sort_values(["modelos_no_top10", "perm_relativa_media"], ascending=False)
    print(f"\nAtributos mais consistentes entre os modelos (cenário '{scenario}'; importância relativa = 1 no atributo mais importante de cada modelo):")
    print(cons.head(TOP_N).round(2).to_string())

    cons.to_csv(f"data/results/csv/importancia_consenso_modelos{suffix}.csv", index=True, index_label="atributo")

    try:
        top = cons.head(TOP_N)[[c for c in cons.columns if c.startswith("rel_")]]
        top.columns = [c[4:] for c in top.columns]
        fig, ax = plt.subplots(figsize=(1.3 * len(top.columns) + 4, 0.4 * len(top) + 1.5))
        im = ax.imshow(top.clip(lower=0).values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(top.columns)), top.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(top)), top.index)
        for i in range(top.shape[0]):
            for j in range(top.shape[1]):
                ax.text(j, i, f"{top.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, label="importância relativa (permutação)")
        fig.tight_layout()
        fig.savefig(f"data/results/figures/importancia_consenso_modelos{suffix}.png", dpi=150)
        print(f"Gráfico: importancia_consenso_modelos{suffix}.png")
        plt.close(fig)
    except Exception as e:
        print("Gráfico não gerado:", e)


def main():
    """
    Fluxo completo: limpeza, atributos, validação cruzada de todos os modelos em todos os cenários,
    métricas (geral e por ano), importância por permutação e consenso entre modelos.
    """
    os.makedirs("data/results/csv", exist_ok=True)
    os.makedirs("data/results/figures", exist_ok=True)

    d = cleaning_data().reset_index(drop=True)
    anos = sorted(d["ano"].unique().tolist())
    print(f"Painel t->t: {len(d)} linhas | {d.id_curso.nunique()} cursos | {d.id_ies.nunique()} IES | anos {anos}")
    X = assemble_attributes(d)
    models = define_models()
    if USE_ARTICLE:
        print("Configuração utilizada do artigo de referência (sem ajuste de hiperparâmetros).")

    metrics, importances, predictions = [], {}, []
    for scenario, columns in attribute_group(X).items():
        with_imp = (not ONLY_COURSE_IMPORTANCE) or scenario in IMPORTANCE_SCENARIOS
        for name, (factory, grid) in models.items():
            t0 = time.time()
            m, imp, oof = validate_model(d, X, columns, name, factory, grid, with_imp)
            m.insert(0, "scenario", scenario)
            metrics.append(m)
            oof.insert(0, "scenario", scenario)
            oof.insert(1, "model", name)
            predictions.append(oof)
            if imp is not None:
                importances.setdefault(scenario, {})[name] = imp
            print(f"[{scenario}] {name}: F1-macro {m['F1-macro'].mean():.3f} ± {m['F1-macro'].std():.3f} "
                  f"| AUC {m['AUC OvR'].mean():.3f} | {time.time() - t0:.0f}s", flush=True)
            # Salva o progresso
            pd.concat(metrics).to_csv(f"data/results/csv/metrics_cv_all_models{SUFIXO}.csv", index=False)
            pd.concat(predictions).to_csv(f"data/results/csv/predictions_oof_cv{SUFIXO}.csv", index=False)

    M = pd.concat(metrics, ignore_index=True)
    summary = (
        M.groupby(["scenario", "modelo"], sort=False)[
            ["acurácia", "F1-macro", "AUC OvR", "kappa quad.",
             "F1-macro (hist. válido)", "F1 persistência"]
        ].agg(["mean", "std"]).round(3)
    )
    summary.columns = [f"{a} {b}" for a, b in summary.columns]

    print("\n================ SUMMARY (média e desvio dos 5 folds) ================")
    pd.set_option("display.width", 250)
    print(summary[[
        "acurácia mean", "F1-macro mean", "F1-macro std", "AUC OvR mean",
        "kappa quad. mean", "F1-macro (hist. válido) mean", "F1 persistência mean",
    ]].to_string())
    summary.to_csv(f"data/results/csv/summary_cv_all_models{SUFIXO}.csv")

    by_year = summarize_by_year(pd.concat(predictions, ignore_index=True))
    by_year.to_csv(f"data/results/csv/metricas_por_ano{SUFIXO}.csv", index=False)
    print("\n================ POR ANO (linhas com histórico válido na persistência) ================")
    print(by_year.to_string(index=False))

    print("\n================ IMPORTÂNCIAS ================")
    for scenario, imps in importances.items():
        print(f"Cenário: {scenario}")
        print(f"Modelos: {list(imps.keys())}")
        save_consensus(imps, scenario)


if __name__ == "__main__":
    main()