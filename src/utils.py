import numpy as np
import pandas as pd

SEED = 42              # Semente para reprodutibilidade
K_EXTERNO = 5          # Folds da avaliação
K_INTERNO = 3          # Folds do ajuste de hiperparâmetros
TOP_N = 15             # Quantidade de atributos para análise de importância

MISSING = -999.0       # Marcador de "sem dado" na matriz de atributos (único lugar onde é definido)
ALVO = "taxa_evasao_alvo"  # Taxa de evasão do MESMO ano das features (desenho t -> t)

REGIAO = {
    **dict.fromkeys(["AC", "AM", "AP", "PA", "RO", "RR", "TO"], "N"),
    **dict.fromkeys(["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"], "NE"),
    **dict.fromkeys(["DF", "GO", "MS", "MT"], "CO"),
    **dict.fromkeys(["ES", "MG", "RJ", "SP"], "SE"),
    **dict.fromkeys(["PR", "RS", "SC"], "S")
}

# Atributos de histórico do PRÓPRIO curso (anos anteriores)
HIST_COLS = [
    "taxa_evasao_t1", "taxa_evasao_t2",
    "delta_taxa", "cresc_matriculas",
    "media_historico", "std_historico_taxa",
]

# Taxa de evasão dos demais cursos da IES no ano t e atributos derivados dela.
# `diff_taxa_curso_ies` = taxa_evasao_t1 - ies_taxa_evasao_t. Como taxa_evasao_t1 também é
# atributo, o modelo conseguiria recuperar ies_taxa_evasao_t a partir da diferença; por isso
# ela pertence ao bloco IES e NÃO ao bloco de histórico.
IES_TAXA = "ies_taxa_evasao_t"
IES_DERIVADAS = ["diff_taxa_curso_ies"]


def assemble_attributes(df: pd.DataFrame, remove_constant_features: bool = True, include_year: bool = True) -> pd.DataFrame:
    """
    Monta os atributos para o modelo a partir do DataFrame de entrada (modelagem t -> t).

    Nenhum atributo usa a taxa de evasão do próprio ano t do curso (é o alvo). Entram apenas:
    perfil/fluxo do curso em t, estrutura da IES em t, taxa dos OUTROS cursos da IES em t e
    histórico do curso em t-1 e t-2 (somente se os anos forem consecutivos).
    """

    def rate(numerator, denominator):
        """Calcula a razão, tratando divisão por zero."""
        return df[numerator] / df[denominator].replace(0, np.nan)

    def has_data(*cols):
        """Verifica se todas as colunas existem no DataFrame."""
        return all(c in df.columns for c in cols)

    X = {}

    # Aplica log(1 + valor) sobre o número de matrículas e vagas
    X["log_matriculas"] = np.log1p(df["matriculas"])
    X["log_vagas"] = np.log1p(df["vagas"])

    X["concorrencia"] = rate("inscritos", "vagas")
    X["ocupacao_vagas"] = rate("ingressantes", "vagas")
    # Usa 'matriculas' em vez de 'base_evasao' para não vazar o número de evadidos do ano t no denominador
    X["ingress_sobre_matric"] = rate("ingressantes", "matriculas")
    X["concl_sobre_matric"] = rate("concluintes", "matriculas")

    X["n_locais_oferta"] = df["n_locais_oferta"]
    X["ead"] = (df["tipo_modalidade_ensino"] == 2).astype(int)

    # Perfil do curso: proporção de matriculados com determinada característica
    for col in [c for c in df.columns if c.startswith("mat_")]:
        X["p_" + col[len("mat_"):]] = rate(col, "matriculas")

    # Dados estruturais da IES
    if has_data("ies_matriculas"):
        X["ies_log_matriculas"] = np.log1p(df["ies_matriculas"])
    if has_data("ies_docentes", "ies_matriculas"):
        X["ies_docentes_100alunos"] = df["ies_docentes"] / df["ies_matriculas"].replace(0, np.nan) * 100
    if has_data("ies_tecnicos", "ies_matriculas"):
        X["ies_tecnicos_100alunos"] = df["ies_tecnicos"] / df["ies_matriculas"].replace(0, np.nan) * 100
    if has_data("ies_docentes", "ies_docentes_sem_graduacao"):
        X["ies_p_sem_graduacao"] = rate("ies_docentes_sem_graduacao", "ies_docentes")
    if has_data("ies_docentes", "ies_docentes_especializacao"):
        X["ies_p_especializacao"] = rate("ies_docentes_especializacao", "ies_docentes")
    if has_data("ies_docentes", "ies_docentes_doutores"):
        X["ies_p_doutores"] = rate("ies_docentes_doutores", "ies_docentes")
    if has_data("ies_docentes", "ies_docentes_doutores", "ies_docentes_mestres"):
        X["ies_p_mestres_doutores"] = (df["ies_docentes_doutores"] + df["ies_docentes_mestres"]) / df["ies_docentes"].replace(0, np.nan)
    if has_data("ies_docentes", "ies_docentes_de"):
        X["ies_p_dedicacao_exclusiva"] = rate("ies_docentes_de", "ies_docentes")
    if has_data("ies_periodicos"):
        X["ies_log_periodicos"] = np.log1p(df["ies_periodicos"])
    if has_data("ies_livros_eletronicos"):
        X["ies_log_livros_ele"] = np.log1p(df["ies_livros_eletronicos"])

    # Variáveis de contexto: ano e tipo de organização administrativa da IES em binário
    if include_year:
        X["ano_base"] = df["ano"]
    X["federal"] = (df["tipo_organizacao_administrativa"] == 1).astype(int)
    X["estadual"] = (df["tipo_organizacao_administrativa"] == 2).astype(int)
    X["municipal"] = (df["tipo_organizacao_administrativa"] == 3).astype(int)

    X = pd.DataFrame(X)

    # Transforma variáveis categóricas em variáveis binárias (one-hot)
    dummies = [
        pd.get_dummies(df["sigla_uf_ies"].map(REGIAO).fillna("ND"), prefix="reg", dtype=int),
        pd.get_dummies(df["id_area_geral"].fillna(-1).astype(int), prefix="area", dtype=int),
        pd.get_dummies(df["tipo_grau_academico"].fillna(-1).astype(int), prefix="grau", dtype=int),
        pd.get_dummies(df["tipo_organizacao_academica_ies"].fillna(-1).astype(int), prefix="org_acad", dtype=int),
    ]

    X = pd.concat([X] + dummies, axis=1)

    # Taxa de evasão dos OUTROS cursos da IES no ano t (desconta o próprio curso)
    if has_data("ies_evadidos", "ies_base", "evadidos", "base_evasao"):
        ies_evad_outros = (df["ies_evadidos"] - df["evadidos"]).clip(lower=0)
        ies_base_outros = (df["ies_base"] - df["base_evasao"]).replace(0, np.nan)
        X[IES_TAXA] = ies_evad_outros / ies_base_outros

    # 1) Histórico estrito de t-1 (ano anterior)
    if has_data("ano_ant", "taxa_evasao_ant"):
        ok1 = df["ano_ant"] == df["ano"] - 1
        X["taxa_evasao_t1"] = df["taxa_evasao_ant"].where(ok1)

    if has_data("ano_ant", "matriculas_ant"):
        ok1 = df["ano_ant"] == df["ano"] - 1
        mat_t1 = df["matriculas_ant"].where(ok1).replace(0, np.nan)
        X["cresc_matriculas"] = (df["matriculas"] - mat_t1) / mat_t1

    # 2) Histórico estrito de t-2 (dois anos antes) e variação passada (t-1 vs t-2)
    if has_data("ano_ant2", "taxa_evasao_ant2"):
        ok2 = df["ano_ant2"] == df["ano"] - 2
        X["taxa_evasao_t2"] = df["taxa_evasao_ant2"].where(ok2)
        if "taxa_evasao_t1" in X.columns:
            X["delta_taxa"] = X["taxa_evasao_t1"] - X["taxa_evasao_t2"]

    # Média móvel e desvio padrão apenas do PASSADO (t-1 e t-2), sem incluir o ano t
    cols_hist_taxa = [c for c in ["taxa_evasao_t1", "taxa_evasao_t2"] if c in X.columns]
    if cols_hist_taxa:
        X["media_historico"] = X[cols_hist_taxa].mean(axis=1, skipna=True)
    if len(cols_hist_taxa) >= 2:
        X["std_historico_taxa"] = X[cols_hist_taxa].std(axis=1, skipna=True)

    # Diferença entre a evasão passada do curso (t-1) e a evasão dos demais cursos da IES em t
    if IES_TAXA in X.columns and "taxa_evasao_t1" in X.columns:
        X["diff_taxa_curso_ies"] = X["taxa_evasao_t1"] - X[IES_TAXA]

    X = X.replace([np.inf, -np.inf], np.nan)

    if remove_constant_features:
        const = [c for c in X.columns
                 if X[c].nunique(dropna=True) <= 1 or (X[c].fillna(0) == 0).mean() > 0.995]
        print(f"Removendo {len(const)} colunas constantes ou quase constantes: {const}")
        X = X.drop(columns=const)

    return X.fillna(MISSING)


def attribute_group(X, include_ies=True, include_ablation=True):
    """
    Organiza os atributos em cenários para a análise de desempenho e importância.

    - Somente curso: perfil/fluxo do curso (sem IES e sem histórico);  
    - Curso + histórico: curso + evasão do próprio curso em t-1/t-2 (sem nenhuma informação da IES);
    - Curso + IES/contexto: curso + atributos da IES (inclui a taxa dos demais cursos da IES em t e
      `diff_taxa_curso_ies`), região, organização acadêmica e tipo administrativo;
    - Completo: curso + histórico + IES/contexto;
    - Ablações (include_ablation=True): os dois cenários com IES, mas SEM a taxa de evasão da IES
      no ano t e sem a diferença derivada dela. Isolam quanto do ganho vem dessa informação
      contemporânea ao alvo.
    """
    hist = [c for c in HIST_COLS if c in X.columns]
    ies_derivadas = [c for c in IES_DERIVADAS if c in X.columns]
    ies = [c for c in X.columns if c.startswith("ies_")] + ies_derivadas
    ctx = [c for c in X.columns if c.startswith(("reg_", "org_acad", "federal", "estadual", "municipal"))]
    curso = [c for c in X.columns if c not in ies + ctx + hist]

    cen = {
        "Somente curso": curso,
        "Curso + histórico": curso + hist,
    }
    if include_ies:
        cen["Curso + IES/contexto"] = curso + ies + ctx
        cen["Completo - Curso + histórico + IES/contexto"] = curso + hist + ies + ctx
        if include_ablation:
            ies_sem_taxa = [c for c in ies if c != IES_TAXA and c not in ies_derivadas]
            cen["Curso + IES/contexto (sem taxa da IES)"] = curso + ies_sem_taxa + ctx
            cen["Completo (sem taxa da IES)"] = curso + hist + ies_sem_taxa + ctx

    return cen


def cut_quartis(train_tax):
    """
    Calcula os cortes para discretizar a taxa de evasão em três classes de risco.

    Usa o 1º e o 3º quartis (25% baixo, 50% médio/interquartil, 25% alto). Os cortes devem ser
    calculados apenas com o conjunto de treino, para evitar vazamento.
    """
    q1, q3 = train_tax.quantile([0.25, 0.75])
    return [-np.inf, q1, q3, np.inf]


def for_risk(tax, cuts):
    """
    Transforma uma taxa numérica em classe de risco (0 = baixo, 1 = médio, 2 = alto).

    Args:
        tax (pd.Series): taxa de evasão (sem NaN).
        cuts (list): cortes retornados por `cut_quartis`.F

    Returns:
        pd.Series: classes de risco (int).
    """
    return pd.cut(tax, cuts, labels=[0, 1, 2], right=False).astype(int)