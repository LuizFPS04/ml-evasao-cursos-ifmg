from src.database import get_duckdb_connection

# Escopo: IES públicas federais (1) e estaduais (2)
CATEGORIAS_ADM = (1, 2)


def cleaning_data(min_base: int = 80, only_last_year: bool = False):
    """
    Limpa e agrega os dados por CURSO x IES x ANO (desenho t -> t).

    - Atributos do ano t (perfil do curso e da IES) e histórico (t-1, t-2) do próprio curso.
    - Alvo: taxa de evasão do MESMO ano t (`taxa_evasao_alvo`), discretizada depois em risco
      baixo/médio/alto com cortes (quartis) calculados só no treino de cada fold.

    Definição da taxa (INEP: matrículas = alunos Cursando e/ou Formado):
        taxa_evasao = (trancados + desvinculados) / (matrículas + trancados + desvinculados)

    Args:
        min_base: mínimo de (matrículas + evadidos) no ano t para a taxa ser estável.
        only_last_year: False (padrão) = todos os anos da base; True = só o ÚLTIMO ano
            (útil para um teste temporal: treinar nos anos anteriores e testar no último).

    Returns:
        pandas.DataFrame: uma linha por curso x IES x ano, com o alvo `taxa_evasao_alvo`.
    """
    adm = ", ".join(str(c) for c in CATEGORIAS_ADM)
    filtro = f"""WHERE base_evasao >= {int(min_base)}
          AND taxa_evasao IS NOT NULL"""
    if only_last_year:
        filtro += "\n          AND ano = (SELECT MAX(ano) FROM alvo)"

    query = f"""
        WITH curso_ano AS (
            -- 1) Agrega polos EAD e a linha 'tipo_dimensao = 3' em uma linha por curso x IES x ano.
            --    Contagens do CURSO: SUM.  Colunas do nível IES (repetidas em toda linha): MAX.
            SELECT
                ano,
                id_ies,
                id_curso,

                -- Identificação (não entram como atributos do modelo)
                MAX(nome_curso)                                  AS nome_curso,

                -- Variáveis do CURSO (categóricas / contexto)
                MAX(id_area_geral)                               AS id_area_geral,
                MAX(tipo_grau_academico)                         AS tipo_grau_academico,
                MAX(tipo_modalidade_ensino)                      AS tipo_modalidade_ensino,   -- 2 = tem oferta EAD
                COUNT(DISTINCT id_municipio)                     AS n_locais_oferta,

                -- Variáveis do CURSO (fluxo)
                SUM(quantidade_matriculas)                       AS matriculas,
                SUM(quantidade_ingressantes)                     AS ingressantes,
                SUM(quantidade_concluintes)                      AS concluintes,
                SUM(quantidade_vagas)                            AS vagas,
                SUM(quantidade_inscritos)                        AS inscritos,
                SUM(quantidade_alunos_situacao_trancada)         AS trancados,
                SUM(quantidade_alunos_situacao_desvinculada)     AS desvinculados,

                -- Variáveis do CURSO (perfil dos matriculados)
                SUM(quantidade_matriculas_feminino)              AS mat_feminino,
                SUM(quantidade_matriculas_noturno)               AS mat_noturno,
                SUM(quantidade_matriculas_0_17 + quantidade_matriculas_18_24)  AS mat_idade_0_24,
                SUM(quantidade_matriculas_35_39 + quantidade_matriculas_40_49) AS mat_idade_35_49,
                SUM(quantidade_matriculas_50_59 + quantidade_matriculas_60_mais) AS mat_idade_50_mais,
                SUM(quantidade_matriculas_branca)               AS mat_branca,
                SUM(quantidade_matriculas_amarela)              AS mat_amarela,
                SUM(quantidade_matriculas_indigena)             AS mat_indigena,
                SUM(quantidade_matriculas_preta + quantidade_matriculas_parda)     AS mat_preta_parda,
                SUM(quantidade_matriculas_deficiencia)           AS mat_deficiencia,
                SUM(quantidade_matriculas_reserva_vaga)          AS mat_reserva_vaga,
                SUM(quantidade_matriculas_reserva_vaga_etnico) AS mat_reserva_etnico,
                SUM(quantidade_matriculas_reserva_vaga_rede_publica) AS mat_reserva_rede_publica,
                SUM(quantidade_matriculas_reserva_vaga_social_renda_familiar) AS mat_reserva_renda,
                SUM(quantidade_matriculas_apoio_social)          AS mat_apoio_social,
                SUM(quantidade_matriculas_atividade_extracurricular) AS mat_extracurricular,
                SUM(quantidade_matriculas_mobilidade_academica)  AS mat_mobilidade,
                SUM(quantidade_matriculas_financiamento)       AS mat_financiamento,

                -- Componentes do alvo (NUNCA usar como atributo preditor, pois são o próprio alvo)
                SUM(quantidade_alunos_situacao_trancada + quantidade_alunos_situacao_desvinculada)   AS evadidos,

                -- Variáveis da IES (uma linha por IES/ano -> MAX, não SUM)
                MAX(sigla_uf_1)                                  AS sigla_uf_ies,
                MAX(tipo_organizacao_administrativa)             AS tipo_organizacao_administrativa,
                MAX(tipo_organizacao_academica_1)                AS tipo_organizacao_academica_ies,
                MAX(quantidade_docentes_exercicio)               AS ies_docentes,
                MAX(quantidade_docentes_exercicio_sem_graduacao) AS ies_docentes_sem_graduacao,
                MAX(quantidade_docentes_exercicio_especializacao) AS ies_docentes_especializacao,
                MAX(quantidade_docentes_exercicio_doutorado)     AS ies_docentes_doutores,
                MAX(quantidade_docentes_exercicio_mestrado)      AS ies_docentes_mestres,
                MAX(quantidade_docentes_exercicio_integral_dedicacao_exclusiva) AS ies_docentes_de,
                MAX(quantidade_docentes_exercicio_horista)       AS ies_docentes_horistas,
                MAX(quantidade_docentes_exercicio_branca)          AS ies_docentes_branca,
                MAX(quantidade_docentes_exercicio_amarela)         AS ies_docentes_amarela,
                MAX(quantidade_docentes_exercicio_indigena)        AS ies_docentes_indigena,
                MAX(quantidade_docentes_exercicio_preta + quantidade_docentes_exercicio_parda) AS ies_docentes_preta_parda,
                MAX(quantidade_tecnicos)                         AS ies_tecnicos,
                MAX(quantidade_biblioteca_periodicos_eletronicos) AS ies_periodicos,
                MAX(quantidade_biblioteca_livros_eletronicos)    AS ies_livros_eletronicos
            FROM censo_merge_ies_course_ifmg
            WHERE tipo_organizacao_administrativa IN ({adm})
            GROUP BY ano, id_ies, id_curso
        ),

        taxa AS (
            -- 2) Taxa de evasão do ano e total de matrículas da IES (p/ razões por aluno)
            SELECT
                *,
                matriculas + evadidos                                          AS base_evasao,
                CAST(evadidos AS DOUBLE) / NULLIF(matriculas + evadidos, 0)    AS taxa_evasao,
                SUM(matriculas) OVER (PARTITION BY ano, id_ies)                AS ies_matriculas,
                SUM(evadidos) OVER (PARTITION BY ano, id_ies)                  AS ies_evadidos,
                SUM(matriculas + evadidos) OVER (PARTITION BY ano, id_ies)     AS ies_base
            FROM curso_ano
        ),

        alvo AS (
            -- 3) Alvo (mesmo ano t) e histórico do curso (LAG sobre a base completa, antes do filtro)
            SELECT
                *,
                taxa_evasao        AS taxa_evasao_alvo,
                base_evasao        AS base_evasao_alvo,
                LAG(ano)           OVER w AS ano_ant,
                LAG(taxa_evasao)   OVER w AS taxa_evasao_ant,
                LAG(matriculas)    OVER w AS matriculas_ant,
                LAG(ano, 2)        OVER w AS ano_ant2,
                LAG(taxa_evasao, 2) OVER w AS taxa_evasao_ant2
            FROM taxa
            WINDOW w AS (PARTITION BY id_ies, id_curso ORDER BY ano)
        )

        SELECT *
        FROM alvo
        {filtro}
        ORDER BY id_ies, id_curso, ano
    """

    conn = get_duckdb_connection()
    try:
        return conn.execute(query).df()
    finally:
        conn.close()