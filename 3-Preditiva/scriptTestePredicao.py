"""
Teste de varios algoritmos de regressao para prever Impact_gd.
Validacao: cross-validation 10 folds repetida (mesmos folds para todos os modelos),
com ajuste de hiperparametros DENTRO de cada fold (nested CV), para nao inflar o resultado.

Como usar (Windows):
    pip install pandas numpy scikit-learn scipy
    python teste_modelos_lesoes.py

Saidas: tabela no terminal e o arquivo resultados_modelos.csv (na mesma pasta do CSV).
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor, StackingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

CSV = r"C:\Users\Flavio\Desktop\Faculdade\ativMario\Projeto Lesoes\Databases-2\database-filtrado-para-predicao.csv"
ALVO = "Impact_gd"
SEED = 42
N_FOLDS, N_REPETICOES = 20, 1   # 10 avaliacoes por modelo (~10 min). Para o resultado final use 3 (~30 min)

# ----------------------------------------------------------------------------
# 1) Leitura (o Orange exporta com TAB; sep=None detecta sozinho)
# ----------------------------------------------------------------------------
df = pd.read_csv(CSV, sep=None, engine="python", na_values=["?", "~", "", "NA"])
df.columns = [c.strip() for c in df.columns]
print(f"Linhas: {len(df)} | Colunas: {df.shape[1]}")
print("Faltantes por coluna (so as que tem):")
print(df.isna().sum()[df.isna().sum() > 0].to_string() or "  nenhum")

df = df.dropna(subset=[ALVO])          # so remove linhas sem alvo
y = df[ALVO].astype(float).values

# ----------------------------------------------------------------------------
# 2) Conjuntos de colunas
# ----------------------------------------------------------------------------
sem_alvo = [c for c in df.columns if c != ALVO]
cols_15 = [c for c in sem_alvo if c != "Age"]          # as 15 com FDR < 0,05

# Variante com "pontos": VITORIA - DERROTA por jogo (1, 0, -1) e somas
def adiciona_pontos(d):
    d = d.copy()
    antes, durante = [], []
    for i in (1, 2, 3):
        v_a = f"Match{i}_before_injury_Result=VITORIA"
        d_a = f"Match{i}_before_injury_Result=DERROTA"
        v_d = f"Match{i}_missed_match_Result=VITORIA"
        d_d = f"Match{i}_missed_match_Result=DERROTA"
        d[f"Pts_antes_{i}"] = d[v_a].fillna(0) - d[d_a].fillna(0)
        d[f"Pts_durante_{i}"] = d[v_d].fillna(0) - d[d_d].fillna(0)
        antes.append(f"Pts_antes_{i}")
        durante.append(f"Pts_durante_{i}")
    d["Pts_antes_soma"] = d[antes].sum(axis=1)
    d["Pts_durante_soma"] = d[durante].sum(axis=1)
    return d

df_p = adiciona_pontos(df)
notas = [c for c in sem_alvo if "Player_rating" in c]
cols_pontos = ["Pts_antes_soma", "Pts_durante_soma"] + notas

conjuntos = {
    "15 colunas (FDR<0,05)": (df_p, cols_15),
    "Pontos (2) + notas (3)": (df_p, cols_pontos),
    "15 colunas + Age": (df_p, sem_alvo),
}

# ----------------------------------------------------------------------------
# 3) Modelos (com busca de hiperparametros pequena, feita dentro de cada fold)
# ----------------------------------------------------------------------------
INNER = 5
def com_busca(modelo, grade):
    return GridSearchCV(modelo, grade, cv=INNER, scoring="neg_mean_squared_error", n_jobs=-1)

def pipe(modelo, escala=True):
    passos = [SimpleImputer(strategy="median")]
    if escala:
        passos.append(StandardScaler())
    passos.append(modelo)
    return make_pipeline(*passos)

modelos = {
    "Constante (media)": pipe(DummyRegressor(strategy="mean"), escala=False),
    "Regressao Linear": pipe(LinearRegression()),
    "Ridge": pipe(com_busca(Ridge(), {"alpha": [0.1, 1, 10, 100]})),
    "Lasso": pipe(com_busca(Lasso(max_iter=20000), {"alpha": [0.001, 0.01, 0.05, 0.1]})),
    "ElasticNet": pipe(com_busca(ElasticNet(max_iter=20000),
                                 {"alpha": [0.001, 0.01, 0.05], "l1_ratio": [0.2, 0.5, 0.8]})),
    "SVR linear": pipe(com_busca(SVR(kernel="linear"), {"C": [0.1, 1, 10], "epsilon": [0.1, 0.3]})),
    "SVR RBF": pipe(com_busca(SVR(kernel="rbf"),
                              {"C": [0.3, 1, 3, 10], "epsilon": [0.1, 0.3], "gamma": ["scale", 0.02]})),
    "kNN": pipe(com_busca(KNeighborsRegressor(weights="distance"),
                          {"n_neighbors": [5, 10, 15, 25]})),
    "Random Forest": pipe(com_busca(RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
                                    {"min_samples_leaf": [3, 8, 15], "max_features": [0.5, 1.0]}), escala=False),
    "Extra Trees": pipe(com_busca(ExtraTreesRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
                                  {"min_samples_leaf": [3, 8, 15], "max_features": [0.5, 1.0]}), escala=False),
    "Gradient Boosting": pipe(com_busca(GradientBoostingRegressor(random_state=SEED, subsample=0.8),
                                        {"n_estimators": [100, 250], "max_depth": [2, 3],
                                         "learning_rate": [0.03, 0.1]}), escala=False),
    "Rede Neural (MLP)": pipe(com_busca(MLPRegressor(max_iter=3000, early_stopping=True, random_state=SEED),
                                        {"hidden_layer_sizes": [(10,), (20,), (30,)],
                                         "alpha": [0.01, 0.1, 1.0]})),
}

# Stacking: linear + boosting + SVR, agregador Ridge
modelos["Stacking (Ridge+GB+SVR)"] = pipe(StackingRegressor(
    estimators=[
        ("ridge", Ridge(alpha=10)),
        ("gb", GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                         subsample=0.8, random_state=SEED)),
        ("svr", SVR(kernel="rbf", C=3, epsilon=0.1)),
    ],
    final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=-1), escala=True)

# ----------------------------------------------------------------------------
# 4) Avaliacao: mesmos folds para todos (permite comparar de forma pareada)
# ----------------------------------------------------------------------------
cv = RepeatedKFold(n_splits=N_FOLDS, n_repeats=N_REPETICOES, random_state=SEED)
scoring = {
    "R2": "r2",
    "RMSE": make_scorer(lambda a, b: np.sqrt(mean_squared_error(a, b))),
    "MAE": make_scorer(mean_absolute_error),
}

linhas = []
r2_por_modelo = {}
for nome_conj, (dados, cols) in conjuntos.items():
    X = dados[cols].astype(float).values
    print(f"\n===== Conjunto: {nome_conj}  ({len(cols)} colunas) =====")
    for nome, modelo in modelos.items():
        res = cross_validate(modelo, X, y, cv=cv, scoring=scoring, n_jobs=1)
        r2 = res["test_R2"]
        r2_por_modelo[(nome_conj, nome)] = r2
        linhas.append({
            "conjunto": nome_conj, "modelo": nome,
            "R2_medio": r2.mean(), "R2_desvio": r2.std(),
            "RMSE": res["test_RMSE"].mean(), "MAE": res["test_MAE"].mean(),
        })
        print(f"{nome:26s} R2 = {r2.mean():6.3f} (+-{r2.std():.3f})  "
              f"RMSE = {res['test_RMSE'].mean():.3f}  MAE = {res['test_MAE'].mean():.3f}")

tab = pd.DataFrame(linhas).sort_values(["conjunto", "R2_medio"], ascending=[True, False])

# ----------------------------------------------------------------------------
# 5) Comparacao pareada com a Regressao Linear (mesmos folds)
# ----------------------------------------------------------------------------
print("\n===== Diferenca de R2 em relacao a Regressao Linear (t pareado, mesmos folds) =====")
print("Obs.: folds repetidos nao sao independentes; trate p-valores como indicativos.")
for nome_conj in conjuntos:
    base = r2_por_modelo[(nome_conj, "Regressao Linear")]
    print(f"\n[{nome_conj}]")
    for nome in modelos:
        if nome in ("Regressao Linear", "Constante (media)"):
            continue
        d = r2_por_modelo[(nome_conj, nome)] - base
        t, p = stats.ttest_1samp(d, 0.0)
        print(f"{nome:26s} dif = {d.mean():+.3f}   p = {p:.3f}")

# ----------------------------------------------------------------------------
# 6) Coeficientes do Ridge (interpretacao para o relatorio)
# ----------------------------------------------------------------------------
dados, cols = conjuntos["15 colunas (FDR<0,05)"]
X = dados[cols].astype(float).values
m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10)).fit(X, y)
coef = pd.Series(m[-1].coef_, index=cols).sort_values(key=np.abs, ascending=False)
print("\n===== Coeficientes do Ridge (colunas padronizadas) =====")
print(coef.round(3).to_string())

saida = Path(CSV).with_name("resultados_modelos.csv")
try:
    tab.to_csv(saida, index=False, sep=";", decimal=",")
    print(f"\nTabela salva em: {saida}")
except Exception as e:
    tab.to_csv("resultados_modelos.csv", index=False, sep=";", decimal=",")
    print("\nTabela salva em resultados_modelos.csv (pasta atual)")