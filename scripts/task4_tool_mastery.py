"""Task 4 - Tool mastery: NumPy vs loops, pandas idioms, SQL via sqlite, scikit-learn pipelines, plotting."""
import json, time, sqlite3, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
sns.set_theme(style="whitegrid", palette="deep")
O = {}
def timeit(f, n=3):
    best = 1e9
    for _ in range(n):
        t = time.perf_counter(); f(); best = min(best, time.perf_counter() - t)
    return best
# 1. NumPy vectorisation
N = 1_000_000
a = np.random.RandomState(0).rand(N); lst = a.tolist()
loop = timeit(lambda: [x * 2 + 1 for x in lst]); vec = timeit(lambda: a * 2 + 1)
O["numpy_loop_s"], O["numpy_vec_s"] = round(loop, 4), round(vec, 5); O["numpy_speedup"] = round(loop / vec, 1)
loop2 = timeit(lambda: sum(x * x for x in lst), 2); vec2 = timeit(lambda: np.dot(a, a)); O["dot_speedup"] = round(loop2 / vec2, 1)
# 2. pandas: iterrows vs apply vs vectorised
d = pd.read_csv("../data/diamonds.csv").head(20000)
def r_iter():
    out = []
    for _, r in d.iterrows(): out.append(r.price / r.carat)
    return out
O["pd_iterrows_s"] = round(timeit(r_iter, 1), 3)
O["pd_apply_s"] = round(timeit(lambda: d.apply(lambda r: r.price / r.carat, axis=1), 1), 3)
O["pd_vector_s"] = round(timeit(lambda: d.price / d.carat), 5)
O["pd_speedup_iter_vs_vec"] = round(O["pd_iterrows_s"] / O["pd_vector_s"], 0)
# 3. SQL with sqlite on same data (pandas vs SQL give same answer)
con = sqlite3.connect(":memory:"); d.to_sql("diamonds", con, index=False)
q = "SELECT cut, COUNT(*) n, ROUND(AVG(price),1) avg_price FROM diamonds GROUP BY cut ORDER BY avg_price DESC"
sql = pd.read_sql(q, con); pdres = d.groupby("cut").price.agg(n="size", avg_price="mean").round(1).sort_values("avg_price", ascending=False).reset_index()
O["sql_matches_pandas"] = bool(np.allclose(sql.avg_price.values, pdres.avg_price.values) and (sql.n.values == pdres.n.values).all())
O["sql_result"] = sql.to_dict("records")
# 4. scikit-learn pipeline on Titanic
t = pd.read_csv("../data/titanic.csv")
feats_num = ["age", "fare", "sibsp", "parch", "pclass"]; feats_cat = ["sex", "embarked"]
X = t[feats_num + feats_cat]; y = t.survived
pre = ColumnTransformer([("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), feats_num),
                         ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("o", OneHotEncoder(handle_unknown="ignore"))]), feats_cat)])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
res = {}
for n, m in {"Logistic Regression": LogisticRegression(max_iter=1000), "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42)}.items():
    p = Pipeline([("pre", pre), ("m", m)]); cv = cross_val_score(p, Xtr, ytr, cv=5, scoring="accuracy")
    p.fit(Xtr, ytr); pr = p.predict(Xte)
    res[n] = {"cv_acc": round(cv.mean(), 3), "test_acc": round(accuracy_score(yte, pr), 3), "test_f1": round(f1_score(yte, pr), 3)}
    if n == "Random Forest": O["cm_rf"] = confusion_matrix(yte, pr).tolist()
O["sk"] = res; O["majority_baseline"] = round(max(yte.mean(), 1 - yte.mean()), 3)
# chart
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].bar(["Python loop", "NumPy"], [loop, vec], color=["#C44E52", "#55A868"]); ax[0].set(title="1M element arithmetic (seconds)", yscale="log")
ax[1].bar(["iterrows", "apply", "vectorised"], [O["pd_iterrows_s"], O["pd_apply_s"], O["pd_vector_s"]], color=["#C44E52", "#DD8452", "#55A868"]); ax[1].set(title="pandas: price per carat, 20k rows (s)", yscale="log")
plt.tight_layout(); plt.savefig("../charts/t4_benchmarks.png", dpi=150); plt.close()
json.dump(O, open("../charts/t4_results.json", "w"), indent=1, default=str); print(json.dumps(O, indent=1, default=str))
