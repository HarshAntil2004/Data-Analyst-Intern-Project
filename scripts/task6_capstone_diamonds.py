"""Task 6 - Capstone: predicting diamond prices (regression)."""
import json, time, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
sns.set_theme(style="whitegrid", palette="deep")
O = {}
df = pd.read_csv("../data/diamonds.csv")
O["raw_shape"] = df.shape
O["missing"] = int(df.isna().sum().sum())
O["dups"] = int(df.duplicated().sum())
O["zero_dims"] = int(((df.x == 0) | (df.y == 0) | (df.z == 0)).sum())
O["big_dims"] = int(((df.y > 20) | (df.z > 20)).sum())
O["price_desc"] = df.price.describe().round(1).to_dict()
O["price_skew"] = round(df.price.skew(), 2)
df = df.drop_duplicates()
df = df[(df.x > 0) & (df.y > 0) & (df.z > 0) & (df.y < 20) & (df.z < 20)]
O["clean_shape"] = df.shape
O["corr_price"] = df.select_dtypes("number").corr()["price"].round(3).to_dict()
O["price_by_cut"] = df.groupby("cut").price.median().round(0).to_dict()
O["price_by_color"] = df.groupby("color").price.median().round(0).to_dict()
O["price_by_clarity"] = df.groupby("clarity").price.median().round(0).to_dict()
O["carat_median"] = float(df.carat.median())

cut_o = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
col_o = ["J", "I", "H", "G", "F", "E", "D"]
cla_o = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
df["volume"] = df.x * df.y * df.z
df["depth_calc"] = 2 * df.z / (df.x + df.y) * 100
y = np.log(df.price)
X = df.drop(columns="price")
num = ["carat", "depth", "table", "x", "y", "z", "volume", "depth_calc"]
pre = ColumnTransformer([
    ("ord", OrdinalEncoder(categories=[cut_o, col_o, cla_o]), ["cut", "color", "clarity"]),
    ("num", StandardScaler(), num)])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
O["split"] = [len(Xtr), len(Xte)]
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=120, max_depth=18, n_jobs=-1, random_state=42),
    "Gradient Boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=42),
}
# baseline
base = np.full(len(yte), ytr.mean())
O["baseline"] = {"MAE": round(mean_absolute_error(np.exp(yte), np.exp(base)), 0), "R2_log": round(r2_score(yte, base), 3)}
res = {}; preds = {}
for n, m in models.items():
    p = Pipeline([("pre", pre), ("m", m)])
    t0 = time.time(); p.fit(Xtr, ytr); sec = time.time() - t0
    pr = p.predict(Xte); preds[n] = (p, pr)
    pe, ae = np.exp(pr), np.exp(yte)
    res[n] = {"R2": round(r2_score(yte, pr), 4), "MAE": round(mean_absolute_error(ae, pe), 0),
              "RMSE": round(float(np.sqrt(mean_squared_error(ae, pe))), 0),
              "MAPE": round(float(np.mean(np.abs(pe - ae) / ae) * 100), 1), "fit_sec": round(sec, 1)}
O["results"] = res
best = max(res, key=lambda k: res[k]["R2"]); O["best"] = best
cvp = Pipeline([("pre", pre), ("m", models[best])])
cv = cross_val_score(cvp, Xtr, ytr, cv=5, scoring="r2", n_jobs=1)
O["cv_best"] = [round(cv.mean(), 4), round(cv.std(), 4)]
p, pr = preds[best]
pe, ae = np.exp(pr), np.exp(yte)
err = (pe - ae)
O["err_pct_median"] = round(float(np.median(np.abs(err) / ae) * 100), 1)
bins = pd.cut(ae.values, [0, 1000, 2500, 5000, 10000, 20000])
e = pd.DataFrame({"bin": bins, "ape": np.abs(err.values) / ae.values * 100, "ae": np.abs(err.values)})
g = e.groupby("bin", observed=True).agg(mape=("ape", "mean"), mae=("ae", "mean")).round(1)
g.index = g.index.astype(str)
O["err_by_price_band"] = g.to_dict()
O["under_share"] = round(float((err < 0).mean() * 100), 1)
# feature importance via permutation on a sample
from sklearn.inspection import permutation_importance
idx = np.random.RandomState(0).choice(len(Xte), 4000, replace=False)
pi = permutation_importance(p, Xte.iloc[idx], yte.iloc[idx], n_repeats=5, random_state=0, scoring="r2")
names = ["cut", "color", "clarity", "carat", "depth", "table", "x", "y", "z", "volume", "depth_calc"]
cols = list(Xte.columns)
imp = pd.Series(pi.importances_mean, index=cols).sort_values(ascending=False)
O["importance"] = imp.round(4).head(6).to_dict()

# charts
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
sns.histplot(df.price, bins=50, ax=ax[0]); ax[0].set(title="Price (right-skewed)", xlabel="Price ($)")
sns.histplot(np.log(df.price), bins=50, ax=ax[1], color="#55A868"); ax[1].set(title="log(Price) is much closer to normal", xlabel="log(price)")
plt.tight_layout(); plt.savefig("../charts/t6_price_dist.png", dpi=150); plt.close()
s = df.sample(4000, random_state=2)
fig, ax = plt.subplots(figsize=(7, 4.2)); sns.scatterplot(data=s, x="carat", y="price", hue="clarity", s=10, alpha=.6, ax=ax, hue_order=cla_o); ax.set(title="Price vs carat, coloured by clarity"); plt.tight_layout(); plt.savefig("../charts/t6_carat_price.png", dpi=150); plt.close()
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
for a, c, o in zip(ax, ["cut", "color", "clarity"], [cut_o, col_o, cla_o]):
    sns.boxplot(data=df, x=c, y="price", order=o, ax=a, showfliers=False); a.set(title=f"Price by {c}", xlabel="")
    a.tick_params(axis="x", rotation=40)
plt.tight_layout(); plt.savefig("../charts/t6_cat_price.png", dpi=150); plt.close()
plt.figure(figsize=(6, 4.6)); sns.heatmap(df[["price", "carat", "depth", "table", "x", "y", "z"]].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0); plt.title("Correlation matrix"); plt.tight_layout(); plt.savefig("../charts/t6_corr.png", dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(6, 4.6)); ax.scatter(ae[:3000], pe[:3000], s=6, alpha=.4); lim = [0, 18000]; ax.plot(lim, lim, "r--"); ax.set(title=f"{best}: actual vs predicted", xlabel="Actual ($)", ylabel="Predicted ($)", xlim=lim, ylim=lim); plt.tight_layout(); plt.savefig("../charts/t6_actual_pred.png", dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(7, 3.8)); imp.head(8).sort_values().plot.barh(ax=ax, color="#4C72B0"); ax.set(title="Permutation importance (drop in R²)"); plt.tight_layout(); plt.savefig("../charts/t6_importance.png", dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(7, 3.6)); ax.bar(res.keys(), [v["MAE"] for v in res.values()], color="#C44E52"); ax.set(title="Test MAE by model ($)"); ax.tick_params(axis="x", rotation=15); plt.tight_layout(); plt.savefig("../charts/t6_model_mae.png", dpi=150); plt.close()
json.dump(O, open("../charts/t6_results.json", "w"), indent=1, default=str); print(json.dumps(O, indent=1, default=str))
