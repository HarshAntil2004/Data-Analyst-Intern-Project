"""Task 2 - Hands-on lab: data manipulation, visualisation and basic analysis on the tips dataset."""
import json, sys, platform, pandas as pd, numpy as np, matplotlib, seaborn, scipy, sklearn
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style="whitegrid", palette="deep")
O = {"env": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__,
             "seaborn": seaborn.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__}}
df = pd.read_csv("../data/tips.csv")
O["shape"] = df.shape; O["dtypes"] = df.dtypes.astype(str).to_dict()
O["missing"] = int(df.isna().sum().sum()); O["dups"] = int(df.duplicated().sum())
O["describe"] = df[["total_bill", "tip", "size"]].describe().round(2).to_dict()
df["tip_pct"] = (df.tip / df.total_bill * 100).round(2)
df["bill_per_person"] = (df.total_bill / df["size"]).round(2)
# manipulation exercises
O["filter_big_bills"] = int((df.total_bill > 30).sum())
O["tip_pct_mean"] = round(df.tip_pct.mean(), 2); O["tip_pct_median"] = round(df.tip_pct.median(), 2)
O["by_day"] = df.groupby("day", observed=True).agg(n=("tip", "size"), avg_bill=("total_bill", "mean"), avg_tip=("tip", "mean"), tip_pct=("tip_pct", "mean")).round(2).to_dict("index")
O["by_time"] = df.groupby("time", observed=True).agg(n=("tip", "size"), avg_bill=("total_bill", "mean"), tip_pct=("tip_pct", "mean")).round(2).to_dict("index")
O["by_smoker"] = df.groupby("smoker", observed=True).tip_pct.mean().round(2).to_dict()
O["by_sex"] = df.groupby("sex", observed=True).tip_pct.mean().round(2).to_dict()
O["by_size"] = df.groupby("size").agg(n=("tip", "size"), tip_pct=("tip_pct", "mean")).round(2).to_dict("index")
O["pivot_day_time"] = df.pivot_table(index="day", columns="time", values="total_bill", aggfunc="mean", observed=True).round(2).fillna(0).to_dict()
O["corr_bill_tip"] = round(df.total_bill.corr(df.tip), 3)
O["corr_bill_tip_pct"] = round(df.total_bill.corr(df.tip_pct), 3)
O["top_tip_pct"] = df.nlargest(3, "tip_pct")[["total_bill", "tip", "size", "tip_pct"]].to_dict("records")
# simple regression by hand with numpy
b, a = np.polyfit(df.total_bill, df.tip, 1)
pred = a + b * df.total_bill; r2 = 1 - ((df.tip - pred) ** 2).sum() / ((df.tip - df.tip.mean()) ** 2).sum()
O["fit"] = {"slope": round(b, 4), "intercept": round(a, 3), "r2": round(r2, 3)}
# charts
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
sns.histplot(df.total_bill, bins=25, kde=True, ax=ax[0]); ax[0].set(title="Total bill distribution")
sns.scatterplot(data=df, x="total_bill", y="tip", hue="time", ax=ax[1]); ax[1].plot(df.total_bill.sort_values(), a + b * df.total_bill.sort_values(), "k--", lw=1); ax[1].set(title="Bill vs tip with fitted line")
sns.boxplot(data=df, x="day", y="tip_pct", ax=ax[2]); ax[2].set(title="Tip % by day", ylabel="Tip %")
plt.tight_layout(); plt.savefig("../charts/t2_lab.png", dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(6.5, 3.6)); sns.barplot(data=df, x="size", y="tip_pct", errorbar=None, ax=ax); ax.set(title="Average tip % by party size", ylabel="Tip %", xlabel="Party size"); plt.tight_layout(); plt.savefig("../charts/t2_size.png", dpi=150); plt.close()
json.dump(O, open("../charts/t2_results.json", "w"), indent=1, default=str); print(json.dumps(O, indent=1, default=str))
