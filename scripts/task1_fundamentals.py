"""Task 1 - Fundamentals: Python basics, data types, descriptive stats and a few statistical ideas, all executed."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
sns.set_theme(style="whitegrid", palette="deep")
O = {}
# Python data types
O["types"] = {"int": type(42).__name__, "float": type(3.14).__name__, "str": type("a").__name__, "bool": type(True).__name__, "list": type([1]).__name__, "tuple": type((1,)).__name__, "dict": type({"a": 1}).__name__, "set": type({1}).__name__}
O["float_gotcha"] = [0.1 + 0.2, 0.1 + 0.2 == 0.3, bool(np.isclose(0.1 + 0.2, 0.3))]
# descriptive stats on a real small dataset
tips = pd.read_csv("../data/tips.csv"); x = tips.total_bill
O["bill"] = {"n": len(x), "mean": round(x.mean(), 2), "median": round(x.median(), 2), "mode": float(x.mode()[0]), "std": round(x.std(), 2), "var": round(x.var(), 1),
             "min": x.min(), "max": x.max(), "q1": round(x.quantile(.25), 2), "q3": round(x.quantile(.75), 2), "iqr": round(x.quantile(.75) - x.quantile(.25), 2),
             "skew": round(x.skew(), 2), "kurtosis": round(x.kurt(), 2)}
# Central limit theorem demo with skewed data
rng = np.random.RandomState(7)
means = [rng.choice(x.values, 30).mean() for _ in range(2000)]
O["clt"] = {"pop_mean": round(x.mean(), 2), "mean_of_means": round(np.mean(means), 2), "pop_skew": round(x.skew(), 2), "means_skew": round(float(stats.skew(means)), 2), "se_theory": round(x.std() / np.sqrt(30), 2), "se_observed": round(np.std(means), 2)}
# confidence interval + hypothesis test
ci = stats.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=stats.sem(x)); O["ci95_bill"] = [round(ci[0], 2), round(ci[1], 2)]
din = tips[tips.time == "Dinner"].total_bill; lun = tips[tips.time == "Lunch"].total_bill
t, p = stats.ttest_ind(din, lun, equal_var=False); O["ttest_dinner_lunch"] = {"dinner_mean": round(din.mean(), 2), "lunch_mean": round(lun.mean(), 2), "t": round(t, 2), "p": round(float(p), 4)}
# correlation
O["pearson"] = round(tips.total_bill.corr(tips.tip), 3); O["spearman"] = round(tips.total_bill.corr(tips.tip, method="spearman"), 3)
# probability: coin flips
flips = rng.randint(0, 2, 10000); O["coin_heads"] = round(flips.mean(), 4)
# charts
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
sns.histplot(x, bins=25, ax=ax[0]); ax[0].axvline(x.mean(), color="r", ls="--", label="mean"); ax[0].axvline(x.median(), color="g", ls="--", label="median"); ax[0].legend(); ax[0].set(title="Bill: mean is pulled right of median")
sns.histplot(means, bins=30, ax=ax[1], color="#55A868"); ax[1].set(title="Means of 2000 samples (n=30): near-normal")
sns.boxplot(x=x, ax=ax[2]); ax[2].set(title="Boxplot of total bill")
plt.tight_layout(); plt.savefig("../charts/t1_stats.png", dpi=150); plt.close()
json.dump(O, open("../charts/t1_results.json", "w"), indent=1, default=str); print(json.dumps(O, indent=1, default=str))
