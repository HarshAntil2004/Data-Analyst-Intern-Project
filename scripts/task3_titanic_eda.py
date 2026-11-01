"""Task 3 - Real-world dataset analysis: Titanic passenger data."""
import json, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
sns.set_theme(style="whitegrid", palette="deep")
OUT = {}
df = pd.read_csv("../data/titanic.csv")
OUT["shape_raw"] = df.shape
OUT["dtypes"] = df.dtypes.astype(str).to_dict()
OUT["missing"] = df.isna().sum()[lambda s: s > 0].to_dict()
OUT["missing_pct"] = (df.isna().mean() * 100).round(1)[lambda s: s > 0].to_dict()
OUT["duplicates"] = int(df.duplicated().sum())
OUT["overall_survival"] = round(df.survived.mean() * 100, 1)

# ---- cleaning ----
clean = df.drop_duplicates().copy()
clean["age_missing"] = clean.age.isna()
clean["age"] = clean.groupby(["pclass", "sex"])["age"].transform(lambda s: s.fillna(s.median()))
clean["embarked"] = clean.embarked.fillna(clean.embarked.mode()[0])
clean["embark_town"] = clean.embark_town.fillna(clean.embark_town.mode()[0])
clean["has_cabin_info"] = clean.deck.notna()
clean = clean.drop(columns=["deck", "alive", "class", "who", "adult_male"])
for c in ["sex", "embarked", "embark_town"]:
    clean[c] = clean[c].astype("category")
clean["survived"] = clean.survived.astype("int8")
clean["family_size"] = clean.sibsp + clean.parch + 1
clean["age_group"] = pd.cut(clean.age, [0, 12, 18, 35, 60, 100], labels=["Child", "Teen", "Young adult", "Adult", "Senior"])
OUT["shape_clean"] = clean.shape
OUT["missing_after"] = int(clean.isna().sum().sum())

# outliers (IQR) on fare and age
def iqr(s):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    lo, hi = q1 - 1.5 * i, q3 + 1.5 * i
    return int(((s < lo) | (s > hi)).sum()), round(hi, 2)
OUT["fare_outliers"], OUT["fare_upper"] = iqr(clean.fare)
OUT["age_outliers"], OUT["age_upper"] = iqr(clean.age)
OUT["fare_skew"] = round(clean.fare.skew(), 2)
clean["log_fare"] = np.log1p(clean.fare)
OUT["log_fare_skew"] = round(clean.log_fare.skew(), 2)

# ---- EDA numbers ----
OUT["surv_by_sex"] = (clean.groupby("sex", observed=True).survived.mean() * 100).round(1).to_dict()
OUT["surv_by_class"] = (clean.groupby("pclass").survived.mean() * 100).round(1).to_dict()
OUT["surv_by_agegroup"] = (clean.groupby("age_group", observed=True).survived.mean() * 100).round(1).to_dict()
OUT["surv_by_port"] = (clean.groupby("embark_town", observed=True).survived.mean() * 100).round(1).to_dict()
OUT["surv_by_family"] = (clean.groupby(pd.cut(clean.family_size, [0, 1, 4, 20], labels=["Alone", "2-4", "5+"]), observed=True).survived.mean() * 100).round(1).to_dict()
OUT["sex_class"] = (clean.pivot_table(index="pclass", columns="sex", values="survived", aggfunc="mean", observed=True) * 100).round(1).to_dict()
OUT["fare_by_class_median"] = clean.groupby("pclass").fare.median().round(2).to_dict()
OUT["age_mean"] = round(clean.age.mean(), 1)
OUT["corr_surv"] = clean[["survived", "pclass", "age", "fare", "sibsp", "parch", "family_size"]].corr()["survived"].round(3).to_dict()
ct = pd.crosstab(clean.sex, clean.survived)
chi2, p, _, _ = stats.chi2_contingency(ct)
OUT["chi2_sex"] = (round(chi2, 1), float(p))
t, p2 = stats.ttest_ind(clean[clean.survived == 1].fare, clean[clean.survived == 0].fare, equal_var=False)
OUT["ttest_fare"] = (round(t, 2), float(p2))

# ---- charts ----
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
sns.barplot(data=clean, x="sex", y="survived", ax=ax[0], errorbar=None); ax[0].set(title="Survival rate by sex", ylabel="Survival rate", xlabel="")
sns.barplot(data=clean, x="pclass", y="survived", hue="sex", ax=ax[1], errorbar=None); ax[1].set(title="Survival rate by class and sex", ylabel="Survival rate", xlabel="Passenger class")
plt.tight_layout(); plt.savefig("../charts/t3_survival_sex_class.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
sns.histplot(clean.age, bins=30, kde=True, ax=ax[0]); ax[0].set(title="Age distribution (after imputation)", xlabel="Age")
sns.histplot(clean.fare, bins=40, ax=ax[1]); ax[1].set(title="Fare distribution (right-skewed)", xlabel="Fare")
plt.tight_layout(); plt.savefig("../charts/t3_distributions.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
sns.boxplot(data=clean, x="pclass", y="fare", ax=ax[0]); ax[0].set(title="Fare by class (outliers visible)", xlabel="Passenger class")
sns.boxplot(data=clean, x="survived", y="age", ax=ax[1]); ax[1].set(title="Age by survival outcome", xlabel="Survived (0 = no, 1 = yes)")
plt.tight_layout(); plt.savefig("../charts/t3_boxplots.png", dpi=150); plt.close()

plt.figure(figsize=(6, 4.6))
sns.heatmap(clean[["survived", "pclass", "age", "fare", "sibsp", "parch", "family_size"]].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation matrix"); plt.tight_layout(); plt.savefig("../charts/t3_corr.png", dpi=150); plt.close()

plt.figure(figsize=(7, 3.8))
sns.barplot(data=clean, x="age_group", y="survived", errorbar=None); plt.title("Survival rate by age group"); plt.ylabel("Survival rate"); plt.xlabel("")
plt.tight_layout(); plt.savefig("../charts/t3_agegroup.png", dpi=150); plt.close()

clean.to_csv("../data/titanic_clean.csv", index=False)
json.dump(OUT, open("../charts/t3_results.json", "w"), indent=1, default=str)
print(json.dumps(OUT, indent=1, default=str))
