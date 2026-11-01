"""Task 5 - Business analytics: NYC taxi trips (March 2019 sample)."""
import json, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style="whitegrid", palette="deep")
O = {}
df = pd.read_csv("../data/taxis.csv", parse_dates=["pickup", "dropoff"])
O["raw_shape"] = df.shape
O["missing"] = df.isna().sum()[lambda s: s > 0].to_dict()
df = df.dropna(subset=["pickup_borough", "dropoff_borough", "payment"]).copy()
df["minutes"] = (df.dropoff - df.pickup).dt.total_seconds() / 60
df = df[(df.distance > 0) & (df.minutes > 0)]
df["hour"] = df.pickup.dt.hour
df["dow"] = df.pickup.dt.day_name()
df["tip_pct"] = np.where(df.fare > 0, df.tip / df.fare * 100, np.nan)
df["rev_per_mile"] = df.total / df.distance
df["rev_per_min"] = df.total / df.minutes
df["mph"] = df.distance / (df.minutes / 60)
df = df[df.mph < 80]
O["clean_shape"] = df.shape
O["total_revenue"] = round(df.total.sum(), 0)
O["avg_total"] = round(df.total.mean(), 2)
O["avg_distance"] = round(df.distance.mean(), 2)
O["avg_minutes"] = round(df.minutes.mean(), 1)
O["avg_fare"] = round(df.fare.mean(), 2)
O["color_share"] = (df.color.value_counts(normalize=True) * 100).round(1).to_dict()
O["rev_by_color"] = df.groupby("color").total.agg(["count", "mean", "sum"]).round(2).to_dict()
O["pay_share"] = (df.payment.value_counts(normalize=True) * 100).round(1).to_dict()
cc = df[df.payment == "credit card"]; cash = df[df.payment == "cash"]
O["tip_pct_cc_mean"] = round(cc.tip_pct.mean(), 1)
O["tip_recorded_cash_share"] = round((cash.tip > 0).mean() * 100, 1)
O["cc_zero_tip_share"] = round((cc.tip == 0).mean() * 100, 1)
O["borough_pickups"] = (df.pickup_borough.value_counts(normalize=True) * 100).round(1).to_dict()
O["borough_rev"] = df.groupby("pickup_borough").total.agg(["count", "mean", "sum"]).round(2).to_dict()
O["borough_revpermin"] = df.groupby("pickup_borough").rev_per_min.median().round(2).to_dict()
hr = df.groupby("hour").agg(trips=("total", "size"), avg_total=("total", "mean"), avg_mph=("mph", "mean"), rpm=("rev_per_min", "median"))
O["peak_hour_trips"] = int(hr.trips.idxmax()); O["peak_hour_count"] = int(hr.trips.max())
O["quiet_hour_trips"] = int(hr.trips.idxmin()); O["quiet_hour_count"] = int(hr.trips.min())
O["hour_avg_total_max"] = [int(hr.avg_total.idxmax()), round(hr.avg_total.max(), 2)]
O["hour_mph"] = hr.avg_mph.round(1).to_dict()
O["rpm_by_hour_top3"] = hr.rpm.sort_values(ascending=False).head(3).round(2).to_dict()
O["rpm_by_hour_bottom3"] = hr.rpm.sort_values().head(3).round(2).to_dict()
O["evening_share"] = round(df.hour.between(17, 21).mean() * 100, 1)
dow = df.groupby("dow").total.agg(["count", "mean"]).round(2)
O["dow_counts"] = dow["count"].to_dict()
O["pax_share"] = (df.passengers.value_counts(normalize=True) * 100).round(1).head(4).to_dict()
O["tolls_share_trips"] = round((df.tolls > 0).mean() * 100, 1)
O["toll_avg_total"] = round(df[df.tolls > 0].total.mean(), 2)
O["short_trip_share"] = round((df.distance < 2).mean() * 100, 1)
O["short_trip_rev_share"] = round(df[df.distance < 2].total.sum() / df.total.sum() * 100, 1)
O["long_trip_share"] = round((df.distance >= 8).mean() * 100, 1)
O["long_trip_rev_share"] = round(df[df.distance >= 8].total.sum() / df.total.sum() * 100, 1)
O["rpm_short_vs_long"] = [round(df[df.distance < 2].rev_per_min.median(), 2), round(df[df.distance >= 8].rev_per_min.median(), 2)]
O["rpmile_short_vs_long"] = [round(df[df.distance < 2].rev_per_mile.median(), 2), round(df[df.distance >= 8].rev_per_mile.median(), 2)]
O["corr_dist_total"] = round(df.distance.corr(df.total), 3)
top_routes = df.groupby(["pickup_zone", "dropoff_zone"]).total.agg(["count", "sum"]).sort_values("count", ascending=False).head(5).round(1)
O["top_routes"] = {f"{a} -> {b}": [int(r["count"]), r["sum"]] for (a, b), r in top_routes.iterrows()}
O["top_pickup_zones"] = df.pickup_zone.value_counts().head(5).to_dict()

# charts
fig, ax = plt.subplots(figsize=(8, 3.6)); ax.bar(hr.index, hr.trips, color="#4C72B0"); ax.set(title="Trips by hour of day", xlabel="Pickup hour (24h)", ylabel="Trips"); plt.tight_layout(); plt.savefig("../charts/t5_hourly.png", dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(8, 3.6)); ax.plot(hr.index, hr.avg_mph, marker="o"); ax.set(title="Average trip speed by hour", xlabel="Pickup hour (24h)", ylabel="Miles per hour"); plt.tight_layout(); plt.savefig("../charts/t5_speed.png", dpi=150); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
b = df.groupby("pickup_borough").total.mean().sort_values(); ax[0].barh(b.index, b.values, color="#55A868"); ax[0].set(title="Average total per trip ($) by pickup borough")
p = df.payment.value_counts(); ax[1].bar(p.index, p.values, color=["#4C72B0", "#C44E52"]); ax[1].set(title="Trips by payment type")
plt.tight_layout(); plt.savefig("../charts/t5_borough_payment.png", dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(7, 4)); s = df.sample(2500, random_state=1); ax.scatter(s.distance, s.total, s=6, alpha=.4); ax.set(title="Trip distance vs total fare", xlabel="Distance (miles)", ylabel="Total ($)", xlim=(0, 25), ylim=(0, 90)); plt.tight_layout(); plt.savefig("../charts/t5_scatter.png", dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(8, 3.6)); ax.plot(hr.index, hr.rpm, marker="o", color="#C44E52"); ax.set(title="Median revenue per minute by hour ($)", xlabel="Pickup hour (24h)", ylabel="$ per minute"); plt.tight_layout(); plt.savefig("../charts/t5_rev_per_min.png", dpi=150); plt.close()
json.dump(O, open("../charts/t5_results.json", "w"), indent=1, default=str); print(json.dumps(O, indent=1, default=str))
