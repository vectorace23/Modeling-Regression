"""
Used car prices: should the dealer buy this car?

Reproduces every number and the chart for the slides.
Run:  python analysis.py
Needs: pandas, numpy, scipy, scikit-learn, matplotlib
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

SEED = 42            # fixes the random 70/30 split so results reproduce
PREP_COST = 1500     # inspection, reconditioning, detailing, lot time
ALL_FEATURES = ["Mileage_K", "Age_Years", "Horsepower", "Accident_Yes", "Num_Owners"]


# ---------------------------------------------------------------- helpers
def prep(df):
    df = df.copy()
    df["Accident_Yes"] = (df["Accident"].str.strip().str.lower() == "yes").astype(int)
    return df


def ols(df, features, y="Sale_Price"):
    """Ordinary least squares with coefficients, std errors, p-values, R², adj R²."""
    X = np.column_stack([np.ones(len(df))] + [df[f].to_numpy(float) for f in features])
    Y = df[y].to_numpy(float)
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    sse = resid @ resid
    sst = ((Y - Y.mean()) ** 2).sum()
    sigma2 = sse / (n - k)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), n - k)
    r2 = 1 - sse / sst
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k)
    table = pd.DataFrame(
        {"coef": beta, "std_err": se, "t": t, "p_value": p},
        index=["Intercept"] + features,
    )
    return {"features": features, "beta": beta, "table": table,
            "r2": r2, "adj_r2": adj_r2, "n": n, "resid_se": np.sqrt(sigma2)}


def predict(model, df):
    X = np.column_stack([np.ones(len(df))] + [df[f].to_numpy(float) for f in model["features"]])
    return X @ model["beta"]


def r2_score(y, yhat):
    return 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def rmse(y, yhat):
    return float(np.sqrt(((y - yhat) ** 2).mean()))


# ---------------------------------------------------------------- 1. data + split
sales = prep(pd.read_csv(os.path.join(DATA, "used_car_sales.csv")))
lots = prep(pd.read_csv(os.path.join(DATA, "auction_lots.csv")))

train, test = train_test_split(sales, test_size=0.30, random_state=SEED)
print(f"Rows: {len(sales)} total -> {len(train)} train / {len(test)} test (seed={SEED})")

# ---------------------------------------------------------------- 2. fit
full = ols(train, ALL_FEATURES)
print("\n=== Full model (all 5 features), training set ===")
print(full["table"].round(4).to_string())
print(f"R² = {full['r2']:.4f}   adj R² = {full['adj_r2']:.4f}   residual SE = ${full['resid_se']:,.0f}")

# Candidate reduced model: drop any feature that is not significant at 5%
weak = [f for f in ALL_FEATURES if full["table"].loc[f, "p_value"] > 0.05]
if weak:
    reduced = ols(train, [f for f in ALL_FEATURES if f not in weak])
    print(f"\n=== Reduced model (dropped {weak}), training set ===")
    print(reduced["table"].round(4).to_string())
    print(f"R² = {reduced['r2']:.4f}   adj R² = {reduced['adj_r2']:.4f}   residual SE = ${reduced['resid_se']:,.0f}")
    # Hand over the simpler model unless the full one has clearly higher adj R²
    v1 = reduced if reduced["adj_r2"] >= full["adj_r2"] - 0.002 else full
else:
    reduced = None
    v1 = full
print(f"\nChosen v1 features: {v1['features']}  (training adj R² = {v1['adj_r2']:.4f})")

# Per-unit effects in dealer terms
print("\nv1 coefficients in dollars:")
for f, b in zip(v1["features"], v1["beta"][1:]):
    print(f"  {f:13s} {b:+,.0f}")

# ---------------------------------------------------------------- 3. test-set fit
y_test = test["Sale_Price"].to_numpy(float)
baseline_value = train["Sale_Price"].mean()
pred_base = np.full(len(test), baseline_value)
pred_v1 = predict(v1, test)

fit_tbl = pd.DataFrame({
    "R2_test": [r2_score(y_test, pred_base), r2_score(y_test, pred_v1)],
    "RMSE_test_$": [rmse(y_test, pred_base), rmse(y_test, pred_v1)],
}, index=["Baseline (train avg)", "v1 regression"])
print(f"\nBaseline prediction for every car = training average = ${baseline_value:,.0f}")
print("\n=== Fit on the 72 held-out test cars ===")
print(fit_tbl.round(3).to_string())
fit_tbl.to_csv(os.path.join(OUT, "test_fit.csv"))

# Chart: predicted vs actual on test cars
fig, ax = plt.subplots(figsize=(6.5, 5.5))
lo, hi = 0, max(y_test.max(), pred_v1.max()) * 1.05
ax.plot([lo, hi], [lo, hi], color="#999999", lw=1, ls="--", label="Perfect prediction")
ax.scatter(y_test, pred_base, s=22, color="#bbbbbb", label=f"Baseline (RMSE ${fit_tbl.iloc[0,1]:,.0f})")
ax.scatter(y_test, pred_v1, s=22, color="#1f5fa8", label=f"v1 regression (RMSE ${fit_tbl.iloc[1,1]:,.0f})")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("Actual sale price ($)")
ax.set_ylabel("Predicted sale price ($)")
ax.set_title("Test cars the model never saw: predicted vs actual")
fmt = matplotlib.ticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k")
ax.xaxis.set_major_formatter(fmt); ax.yaxis.set_major_formatter(fmt)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "pred_vs_actual_test.png"), dpi=200)

# ---------------------------------------------------------------- 4. decision rule
def buy_or_pass(car, auction_price, model=v1, prep_cost=PREP_COST):
    """Buy if auction price + prep cost < predicted lot sale price, else pass.
    `car` is a dict/Series with the feature columns (Accident may be 'yes'/'no')."""
    row = dict(car)
    if "Accident_Yes" not in row:
        row["Accident_Yes"] = int(str(row["Accident"]).strip().lower() == "yes")
    x = np.array([1.0] + [float(row[f]) for f in model["features"]])
    predicted = float(x @ model["beta"])
    return ("buy" if auction_price + prep_cost < predicted else "pass"), predicted


def baseline_buy_or_pass(auction_price, avg=baseline_value, prep_cost=PREP_COST):
    return "buy" if auction_price + prep_cost < avg else "pass"


lots["v1_pred"] = [buy_or_pass(r, r["Auction_Price"])[1] for _, r in lots.iterrows()]
lots["v1_decision"] = [buy_or_pass(r, r["Auction_Price"])[0] for _, r in lots.iterrows()]
lots["base_decision"] = [baseline_buy_or_pass(p) for p in lots["Auction_Price"]]
lots["profit_if_bought"] = lots["Later_Sale_Price"] - lots["Auction_Price"] - PREP_COST


def summarize(decision_col):
    bought = lots[lots[decision_col] == "buy"]
    return {
        "cars_bought": len(bought),
        "money_losing_bought": int((bought["profit_if_bought"] < 0).sum()),
        "total_profit_$": int(bought["profit_if_bought"].sum()),
        "avg_profit_per_car_$": round(bought["profit_if_bought"].mean(), 0) if len(bought) else 0,
    }


decision_tbl = pd.DataFrame({
    "Baseline rule": summarize("base_decision"),
    "v1 rule": summarize("v1_decision"),
}).T
# Reference: perfect hindsight (buy only cars that actually made money)
hindsight = lots[lots["profit_if_bought"] > 0]
print("\n=== 40 auction lots: decision rule results ===")
print(decision_tbl.to_string())
print(f"(Reference — perfect hindsight: buy {len(hindsight)} cars, profit ${hindsight['profit_if_bought'].sum():,})")
diff = decision_tbl.loc["v1 rule", "total_profit_$"] - decision_tbl.loc["Baseline rule", "total_profit_$"]
print(f"v1 minus baseline total profit at this auction: ${diff:,}")
decision_tbl.to_csv(os.path.join(OUT, "auction_decisions_summary.csv"))
lots.to_csv(os.path.join(OUT, "auction_lot_decisions.csv"), index=False)

# Chart: profit per car bought under each rule
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, col, title in [(axes[0], "base_decision", "Baseline rule"), (axes[1], "v1_decision", "v1 rule")]:
    b = lots[lots[col] == "buy"].sort_values("profit_if_bought")
    colors = ["#c0392b" if v < 0 else "#1f5fa8" for v in b["profit_if_bought"]]
    ax.bar(range(len(b)), b["profit_if_bought"], color=colors)
    ax.axhline(0, color="black", lw=0.8)
    tot = b['profit_if_bought'].sum()
    ax.set_title(f"{title}: {len(b)} cars, total {'-' if tot < 0 else ''}${abs(tot):,.0f}")
    ax.set_xticks([])
    ax.set_xlabel("Cars bought (sorted by profit)")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Profit per car ($)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "auction_profit_by_rule.png"), dpi=200)

print(f"\nOutputs written to {OUT}")
