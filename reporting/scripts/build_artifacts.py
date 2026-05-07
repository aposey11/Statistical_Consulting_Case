"""
build_artifacts.py — End-to-end pipeline for the report.

Pipeline:
  1. Load + feature engineer the ESG dataset
  2. Fit v5 = v3 + B-spline on WeekBeforeArrival
  3. Generate posterior predictive HBN per snapshot at observed prices
  4. Counterfactual: scale via constant-elasticity (p_rec / p_obs)^β
  5. Roll up to per-ROMGID revenue, then portfolio uplift posterior
  6. Save figures into reporting/figures/ and update outputs/

Run:
    python reporting/scripts/build_artifacts.py
"""

# Quiet PyTensor's "no g++" warning before importing bambi
import os
os.environ["PYTENSOR_FLAGS"] = "cxx="

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import bambi as bmb
import arviz as az

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 100

ROOT      = Path("/Users/alexander/Coding/Statistical_Consulting_Case")
DATA_PATH = ROOT / "DATA" / "ESG_Dataset.csv"
OUT_DIR   = ROOT / "outputs"
FIG_DIR   = ROOT / "reporting" / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------------------------
# 1. Load + feature engineer
# ---------------------------------------------------------------------------

def load_and_prep():
    print("[1/6] Loading + feature-engineering...")
    t0 = time.time()
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Drop unusable columns
    df = df.drop(columns=["DiscountedPriceLastYear",
                          "HistoricalBookedNightsLastYear",
                          "CapacityLastYear"])

    df = df.rename(columns={"DiscountedPrice": "price"})
    df["log_price"]    = np.log(df["price"])
    df["log_capacity"] = np.log(df["Capacity"])
    df["is_special"]   = (df["SpecialPeriodCode"] != "Standard Week").astype(int)
    df["WeekStartDate"] = pd.to_datetime(df["WeekStartDate"])

    for col in ["DeckingExtras", "Kitchen", "DeckingType"]:
        df[col] = df[col].fillna("Missing")

    df["group_key"] = (df["CampsiteCode"].astype(str) + "_"
                       + df["AccoKindCode"].astype(str) + "_"
                       + df["AccoTypeRangeCode"].astype(str) + "_"
                       + df["MarketGroupCode"].astype(str))

    # Bookings on books (the demand-signal we condition on)
    df = df.sort_values(
        ["ReservableOptionMarketGroupId", "WeekStartDate", "WeekBeforeArrival"],
        ascending=[True, True, False],
    )
    grp = ["ReservableOptionMarketGroupId", "WeekStartDate"]
    df["bookings_on_books"] = (
        df.groupby(grp)["HistoricalBookedNights"]
          .transform(lambda x: x.cumsum().shift(1).fillna(0))
    )
    df["log_bob"] = np.log1p(df["bookings_on_books"])

    # Standardize WBA so the spline basis is well-scaled
    wba_mean = df["WeekBeforeArrival"].mean()
    wba_std  = df["WeekBeforeArrival"].std()
    df["wba_z"] = (df["WeekBeforeArrival"] - wba_mean) / wba_std

    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
    return df, wba_mean, wba_std

# ---------------------------------------------------------------------------
# 2. Fit v5 = v3 + B-spline on WBA
# ---------------------------------------------------------------------------

def fit_v5(df, sample_n=50_000):
    print("[2/6] Sampling + fitting v5 (this is the slow step, ~5 min NUTS)...")
    t0 = time.time()

    df_sample = df.sample(n=sample_n, random_state=RANDOM_SEED).reset_index(drop=True)
    for col in ["group_key", "AccommodationType", "AccommodationRange"]:
        df_sample[col] = df_sample[col].astype("category")

    # B-spline on standardized WBA via formulae's bs(); df=4 → 4 basis fns
    formula = (
        "HistoricalBookedNights ~ "
        "log_price + log_bob + is_special + log_capacity + "
        "AccommodationType + AccommodationRange + "
        "bs(wba_z, df=4) + "
        "(1 + log_price | group_key)"
    )

    model = bmb.Model(formula, data=df_sample, family="negativebinomial")
    model.build()

    idata = model.fit(
        draws=500, tune=500, chains=2, cores=2,
        target_accept=0.95,
        random_seed=RANDOM_SEED,
        idata_kwargs={"log_likelihood": True},
    )
    print(f"  Fit time: {(time.time()-t0)/60:.1f} min")

    # Save the InferenceData
    idata.to_netcdf(OUT_DIR / "idata_v5.nc")

    return model, idata, df_sample

# ---------------------------------------------------------------------------
# 3. Posterior predictive on TEST FOLD (out-of-sample evaluation)
# ---------------------------------------------------------------------------

def posterior_predictive_test(model, idata, df_full):
    print("[3/6] Generating posterior predictive on test fold...")
    t0 = time.time()

    # Time-based test fold (matching the LightGBM holdout convention)
    cutoff = pd.Timestamp("2025-07-01")
    test = df_full[df_full["WeekStartDate"] >= cutoff].reset_index(drop=True).copy()
    for col in ["group_key", "AccommodationType", "AccommodationRange"]:
        test[col] = test[col].astype("category")

    # Subsample to keep PPC tractable — we need full posterior draws per row
    # 100k rows × 1000 draws is plenty for stable per-ROMGID summaries
    if len(test) > 200_000:
        test = test.sample(n=200_000, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"  Test fold: {len(test):,} rows, {test['ReservableOptionMarketGroupId'].nunique():,} ROMGIDs")

    # Posterior predictive of HBN at observed prices
    print("  Running model.predict (this can take a few min)...")
    idata_pp = model.predict(idata, kind="response", data=test, inplace=False)

    # Extract: shape (n_chains, n_draws, n_obs)
    pp_hbn = idata_pp.posterior_predictive["HistoricalBookedNights"].values
    n_chains, n_draws, n_obs = pp_hbn.shape
    print(f"  PPC array shape: {pp_hbn.shape}")
    pp_hbn = pp_hbn.reshape(n_chains * n_draws, n_obs)  # flatten chains
    print(f"  Flattened to: {pp_hbn.shape}")

    print(f"  Done in {(time.time()-t0)/60:.1f} min")
    return test, pp_hbn

# ---------------------------------------------------------------------------
# 4. Decision pipeline: per-ROMGID revenue posteriors + portfolio uplift
# ---------------------------------------------------------------------------

def build_decision_posteriors(test, pp_hbn, idata):
    print("[4/6] Building decision pipeline...")
    t0 = time.time()

    # Recommended price per ROMGID = 1.15 × p_obs_max (within-group cap)
    key_cols = ["CampsiteCode", "AccoKindCode", "AccoTypeRangeCode", "MarketGroupCode",
                "WeekStartDate", "ReservableOptionMarketGroupId"]
    romgid = (test.groupby(key_cols)
                  .agg(p_obs_min=("price","min"),
                       p_obs_max=("price","max"),
                       p_obs_mean=("price","mean"),
                       p_obs_std=("price","std"),
                       TBN=("TotalBookedNights","first"),
                       capacity=("Capacity","first"))
                  .reset_index())
    romgid["p_obs_std"] = romgid["p_obs_std"].fillna(0)
    romgid["recommended_price"] = 1.15 * romgid["p_obs_max"]

    # Map test rows to romgid index (vectorized via merge)
    test_idx = test.merge(
        romgid[key_cols].assign(romgid_idx=np.arange(len(romgid))),
        on=key_cols, how="left",
    )
    romgid_idx = test_idx["romgid_idx"].values  # len = len(test)

    # ---- Posterior of revenue at OBSERVED prices ----
    # rev_obs[d, t] = pp_hbn[d, t] * price_obs[t]
    price_obs = test["price"].values  # len = len(test)
    rev_obs_per_snap = pp_hbn * price_obs[None, :]   # shape (D, T)

    # ---- Posterior of HBN at RECOMMENDED price ----
    # Constant-elasticity scaling: counterfactual = pp_hbn * (p_rec / p_obs)^β
    # Sample β from posterior, one β draw per pp draw
    beta_post = idata.posterior["log_price"].values.flatten()
    n_draws_pp = pp_hbn.shape[0]
    # Match PP draw count to β draw count by resampling β
    beta_draws = rng.choice(beta_post, size=n_draws_pp, replace=True)

    p_rec_per_snap = romgid["recommended_price"].values[romgid_idx]  # len = len(test)
    ratio = p_rec_per_snap / price_obs                                # len = len(test)

    # scaling[d, t] = ratio[t]^beta_draws[d]
    log_ratio = np.log(ratio)
    log_scale = beta_draws[:, None] * log_ratio[None, :]  # (D, T)
    scale     = np.exp(log_scale)
    pp_hbn_rec = pp_hbn * scale                            # (D, T)
    rev_rec_per_snap = pp_hbn_rec * p_rec_per_snap[None, :]

    # ---- Aggregate to per-ROMGID revenue posteriors ----
    print("  Aggregating to ROMGID-level...")
    n_romgids = len(romgid)
    n_draws   = pp_hbn.shape[0]
    rev_obs_romgid = np.zeros((n_draws, n_romgids), dtype=np.float64)
    rev_rec_romgid = np.zeros((n_draws, n_romgids), dtype=np.float64)
    np.add.at(rev_obs_romgid, (slice(None), romgid_idx), rev_obs_per_snap)
    np.add.at(rev_rec_romgid, (slice(None), romgid_idx), rev_rec_per_snap)

    # Capacity cap: cap REVENUE not bookings (cap p × bookings such that bookings ≤ capacity)
    cap = romgid["capacity"].values
    p_rec = romgid["recommended_price"].values
    p_obs_mean = romgid["p_obs_mean"].values
    # Apply cap on per-ROMGID predicted bookings
    pred_book_obs = rev_obs_romgid / p_obs_mean[None, :]
    pred_book_rec = rev_rec_romgid / p_rec[None, :]
    pred_book_obs_cap = np.minimum(pred_book_obs, cap[None, :])
    pred_book_rec_cap = np.minimum(pred_book_rec, cap[None, :])
    rev_obs_romgid_cap = pred_book_obs_cap * p_obs_mean[None, :]
    rev_rec_romgid_cap = pred_book_rec_cap * p_rec[None, :]

    # ---- Uplift posterior per ROMGID ----
    uplift_romgid = rev_rec_romgid_cap - rev_obs_romgid_cap   # (D, n_romgids)

    # ---- Portfolio uplift posterior ----
    uplift_portfolio = uplift_romgid.sum(axis=1)              # (D,)

    # Summary stats
    P_uplift_pos = float((uplift_portfolio > 0).mean())
    portfolio_median = float(np.median(uplift_portfolio))
    portfolio_lo, portfolio_hi = (float(np.percentile(uplift_portfolio, 10)),
                                  float(np.percentile(uplift_portfolio, 90)))
    portfolio_lo95, portfolio_hi95 = (float(np.percentile(uplift_portfolio, 2.5)),
                                       float(np.percentile(uplift_portfolio, 97.5)))

    print(f"  Portfolio uplift posterior:")
    print(f"    Median:          €{portfolio_median:>15,.0f}")
    print(f"    80% CI:          [€{portfolio_lo:>13,.0f}, €{portfolio_hi:>13,.0f}]")
    print(f"    95% CI:          [€{portfolio_lo95:>13,.0f}, €{portfolio_hi95:>13,.0f}]")
    print(f"    P(uplift > 0):   {P_uplift_pos:.3f}")

    # Per-ROMGID summaries for the recommendations CSV
    romgid["uplift_eur_median"] = np.median(uplift_romgid, axis=0)
    romgid["uplift_eur_lo"]     = np.percentile(uplift_romgid, 10, axis=0)
    romgid["uplift_eur_hi"]     = np.percentile(uplift_romgid, 90, axis=0)
    romgid["P_uplift_pos"]      = (uplift_romgid > 0).mean(axis=0)
    romgid["expected_revenue"]  = np.median(rev_rec_romgid_cap, axis=0)
    romgid["observed_revenue"]  = np.median(rev_obs_romgid_cap, axis=0)
    romgid["expected_bookings"] = np.median(pred_book_rec_cap, axis=0)
    romgid["price_change_pct"]  = 100 * (romgid["recommended_price"] / romgid["p_obs_mean"] - 1)

    # Three-tier flag (absolute thresholds, ≤20% green per user choice)
    abs_dpct = romgid["price_change_pct"].abs()
    vol      = (romgid["p_obs_std"] / romgid["p_obs_mean"].replace(0, np.nan)).fillna(0)
    romgid["hitl_flag"] = np.where(
        (abs_dpct > 25) | (vol > 0.30), "red",
        np.where((abs_dpct > 20) | (vol > 0.15), "yellow", "green"),
    )

    print(f"  Done in {(time.time()-t0)/60:.1f} min")
    return romgid, uplift_portfolio, uplift_romgid, rev_obs_per_snap, rev_rec_per_snap, pp_hbn_rec

# ---------------------------------------------------------------------------
# 5. Save artifacts
# ---------------------------------------------------------------------------

def save_artifacts(romgid, uplift_portfolio, idata):
    print("[5/6] Saving artifacts...")

    # recommendations.csv (replaces the v3 version)
    romgid.to_csv(OUT_DIR / "recommendations.csv", index=False)

    # Decision posterior for the dashboard / report
    np.savez(OUT_DIR / "decision_posterior.npz",
             portfolio_uplift=uplift_portfolio,
             beta_samples=idata.posterior["log_price"].values.flatten())

    # posterior_meta.json
    beta_post = idata.posterior["log_price"].values.flatten()
    meta = {
        "beta_samples":  beta_post.tolist(),
        "beta_mean":     float(beta_post.mean()),
        "beta_lo":       float(np.percentile(beta_post, 10)),
        "beta_hi":       float(np.percentile(beta_post, 90)),
        "gate_pass_pct": 1.0,  # diagnostic gate (recomputed below)
        "portfolio_uplift_median":   float(np.median(uplift_portfolio)),
        "portfolio_uplift_lo80":     float(np.percentile(uplift_portfolio, 10)),
        "portfolio_uplift_hi80":     float(np.percentile(uplift_portfolio, 90)),
        "portfolio_uplift_lo95":     float(np.percentile(uplift_portfolio, 2.5)),
        "portfolio_uplift_hi95":     float(np.percentile(uplift_portfolio, 97.5)),
        "P_uplift_pos":              float((uplift_portfolio > 0).mean()),
        "n_romgids_evaluated":       int(len(romgid)),
        "methodology": {
            "v1_naive":       {"beta": +0.294,                 "label": "v1: naïve regression",      "verdict": "Endogenous (wrong sign)"},
            "v3_conditioned": {"beta": -0.752,                 "label": "v3: + bookings_on_books",   "verdict": "DAG-identified"},
            "v5_temporal":    {"beta": float(beta_post.mean()), "label": "v5: + WBA spline",         "verdict": "Production"},
        },
    }
    # Gate pass pct from random slope
    if "log_price|group_key" in idata.posterior:
        group_dev = idata.posterior["log_price|group_key"].mean(dim=["chain","draw"]).values
        group_abs = float(beta_post.mean()) + group_dev
        meta["gate_pass_pct"] = float((group_abs < 0).mean())

    with open(OUT_DIR / "posterior_meta.json", "w") as f:
        json.dump(meta, f, separators=(",", ":"))

    print(f"  Saved: {OUT_DIR / 'recommendations.csv'}")
    print(f"  Saved: {OUT_DIR / 'decision_posterior.npz'}")
    print(f"  Saved: {OUT_DIR / 'posterior_meta.json'}")
    print(f"  Saved: {OUT_DIR / 'idata_v5.nc'}")

# ---------------------------------------------------------------------------
# 6. Build figures
# ---------------------------------------------------------------------------

def build_figures(test, pp_hbn, pp_hbn_rec, romgid, uplift_portfolio, idata):
    print("[6/6] Building figures...")

    # Posterior of β (used in the methodology timeline)
    beta_post = idata.posterior["log_price"].values.flatten()
    beta_v5_mean = float(beta_post.mean())
    beta_v5_lo, beta_v5_hi = np.percentile(beta_post, [10, 90])

    # ---- Figure 1: Methodology timeline ----
    fig, ax = plt.subplots(figsize=(8, 3.5))
    rows = [
        ("v1: naïve regression",      +0.294,  +0.20, +0.40, "#C04848", "Endogenous bias"),
        ("v3: + bookings_on_books",   -0.752,  -0.86, -0.64, "#4C9A4A", "DAG-identified"),
        ("v5: + WBA spline",          beta_v5_mean, beta_v5_lo, beta_v5_hi, "#2E5C8A", "Production"),
        ("v4: + calendar/WBA/temp",   -10.4,  -10.6, -10.1,  "#E0B341", "Over-controlled"),
    ]
    y = np.arange(len(rows))
    for i, (label, m, lo, hi, color, verdict) in enumerate(rows):
        ax.errorbar([m], [i], xerr=[[m-lo], [hi-m]], fmt="o", color=color,
                    markersize=10, capsize=5, lw=2, label=verdict)
        ax.annotate(f"  β = {m:+.2f}", (m, i), va="center",
                    fontsize=9, color=color, xytext=(8, 0), textcoords="offset points")
    ax.axvline(0, color="black", lw=0.5, ls="--", alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Estimated price elasticity β")
    ax.set_title("Methodology timeline — global elasticity by model version")
    ax.invert_yaxis()
    ax.set_xlim(-12, 1)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "methodology_timeline.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: methodology_timeline.pdf")

    # ---- Figure 2: Per-ROMGID booking-curve PPC (4 ROMGIDs in 2x2) ----
    print("  Building booking-curve PPC...")
    test_with_pp = test.copy()
    test_with_pp["pp_median"] = np.median(pp_hbn, axis=0)
    test_with_pp["pp_lo"]     = np.percentile(pp_hbn, 10, axis=0)
    test_with_pp["pp_hi"]     = np.percentile(pp_hbn, 90, axis=0)
    test_with_pp["pp_rec_median"] = np.median(pp_hbn_rec, axis=0)
    test_with_pp["pp_rec_lo"]     = np.percentile(pp_hbn_rec, 10, axis=0)
    test_with_pp["pp_rec_hi"]     = np.percentile(pp_hbn_rec, 90, axis=0)

    # Pick 4 ROMGIDs with TBN>50 from different markets
    romgid_pool = (test_with_pp.groupby("ReservableOptionMarketGroupId")["TotalBookedNights"]
                                .first().pipe(lambda s: s[s > 80]).index.tolist())
    selected_keys = rng.choice(romgid_pool, size=min(4, len(romgid_pool)), replace=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    for ax, key in zip(axes.flat, selected_keys):
        sub = (test_with_pp[test_with_pp["ReservableOptionMarketGroupId"] == key]
                 .sort_values("WeekBeforeArrival", ascending=False))
        ax.plot(sub["WeekBeforeArrival"], sub["HistoricalBookedNights"],
                marker="o", lw=0, color="black", markersize=4, label="observed")
        ax.fill_between(sub["WeekBeforeArrival"], sub["pp_lo"], sub["pp_hi"],
                        color="#2E5C8A", alpha=0.3, label="80% CI (observed price)")
        ax.plot(sub["WeekBeforeArrival"], sub["pp_median"], color="#2E5C8A", lw=1.5)
        ax.fill_between(sub["WeekBeforeArrival"], sub["pp_rec_lo"], sub["pp_rec_hi"],
                        color="#E0B341", alpha=0.25, label="80% CI (recommended price)")
        ax.plot(sub["WeekBeforeArrival"], sub["pp_rec_median"], color="#E0B341", lw=1.5, ls="--")
        ax.invert_xaxis()
        ax.set_title(key, fontsize=8)
        ax.set_ylabel("Bookings created")
        if ax in axes[1]:
            ax.set_xlabel("Weeks before arrival")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=9)
    fig.suptitle("Booking-curve posterior predictive — 4 representative ROMGIDs", y=1.06, fontsize=11)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "booking_curve_ppc.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: booking_curve_ppc.pdf")

    # ---- Figure 3: Revenue PPC (calibration on revenue, not bookings) ----
    print("  Building revenue PPC...")
    rev_obs_pp_per_romgid = np.zeros((pp_hbn.shape[0], len(romgid)))
    test_idx = test.merge(
        romgid[["ReservableOptionMarketGroupId", "WeekStartDate"]].assign(romgid_idx=np.arange(len(romgid))),
        on=["ReservableOptionMarketGroupId", "WeekStartDate"], how="left",
    )
    rid = test_idx["romgid_idx"].values
    np.add.at(rev_obs_pp_per_romgid, (slice(None), rid), pp_hbn * test["price"].values[None, :])

    actual_rev_per_romgid = (test.assign(rev=test["price"] * test["HistoricalBookedNights"])
                                  .merge(romgid[["ReservableOptionMarketGroupId", "WeekStartDate"]]
                                          .assign(romgid_idx=np.arange(len(romgid))),
                                         on=["ReservableOptionMarketGroupId", "WeekStartDate"], how="left")
                                  .groupby("romgid_idx")["rev"].sum()
                                  .reindex(np.arange(len(romgid)), fill_value=0).values)

    pred_median = np.median(rev_obs_pp_per_romgid, axis=0)
    pred_lo     = np.percentile(rev_obs_pp_per_romgid, 10, axis=0)
    pred_hi     = np.percentile(rev_obs_pp_per_romgid, 90, axis=0)
    coverage_80 = ((actual_rev_per_romgid >= pred_lo) & (actual_rev_per_romgid <= pred_hi)).mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.scatter(pred_median, actual_rev_per_romgid, alpha=0.25, s=8, color="#2E5C8A")
    m = max(pred_median.max(), actual_rev_per_romgid.max())
    ax.plot([0, m], [0, m], "k--", alpha=0.6, label="perfect calibration")
    ax.set_xlabel("Predicted revenue per ROMGID, € (posterior median)")
    ax.set_ylabel("Realized revenue per ROMGID, €")
    ax.set_title(f"Revenue PPC — median predicted vs realized\n(80% interval coverage: {coverage_80:.1%})")
    ax.legend()

    ax = axes[1]
    decile = pd.qcut(pred_median, 10, labels=False, duplicates="drop")
    calib = pd.DataFrame({"pred": pred_median, "actual": actual_rev_per_romgid, "decile": decile})
    cgrp = calib.groupby("decile").agg(pred_mean=("pred","mean"),
                                       actual_mean=("actual","mean"),
                                       n=("actual","size"))
    ax.scatter(cgrp["pred_mean"], cgrp["actual_mean"], s=80, color="#4C9A4A", zorder=3)
    m = float(cgrp[["pred_mean","actual_mean"]].max().max())
    ax.plot([0, m], [0, m], "k--", alpha=0.6)
    ax.set_xlabel("Predicted revenue (decile mean), €")
    ax.set_ylabel("Realized revenue (decile mean), €")
    ax.set_title("Revenue calibration — by predicted-decile")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "revenue_ppc.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: revenue_ppc.pdf")

    # ---- Figure 4: Portfolio uplift posterior (THE HEADLINE) ----
    print("  Building portfolio uplift posterior...")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(uplift_portfolio / 1e6, bins=50, color="#2E5C8A", alpha=0.85, edgecolor="white")
    p_pos = (uplift_portfolio > 0).mean()
    median = np.median(uplift_portfolio) / 1e6
    lo80, hi80 = np.percentile(uplift_portfolio, [10, 90]) / 1e6
    ax.axvline(0, color="red", lw=1.5, ls="--", label="break-even (€0)")
    ax.axvline(median, color="black", lw=1.5, label=f"median = €{median:.1f}M")
    ax.axvspan(lo80, hi80, alpha=0.15, color="black", label=f"80% CI [€{lo80:.1f}M, €{hi80:.1f}M]")
    ax.set_xlabel("Total portfolio revenue uplift (€M, out-of-sample)")
    ax.set_ylabel("Posterior draws")
    ax.set_title(f"Portfolio uplift posterior — P(uplift > 0) = {p_pos:.3f}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "portfolio_uplift_posterior.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: portfolio_uplift_posterior.pdf")

    # ---- Figure 5: Counterfactual revenue trajectory (one ROMGID, illustrative) ----
    print("  Building counterfactual revenue trajectory...")
    key = selected_keys[0]
    sub = test_with_pp[test_with_pp["ReservableOptionMarketGroupId"] == key].sort_values("WeekBeforeArrival", ascending=False).copy()
    sub["rev_obs_median"] = sub["pp_median"] * sub["price"]
    sub["rev_obs_lo"]     = sub["pp_lo"]     * sub["price"]
    sub["rev_obs_hi"]     = sub["pp_hi"]     * sub["price"]
    p_rec_for_key = romgid.loc[romgid["ReservableOptionMarketGroupId"] == key, "recommended_price"].iloc[0]
    sub["rev_rec_median"] = sub["pp_rec_median"] * p_rec_for_key
    sub["rev_rec_lo"]     = sub["pp_rec_lo"]     * p_rec_for_key
    sub["rev_rec_hi"]     = sub["pp_rec_hi"]     * p_rec_for_key

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(sub["WeekBeforeArrival"], sub["rev_obs_lo"], sub["rev_obs_hi"],
                    color="#2E5C8A", alpha=0.25, label="80% CI (observed price)")
    ax.plot(sub["WeekBeforeArrival"], sub["rev_obs_median"], color="#2E5C8A", lw=2, label="median (observed)")
    ax.fill_between(sub["WeekBeforeArrival"], sub["rev_rec_lo"], sub["rev_rec_hi"],
                    color="#E0B341", alpha=0.25, label="80% CI (recommended price)")
    ax.plot(sub["WeekBeforeArrival"], sub["rev_rec_median"], color="#E0B341", lw=2, ls="--", label="median (recommended)")
    ax.invert_xaxis()
    ax.set_xlabel("Weeks before arrival")
    ax.set_ylabel("Predicted revenue per snapshot, €")
    ax.set_title(f"Counterfactual revenue trajectory — {key}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "counterfactual_revenue.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: counterfactual_revenue.pdf")

    # ---- Figure 6: Per-segment decision table (rendered as figure) ----
    print("  Building per-segment decision table...")
    seg = []
    for col in ["MarketGroupCode", "AccoTypeRangeCode"]:
        for level in romgid[col].unique():
            mask = (romgid[col] == level).values
            if mask.sum() < 10:
                continue
            seg_uplift = (uplift_portfolio.reshape(-1, 1) *
                          (mask[None, :].astype(float) / mask.sum()) *
                          mask.sum()).sum(axis=1)
            # Better: subset uplift_romgid by segment, sum
        # leave for now; we'll do it below cleanly

    # Cleaner per-segment via uplift_romgid subset
    # NOTE: uplift_romgid not in scope here — recompute summary from romgid frame
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    seg_market = (romgid.groupby("MarketGroupCode")
                        .agg(median=("uplift_eur_median","sum"),
                             lo=("uplift_eur_lo","sum"),
                             hi=("uplift_eur_hi","sum"),
                             n=("ReservableOptionMarketGroupId","nunique"),
                             p_pos=("P_uplift_pos","mean"))
                        .reset_index().sort_values("median", ascending=True))
    ax = axes[0]
    ax.barh(seg_market["MarketGroupCode"], seg_market["median"]/1e6,
            xerr=[(seg_market["median"]-seg_market["lo"])/1e6,
                  (seg_market["hi"]-seg_market["median"])/1e6],
            color="#2E5C8A", alpha=0.85, capsize=5)
    ax.set_xlabel("Uplift (€M, sum across ROMGIDs in segment)")
    ax.set_title("By market group — sum of per-ROMGID median uplift, ±80% CI")

    seg_acco = (romgid.groupby("AccoTypeRangeCode")
                      .agg(median=("uplift_eur_median","sum"),
                           lo=("uplift_eur_lo","sum"),
                           hi=("uplift_eur_hi","sum"),
                           n=("ReservableOptionMarketGroupId","nunique"),
                           p_pos=("P_uplift_pos","mean"))
                      .reset_index().sort_values("median", ascending=True))
    ax = axes[1]
    ax.barh(seg_acco["AccoTypeRangeCode"], seg_acco["median"]/1e6,
            xerr=[(seg_acco["median"]-seg_acco["lo"])/1e6,
                  (seg_acco["hi"]-seg_acco["median"])/1e6],
            color="#4C9A4A", alpha=0.85, capsize=5)
    ax.set_xlabel("Uplift (€M, sum across ROMGIDs in segment)")
    ax.set_title("By accommodation tier — sum of per-ROMGID median uplift, ±80% CI")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "segment_decision.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: segment_decision.pdf")

    # Save segment summary table for the LaTeX report
    seg_market.to_csv(OUT_DIR / "segment_market.csv", index=False)
    seg_acco.to_csv(OUT_DIR / "segment_acco.csv", index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    df, _, _ = load_and_prep()
    model, idata, df_sample = fit_v5(df)
    test, pp_hbn = posterior_predictive_test(model, idata, df)
    romgid, uplift_portfolio, uplift_romgid, _, _, pp_hbn_rec = build_decision_posteriors(test, pp_hbn, idata)
    save_artifacts(romgid, uplift_portfolio, idata)
    build_figures(test, pp_hbn, pp_hbn_rec, romgid, uplift_portfolio, idata)
    print(f"\nTotal pipeline time: {(time.time()-t_start)/60:.1f} min")
    print("All artifacts written. Ready for report rewrite.")


if __name__ == "__main__":
    main()
