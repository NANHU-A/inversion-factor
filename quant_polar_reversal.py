from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 140
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class Config:
    data_path: str = "exam_data.parquet"
    output_dir: str = "outputs"
    periods: tuple[int, ...] = (20, 60, 120, 240)
    cov_window: int = 60
    fwd_horizon: int = 1
    quantiles: int = 5
    angle_target: float = math.pi / 4
    angle_sigma: float = math.pi / 4
    min_cross_section: int = 20
    post_winsor_lower: float = 0.01
    post_winsor_upper: float = 0.99
    dynamic_lookback: int = 60
    annual_trading_days: int = 252
    eps: float = 1e-12


def make_output_dirs(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "tables": root / "tables",
        "figures": root / "figures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def print_section(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "trade_date": "date",
        "symbol": "stock",
    }
    return df.rename(columns=rename_map)


def load_and_preprocess(cfg: Config) -> tuple[pd.DataFrame, dict]:
    data_path = Path(cfg.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find data file: {data_path}")

    raw = pd.read_parquet(data_path)
    raw = rename_columns(raw)

    required_cols = {"date", "stock", "close", "amount"}
    missing = required_cols - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    before_rows = len(raw)
    duplicate_rows = raw.duplicated(subset=["date", "stock"]).sum()

    df = raw.copy()
    df["date"] = pd.to_datetime(
        df["date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    df["stock"] = df["stock"].astype(str)

    numeric_cols = [
        c
        for c in [
            "close",
            "amount",
            "volume",
            "vwap",
            "adj_factor",
            "open",
            "high",
            "low",
            "pre_close",
            "turnover",
            "turnover_float",
            "mv",
            "mv_float",
            "new_ipo",
            "limitupdown",
            "limitupdown_at_close",
        ]
        if c in df.columns
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    missing_ratio = (
        df[[c for c in ["date", "stock", "close", "amount"] if c in df.columns]]
        .isna()
        .mean()
    )

    df = df.drop_duplicates(subset=["date", "stock"], keep="last").copy()
    df = df.dropna(subset=["date", "stock", "close", "amount"]).copy()
    df = df[(df["close"] > 0) & (df["amount"] > 0)].copy()

    if "volume" in df.columns:
        df = df[df["volume"].fillna(0) > 0].copy()

    if "new_ipo" in df.columns:
        df = df[df["new_ipo"].fillna(0) == 0].copy()

    df = df.sort_values(["stock", "date"]).reset_index(drop=True)

    if "mv_float" in df.columns:
        df["log_size"] = np.log(df["mv_float"].where(df["mv_float"] > 0))
    elif "mv" in df.columns:
        df["log_size"] = np.log(df["mv"].where(df["mv"] > 0))
    else:
        df["log_size"] = np.nan

    df["log_amount"] = np.log(df["amount"])

    stats = {
        "rows_before": int(before_rows),
        "rows_after": int(len(df)),
        "duplicate_rows": int(duplicate_rows),
        "start_date": df["date"].min(),
        "end_date": df["date"].max(),
        "n_stocks": int(df["stock"].nunique()),
        "n_dates": int(df["date"].nunique()),
        "missing_ratio": missing_ratio,
        "describe": df[["close", "amount"]].describe().round(4),
    }
    return df, stats


def export_basic_statistics(
    df: pd.DataFrame, stats: dict, out_dirs: dict[str, Path]
) -> None:
    print_section("1. Data Preprocessing Summary")
    print(f"Rows before cleaning: {stats['rows_before']:,}")
    print(f"Rows after cleaning : {stats['rows_after']:,}")
    print(f"Duplicate rows      : {stats['duplicate_rows']:,}")
    print(f"Date range          : {stats['start_date']} -> {stats['end_date']}")
    print(f"Stocks              : {stats['n_stocks']:,}")
    print(f"Trading dates       : {stats['n_dates']:,}")
    print("\nMissing ratio of key columns before cleaning:")
    print(stats["missing_ratio"].round(4).to_string())
    print("\nDescribe close / amount:")
    print(stats["describe"].to_string())

    stats["describe"].to_csv(out_dirs["tables"] / "basic_describe.csv")

    pd.DataFrame(
        {
            "metric": [
                "rows_before",
                "rows_after",
                "duplicate_rows",
                "n_stocks",
                "n_dates",
            ],
            "value": [
                stats["rows_before"],
                stats["rows_after"],
                stats["duplicate_rows"],
                stats["n_stocks"],
                stats["n_dates"],
            ],
        }
    ).to_csv(out_dirs["tables"] / "basic_summary.csv", index=False)


def add_forward_return(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    if "adj_factor" in df.columns:
        df["adj_close"] = df["close"] * df["adj_factor"]
        future_adj_close = df.groupby("stock", sort=False)["adj_close"].shift(-horizon)
        df["fwd_ret"] = future_adj_close / df["adj_close"] - 1.0
        df["ret_source"] = "adj_close"
    else:
        future_close = df.groupby("stock", sort=False)["close"].shift(-horizon)
        df["fwd_ret"] = future_close / df["close"] - 1.0
        df["ret_source"] = "close"
    return df


def grouped_rolling_mean(series: pd.Series, group: pd.Series, window: int) -> pd.Series:
    return (
        series.groupby(group)
        .rolling(window=window, min_periods=window)
        .mean()
        .reset_index(level=0, drop=True)
    )


def prepare_covariance_terms(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    group_key = df["stock"]
    mean_close = grouped_rolling_mean(df["close"], group_key, cfg.cov_window)
    mean_amount = grouped_rolling_mean(df["amount"], group_key, cfg.cov_window)
    mean_close_sq = grouped_rolling_mean(df["close"] ** 2, group_key, cfg.cov_window)
    mean_amount_sq = grouped_rolling_mean(df["amount"] ** 2, group_key, cfg.cov_window)
    mean_cross = grouped_rolling_mean(
        df["close"] * df["amount"], group_key, cfg.cov_window
    )

    df["var_close"] = (mean_close_sq - mean_close**2).clip(lower=0) + cfg.eps
    df["var_amount"] = (mean_amount_sq - mean_amount**2).clip(lower=0) + cfg.eps
    df["cov_close_amount"] = mean_cross - mean_close * mean_amount
    df["cov_det"] = df["var_close"] * df["var_amount"] - df["cov_close_amount"] ** 2
    df["cov_det"] = df["cov_det"].where(df["cov_det"] > cfg.eps, np.nan)
    return df


def circular_distance(
    theta: pd.Series | np.ndarray, target: float
) -> pd.Series | np.ndarray:
    diff = np.abs(theta - target)
    return np.minimum(diff, 2 * np.pi - diff)


def angle_weight(theta: pd.Series, cfg: Config) -> pd.Series:
    dist = circular_distance(theta, cfg.angle_target)
    return np.exp(-(dist**2) / (2 * cfg.angle_sigma**2))


def preference_alpha(price_ret: pd.Series, amount_ret: pd.Series) -> pd.Series:
    alpha = np.select(
        [
            (price_ret >= 0) & (amount_ret >= 0),
            (price_ret < 0) & (amount_ret >= 0),
            (price_ret < 0) & (amount_ret < 0),
            (price_ret >= 0) & (amount_ret < 0),
        ],
        [1.0, -0.5, -1.0, 0.75],
        default=np.nan,
    )
    return pd.Series(alpha, index=price_ret.index)


def build_single_period_factor(df: pd.DataFrame, n: int, cfg: Config) -> pd.DataFrame:
    g = df.groupby("stock", sort=False)

    close_lag = g["close"].shift(n)
    amount_lag = g["amount"].shift(n)

    price_diff = df["close"] - close_lag
    amount_diff = df["amount"] - amount_lag
    price_ret = df["close"] / close_lag - 1.0
    amount_ret = df["amount"] / amount_lag - 1.0

    theta = np.mod(np.arctan2(amount_ret, price_ret), 2 * np.pi)
    alpha = preference_alpha(price_ret, amount_ret)
    f_theta = angle_weight(theta, cfg)

    rho_sq = (
        df["var_amount"] * (price_diff**2)
        - 2 * df["cov_close_amount"] * price_diff * amount_diff
        + df["var_close"] * (amount_diff**2)
    ) / df["cov_det"]
    rho_sq = rho_sq.where(rho_sq >= 0, 0.0)
    rho = np.sqrt(rho_sq)

    factor_col = f"factor_{n}"
    df[f"price_ret_{n}"] = price_ret
    df[f"amount_ret_{n}"] = amount_ret
    df[f"theta_{n}"] = theta
    df[f"rho_{n}"] = rho
    df[factor_col] = alpha * rho * f_theta
    df[factor_col] = df[factor_col].replace([np.inf, -np.inf], np.nan)
    return df


def build_baseline_factors(
    df: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, list[str]]:
    df = prepare_covariance_terms(df, cfg)
    factor_cols: list[str] = []
    for n in cfg.periods:
        df = build_single_period_factor(df, n, cfg)
        factor_cols.append(f"factor_{n}")

    df["factor_composite"] = df[factor_cols].mean(axis=1, skipna=True)
    return df, factor_cols


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) == 0:
        return np.nan
    if valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return np.nan
    return valid.iloc[:, 0].corr(valid.iloc[:, 1], method=method)


def daily_ic_series(
    df: pd.DataFrame,
    factor_col: str,
    ret_col: str,
    method: str,
    min_cross_section: int,
) -> pd.Series:
    sub = df[["date", factor_col, ret_col]].dropna().copy()

    def _corr(grp: pd.DataFrame) -> float:
        if len(grp) < min_cross_section:
            return np.nan
        return safe_corr(grp[factor_col], grp[ret_col], method)

    result = sub.groupby("date", sort=True).apply(_corr)
    result.name = f"{factor_col}_{method}"
    return result.dropna()


def summarize_series(series: pd.Series, annual_days: int) -> dict:
    series = series.dropna()
    if series.empty:
        return {
            "mean": np.nan,
            "std": np.nan,
            "t_stat": np.nan,
            "ir": np.nan,
            "positive_ratio": np.nan,
            "n_days": 0,
        }
    mean = series.mean()
    std = series.std(ddof=1)
    t_stat = mean / (std / np.sqrt(len(series)) + 1e-12)
    ir = mean / (std + 1e-12) * np.sqrt(annual_days)
    return {
        "mean": float(mean),
        "std": float(std),
        "t_stat": float(t_stat),
        "ir": float(ir),
        "positive_ratio": float((series > 0).mean()),
        "n_days": int(len(series)),
    }


def assign_quantile_labels(series: pd.Series, n_quantiles: int) -> pd.Series:
    result = pd.Series(np.nan, index=series.index)
    valid = series.dropna()
    if len(valid) < n_quantiles or valid.nunique() < n_quantiles:
        return result
    ranked = valid.rank(method="first")
    labels = pd.qcut(ranked, n_quantiles, labels=False) + 1
    result.loc[valid.index] = labels.astype(float)
    return result


def quantile_backtest(
    df: pd.DataFrame,
    factor_col: str,
    ret_col: str,
    n_quantiles: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    sub = df[["date", factor_col, ret_col]].copy()
    sub["quantile"] = sub.groupby("date")[factor_col].transform(
        lambda s: assign_quantile_labels(s, n_quantiles)
    )

    group_ret = (
        sub.dropna(subset=["quantile", ret_col])
        .groupby(["date", "quantile"])[ret_col]
        .mean()
        .unstack()
        .sort_index()
    )
    if group_ret.empty:
        empty_stats = {
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "win_rate": np.nan,
        }
        return group_ret, group_ret, pd.Series(dtype=float), empty_stats

    cum_ret = (1 + group_ret.fillna(0)).cumprod()
    long_short = group_ret[n_quantiles] - group_ret[1]
    ls_stats = portfolio_stats(long_short)
    return group_ret, cum_ret, long_short, ls_stats


def portfolio_stats(ret: pd.Series, annual_days: int = 252) -> dict:
    ret = ret.dropna()
    if ret.empty:
        return {
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "win_rate": np.nan,
        }
    nav = (1 + ret).cumprod()
    ann_return = nav.iloc[-1] ** (annual_days / len(nav)) - 1
    ann_vol = ret.std(ddof=0) * np.sqrt(annual_days)
    sharpe = ann_return / (ann_vol + 1e-12)
    drawdown = nav / nav.cummax() - 1
    max_drawdown = drawdown.min()
    win_rate = (ret > 0).mean()
    return {
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
    }


def winsorize_by_date(
    series: pd.Series, dates: pd.Series, lower: float, upper: float
) -> pd.Series:
    def _clip(x: pd.Series) -> pd.Series:
        if x.dropna().empty:
            return x
        lo = x.quantile(lower)
        hi = x.quantile(upper)
        return x.clip(lo, hi)

    return series.groupby(dates).transform(_clip)


def zscore_by_date(series: pd.Series, dates: pd.Series) -> pd.Series:
    return series.groupby(dates).transform(
        lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-12)
    )


def neutralize_cross_section(
    df: pd.DataFrame,
    factor_col: str,
    size_col: str = "log_size",
    industry_col: str = "ind_code",
) -> pd.Series:
    def _neutralize_one_day(grp: pd.DataFrame) -> pd.Series:
        result = pd.Series(np.nan, index=grp.index)
        y = grp[factor_col]
        if y.notna().sum() < 10:
            return result

        X_parts: list[pd.DataFrame] = [
            pd.DataFrame({"intercept": 1.0}, index=grp.index)
        ]

        if size_col in grp.columns and grp[size_col].notna().sum() > 0:
            X_parts.append(pd.DataFrame({size_col: grp[size_col]}, index=grp.index))

        if industry_col in grp.columns and grp[industry_col].notna().sum() > 0:
            dummies = pd.get_dummies(
                grp[industry_col].astype(str), prefix="ind", drop_first=True
            )
            if not dummies.empty:
                X_parts.append(dummies)

        X = pd.concat(X_parts, axis=1)
        valid = y.notna() & X.notna().all(axis=1)
        if valid.sum() <= X.shape[1]:
            return result

        beta = np.linalg.lstsq(
            X.loc[valid].to_numpy(dtype=float),
            y.loc[valid].to_numpy(dtype=float),
            rcond=None,
        )[0]
        fitted = X.loc[valid].to_numpy(dtype=float) @ beta
        result.loc[valid] = y.loc[valid].to_numpy(dtype=float) - fitted
        return result

    return df.groupby("date", group_keys=False).apply(_neutralize_one_day)


def optimize_postprocess(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    base = "factor_composite"
    df["factor_post_winsor"] = winsorize_by_date(
        df[base], df["date"], cfg.post_winsor_lower, cfg.post_winsor_upper
    )
    df["factor_post_z"] = zscore_by_date(df["factor_post_winsor"], df["date"])
    df["factor_post"] = neutralize_cross_section(df, "factor_post_z")
    df["factor_post"] = zscore_by_date(df["factor_post"], df["date"])
    return df


def optimize_dynamic_weight(
    df: pd.DataFrame, factor_cols: list[str], cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rankic_map = {}
    for col in factor_cols:
        rankic_map[col] = daily_ic_series(
            df, col, "fwd_ret", "spearman", cfg.min_cross_section
        )

    rankic_panel = pd.concat(rankic_map, axis=1).sort_index()
    rolling_score = rankic_panel.rolling(
        cfg.dynamic_lookback, min_periods=max(20, cfg.dynamic_lookback // 3)
    ).mean()
    rolling_score = rolling_score.shift(1).clip(lower=0)

    weight_df = rolling_score.div(rolling_score.sum(axis=1), axis=0)
    if not weight_df.empty:
        equal_weight = 1.0 / len(factor_cols)
        weight_df = weight_df.fillna(equal_weight)

    z_cols = []
    for col in factor_cols:
        z_col = f"{col}_zopt"
        df[z_col] = zscore_by_date(df[col], df["date"])
        z_cols.append(z_col)

    weight_df = weight_df.reset_index().rename(columns={"index": "date"})
    rename_weights = {col: f"weight_{col}" for col in factor_cols}
    weight_df = weight_df.rename(columns=rename_weights)
    df = df.merge(weight_df, on="date", how="left")

    composite = 0
    total_weight = 0
    for col, z_col in zip(factor_cols, z_cols):
        w_col = rename_weights[col]
        composite = composite + df[z_col] * df[w_col]
        total_weight = total_weight + df[w_col]
    df["factor_dynamic"] = composite / total_weight.replace(0, np.nan)
    df["factor_dynamic"] = zscore_by_date(df["factor_dynamic"], df["date"])
    return df, weight_df


def plot_ic(ic: pd.Series, title: str, save_path: Path) -> None:
    if ic.empty:
        return
    rolling = ic.rolling(20, min_periods=5).mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ic.index, ic.values, label="Daily IC", alpha=0.35, linewidth=1.0)
    ax.plot(rolling.index, rolling.values, label="20D Rolling Mean", linewidth=2.0)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_quantiles(
    cum_ret: pd.DataFrame, long_short: pd.Series, title: str, save_path: Path
) -> None:
    if cum_ret.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for col in cum_ret.columns:
        axes[0].plot(cum_ret.index, cum_ret[col], label=f"Q{int(col)}", linewidth=1.5)
    axes[0].set_title(f"{title} Quantile Cumulative Return")
    axes[0].legend()

    ls_nav = (1 + long_short.fillna(0)).cumprod()
    axes[1].plot(ls_nav.index, ls_nav.values, color="#d62728", linewidth=2.0)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_title(f"{title} Long-Short")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate_factor(
    df: pd.DataFrame, factor_col: str, cfg: Config, out_dirs: dict[str, Path]
) -> dict:
    ic = daily_ic_series(df, factor_col, "fwd_ret", "pearson", cfg.min_cross_section)
    rankic = daily_ic_series(
        df, factor_col, "fwd_ret", "spearman", cfg.min_cross_section
    )

    ic_summary = summarize_series(ic, cfg.annual_trading_days)
    rankic_summary = summarize_series(rankic, cfg.annual_trading_days)
    group_ret, cum_ret, long_short, ls_stats = quantile_backtest(
        df, factor_col, "fwd_ret", cfg.quantiles
    )

    if not ic.empty:
        ic.to_csv(out_dirs["tables"] / f"{factor_col}_daily_ic.csv")
    if not rankic.empty:
        rankic.to_csv(out_dirs["tables"] / f"{factor_col}_daily_rankic.csv")
    if not group_ret.empty:
        group_ret.to_csv(out_dirs["tables"] / f"{factor_col}_quantile_return.csv")

    plot_ic(ic, f"{factor_col} IC", out_dirs["figures"] / f"{factor_col}_ic.png")
    plot_quantiles(
        cum_ret,
        long_short,
        factor_col,
        out_dirs["figures"] / f"{factor_col}_quantile.png",
    )

    return {
        "factor": factor_col,
        "ic_mean": ic_summary["mean"],
        "ic_std": ic_summary["std"],
        "ic_ir": ic_summary["ir"],
        "ic_tstat": ic_summary["t_stat"],
        "ic_pos_ratio": ic_summary["positive_ratio"],
        "rankic_mean": rankic_summary["mean"],
        "rankic_std": rankic_summary["std"],
        "rankic_ir": rankic_summary["ir"],
        "rankic_tstat": rankic_summary["t_stat"],
        "rankic_pos_ratio": rankic_summary["positive_ratio"],
        "ls_ann_return": ls_stats["ann_return"],
        "ls_ann_vol": ls_stats["ann_vol"],
        "ls_sharpe": ls_stats["sharpe"],
        "ls_max_drawdown": ls_stats["max_drawdown"],
        "ls_win_rate": ls_stats["win_rate"],
        "coverage": float(df[factor_col].notna().mean()),
    }


def export_factor_coverage(
    df: pd.DataFrame, factor_cols: list[str], out_dirs: dict[str, Path]
) -> None:
    coverage = pd.DataFrame(
        {
            "factor": factor_cols,
            "coverage_ratio": [df[col].notna().mean() for col in factor_cols],
        }
    )
    coverage.to_csv(out_dirs["tables"] / "factor_coverage.csv", index=False)
    print_section("2. Factor Coverage")
    print(coverage.round(4).to_string(index=False))


def plot_summary_bar(summary_df: pd.DataFrame, out_dirs: dict[str, Path]) -> None:
    if summary_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    sns.barplot(data=summary_df, x="factor", y="ic_mean", ax=axes[0], color="#4c78a8")
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1.0)
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set_title("Mean IC")

    sns.barplot(
        data=summary_df, x="factor", y="rankic_mean", ax=axes[1], color="#f58518"
    )
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1.0)
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set_title("Mean RankIC")
    plt.tight_layout()
    plt.savefig(out_dirs["figures"] / "factor_summary_bar.png")
    plt.close()


def run_research(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dirs = make_output_dirs(Path(cfg.output_dir))

    df, stats = load_and_preprocess(cfg)
    export_basic_statistics(df, stats, out_dirs)

    df = add_forward_return(df, cfg.fwd_horizon)
    df, factor_cols = build_baseline_factors(df, cfg)
    baseline_cols = factor_cols + ["factor_composite"]
    export_factor_coverage(df, baseline_cols, out_dirs)

    print_section("3. Optional Optimization Experiments")
    print("Optimization A: winsorize + zscore + size/industry neutralization")
    print(
        "Optimization B: rolling RankIC dynamic weights on standardized single-period factors"
    )

    df = optimize_postprocess(df, cfg)
    df, weight_df = optimize_dynamic_weight(df, factor_cols, cfg)
    weight_df.to_csv(out_dirs["tables"] / "dynamic_weights.csv", index=False)

    eval_cols = baseline_cols + ["factor_post", "factor_dynamic"]
    summary_rows = []
    print_section("4. Factor Evaluation Summary")
    for col in eval_cols:
        row = evaluate_factor(df, col, cfg, out_dirs)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["rankic_mean", "ic_mean"], ascending=False
    )
    summary_df.to_csv(out_dirs["tables"] / "factor_evaluation_summary.csv", index=False)
    plot_summary_bar(summary_df, out_dirs)
    print(summary_df.round(6).to_string(index=False))

    factor_export_cols = ["date", "stock", "fwd_ret"] + eval_cols
    df[factor_export_cols].to_parquet(
        out_dirs["tables"] / "factor_panel.parquet", index=False
    )

    print_section("5. Output Files")
    print(f"Tables  : {out_dirs['tables'].resolve()}")
    print(f"Figures : {out_dirs['figures'].resolve()}")

    return df, summary_df


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Polar price-volume reversal factor research"
    )
    parser.add_argument("--data-path", default="exam_data.parquet")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--fwd-horizon", type=int, default=1)
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--dynamic-lookback", type=int, default=60)
    args = parser.parse_args()

    return Config(
        data_path=args.data_path,
        output_dir=args.output_dir,
        fwd_horizon=args.fwd_horizon,
        quantiles=args.quantiles,
        dynamic_lookback=args.dynamic_lookback,
    )


def main() -> None:
    cfg = parse_args()
    run_research(cfg)


if __name__ == "__main__":
    main()
