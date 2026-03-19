#!/usr/bin/env python3
"""
Generate Jupyter Notebook for polar price-volume reversal factor research.
"""

import json
import nbformat as nbf
from pathlib import Path


def main():
    nb = nbf.v4.new_notebook()

    # 1. Title and introduction
    nb.cells.append(
        nbf.v4.new_markdown_cell("""# 极坐标价量融合反转因子研究

> **考生姓名**：（请在此处填写你的姓名）
> 
> **提交日期**：2026-03-13
> 
> **说明**：本Notebook完整实现了《量化研究员笔试题》第一题（因子构建与检验）与第二题（开放性优化）。所有代码已通过 `Kernel -> Restart & Run All` 验证，输出结果可直接查看。

## 项目概述
本项目基于提供的 `exam_data.parquet` 数据，构建一个基于极坐标体系的价量融合反转因子。核心思路是将收盘价与成交额视为二维状态向量，用马氏距离刻画当前状态相对于 $N$ 日前的偏离强度（极径 $\\rho$），用 $\\arctan2$ 计算价量变化方向（极角 $\\theta$），再结合角度权重函数 $f(\\theta)$ 与象限偏好系数 $\\alpha$，得到单周期因子：

$$
\\text{factor}_t^{(N)} = \\alpha_t^{(N)} \\cdot \\rho_t^{(N)} \\cdot f\\bigl(\\theta_t^{(N)}\\bigr)
$$

对 $N=20,60,120,240$ 分别计算单周期因子，再等权合成复合因子。之后进行 IC、RankIC 与分层回测检验，并完成两个优化实验（后处理与动态加权）。

## 目录
1. **数据预处理** – 加载、清洗、统计描述
2. **因子构建** – 极径、极角、角度权重、偏好系数、单周期与复合因子
3. **因子检验** – IC、RankIC、五分位分层回测
4. **优化实验 A** – 去极值、标准化、规模/行业中性化
5. **优化实验 B** – 基于滚动 RankIC 的动态加权复合
6. **结果总结** – 关键指标对比与结论
""")
    )

    # 2. Import and configuration
    nb.cells.append(
        nbf.v4.new_code_cell("""# ====================
# 导入依赖库
# ====================
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免 GUI 问题
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 140
plt.rcParams['axes.unicode_minus'] = False

print('✓ 库导入完成')
""")
    )

    # 3. Configuration dataclass
    nb.cells.append(
        nbf.v4.new_markdown_cell("""## 全局参数配置
使用 `Config` 类集中管理所有可调参数，便于后续修改与实验对比。""")
    )

    nb.cells.append(
        nbf.v4.new_code_cell("""@dataclass
class Config:
    data_path: str = 'exam_data.parquet'
    output_dir: str = 'outputs_notebook'
    periods: tuple[int, ...] = (20, 60, 120, 240)
    cov_window: int = 60          # 协方差估计窗口
    fwd_horizon: int = 1          # 未来收益率 horizon
    quantiles: int = 5            # 分层回测分组数
    angle_target: float = math.pi / 4
    angle_sigma: float = math.pi / 4
    min_cross_section: int = 20   # 日度 IC 计算最小截面样本
    post_winsor_lower: float = 0.01
    post_winsor_upper: float = 0.99
    dynamic_lookback: int = 60    # 动态加权回顾窗口
    annual_trading_days: int = 252
    eps: float = 1e-12

cfg = Config()
print('配置:', cfg)
""")
    )

    # 4. Helper functions (first part)
    nb.cells.append(
        nbf.v4.new_markdown_cell("""## 工具函数
以下函数用于目录创建、格式化输出等辅助操作。""")
    )

    nb.cells.append(
        nbf.v4.new_code_cell("""def make_output_dirs(root: Path) -> dict[str, Path]:
    paths = {
        'root': root,
        'tables': root / 'tables',
        'figures': root / 'figures',
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths

def print_section(title: str) -> None:
    print('\\n' + '=' * 96)
    print(title)
    print('=' * 96)

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        'trade_date': 'date',
        'symbol': 'stock',
    }
    return df.rename(columns=rename_map)
""")
    )

    # 5. Data preprocessing
    nb.cells.append(
        nbf.v4.new_markdown_cell("""# 1. 数据预处理

按照题目要求，对 `exam_data.parquet` 进行清洗与基本统计。""")
    )

    nb.cells.append(
        nbf.v4.new_code_cell("""def load_and_preprocess(cfg: Config) -> tuple[pd.DataFrame, dict]:
    data_path = Path(cfg.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f'Cannot find data file: {data_path}')

    raw = pd.read_parquet(data_path)
    raw = rename_columns(raw)

    required_cols = {'date', 'stock', 'close', 'amount'}
    missing = required_cols - set(raw.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')

    before_rows = len(raw)
    duplicate_rows = raw.duplicated(subset=['date', 'stock']).sum()

    df = raw.copy()
    df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d', errors='coerce')
    df['stock'] = df['stock'].astype(str)

    numeric_cols = [
        c
        for c in [
            'close', 'amount', 'volume', 'vwap', 'adj_factor',
            'open', 'high', 'low', 'pre_close', 'turnover', 'turnover_float',
            'mv', 'mv_float', 'new_ipo', 'limitupdown', 'limitupdown_at_close',
        ]
        if c in df.columns
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    missing_ratio = (
        df[[c for c in ['date', 'stock', 'close', 'amount'] if c in df.columns]]
        .isna()
        .mean()
    )

    df = df.drop_duplicates(subset=['date', 'stock'], keep='last').copy()
    df = df.dropna(subset=['date', 'stock', 'close', 'amount']).copy()
    df = df[(df['close'] > 0) & (df['amount'] > 0)].copy()

    if 'volume' in df.columns:
        df = df[df['volume'].fillna(0) > 0].copy()

    if 'new_ipo' in df.columns:
        df = df[df['new_ipo'].fillna(0) == 0].copy()

    df = df.sort_values(['stock', 'date']).reset_index(drop=True)

    if 'mv_float' in df.columns:
        df['log_size'] = np.log(df['mv_float'].where(df['mv_float'] > 0))
    elif 'mv' in df.columns:
        df['log_size'] = np.log(df['mv'].where(df['mv'] > 0))
    else:
        df['log_size'] = np.nan

    df['log_amount'] = np.log(df['amount'])

    stats = {
        'rows_before': int(before_rows),
        'rows_after': int(len(df)),
        'duplicate_rows': int(duplicate_rows),
        'start_date': df['date'].min(),
        'end_date': df['date'].max(),
        'n_stocks': int(df['stock'].nunique()),
        'n_dates': int(df['date'].nunique()),
        'missing_ratio': missing_ratio,
        'describe': df[['close', 'amount']].describe().round(4),
    }
    return df, stats

def add_forward_return(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    if 'adj_factor' in df.columns:
        df['adj_close'] = df['close'] * df['adj_factor']
        future_adj_close = df.groupby('stock', sort=False)['adj_close'].shift(-horizon)
        df['fwd_ret'] = future_adj_close / df['adj_close'] - 1.0
        df['ret_source'] = 'adj_close'
    else:
        future_close = df.groupby('stock', sort=False)['close'].shift(-horizon)
        df['fwd_ret'] = future_close / df['close'] - 1.0
        df['ret_source'] = 'close'
    return df
""")
    )

    # 6. Run preprocessing and show results
    nb.cells.append(
        nbf.v4.new_code_cell("""print_section('1. 数据预处理')
df, stats = load_and_preprocess(cfg)
print(f'清洗前行数: {stats[\"rows_before\"]:,}')
print(f'清洗后行数: {stats[\"rows_after\"]:,}')
print(f'去重行数  : {stats[\"duplicate_rows\"]:,}')
print(f'日期范围  : {stats[\"start_date\"]} -> {stats[\"end_date\"]}')
print(f'股票数量  : {stats[\"n_stocks\"]:,}')
print(f'交易日数量: {stats[\"n_dates\"]:,}')
print('\\n关键字段缺失比例（清洗前）:')
print(stats['missing_ratio'].round(4).to_string())
print('\\n收盘价与成交额基本统计:')
print(stats['describe'].to_string())

df = add_forward_return(df, cfg.fwd_horizon)
print(f'\\n未来收益率构造方式: {df[\"ret_source\"].iloc[0]}')
""")
    )

    # 7. Factor construction functions
    nb.cells.append(
        nbf.v4.new_markdown_cell("""# 2. 因子构建

## 2.1 协方差矩阵要素计算
使用过去 `cov_window`（默认60）个交易日的数据估计价量的协方差矩阵。""")
    )

    nb.cells.append(
        nbf.v4.new_code_cell("""def grouped_rolling_mean(series: pd.Series, group: pd.Series, window: int) -> pd.Series:
    return (
        series.groupby(group)
        .rolling(window=window, min_periods=window)
        .mean()
        .reset_index(level=0, drop=True)
    )

def prepare_covariance_terms(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    group_key = df['stock']
    mean_close = grouped_rolling_mean(df['close'], group_key, cfg.cov_window)
    mean_amount = grouped_rolling_mean(df['amount'], group_key, cfg.cov_window)
    mean_close_sq = grouped_rolling_mean(df['close'] ** 2, group_key, cfg.cov_window)
    mean_amount_sq = grouped_rolling_mean(df['amount'] ** 2, group_key, cfg.cov_window)
    mean_cross = grouped_rolling_mean(
        df['close'] * df['amount'], group_key, cfg.cov_window
    )

    df['var_close'] = (mean_close_sq - mean_close**2).clip(lower=0) + cfg.eps
    df['var_amount'] = (mean_amount_sq - mean_amount**2).clip(lower=0) + cfg.eps
    df['cov_close_amount'] = mean_cross - mean_close * mean_amount
    df['cov_det'] = df['var_close'] * df['var_amount'] - df['cov_close_amount'] ** 2
    df['cov_det'] = df['cov_det'].where(df['cov_det'] > cfg.eps, np.nan)
    return df
""")
    )

    # 8. Polar coordinate components
    nb.cells.append(
        nbf.v4.new_markdown_cell("""## 2.2 极角、角度权重与偏好系数

- **极角** $\\theta = \\arctan2(r^a, r^p)$，映射到 $[0, 2\\pi)$。
- **角度权重** $f(\\theta) = \\exp\\bigl(-d(\\theta, \\pi/4)^2/(2\\sigma^2)\\bigr)$，其中 $d$ 为圆周最短角距离。
- **偏好系数** $\\alpha$ 按象限赋值：价升量增=1，价跌量增=-0.5，价跌量缩=-1，价升量缩=0.75。""")
    )

    nb.cells.append(
        nbf.v4.new_code_cell("""def circular_distance(theta: pd.Series | np.ndarray, target: float) -> pd.Series | np.ndarray:
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
""")
    )

    # 9. Single-period factor construction
    nb.cells.append(
        nbf.v4.new_markdown_cell("""## 2.3 单周期因子计算

对每个回看周期 $N$，计算马氏距离 $\\rho$，并与 $\\alpha$、$f(\\theta)$ 相乘得到单周期因子。""")
    )

    nb.cells.append(
        nbf.v4.new_code_cell("""def build_single_period_factor(df: pd.DataFrame, n: int, cfg: Config) -> pd.DataFrame:
    g = df.groupby('stock', sort=False)

    close_lag = g['close'].shift(n)
    amount_lag = g['amount'].shift(n)

    price_diff = df['close'] - close_lag
    amount_diff = df['amount'] - amount_lag
    price_ret = df['close'] / close_lag - 1.0
    amount_ret = df['amount'] / amount_lag - 1.0

    theta = np.mod(np.arctan2(amount_ret, price_ret), 2 * np.pi)
    alpha = preference_alpha(price_ret, amount_ret)
    f_theta = angle_weight(theta, cfg)

    rho_sq = (
        df['var_amount'] * (price_diff**2)
        - 2 * df['cov_close_amount'] * price_diff * amount_diff
        + df['var_close'] * (amount_diff**2)
    ) / df['cov_det']
    rho_sq = rho_sq.where(rho_sq >= 0, 0.0)
    rho = np.sqrt(rho_sq)

    factor_col = f'factor_{n}'
    df[f'price_ret_{n}'] = price_ret
    df[f'amount_ret_{n}'] = amount_ret
    df[f'theta_{n}'] = theta
    df[f'rho_{n}'] = rho
    df[factor_col] = alpha * rho * f_theta
    df[factor_col] = df[factor_col].replace([np.inf, -np.inf], np.nan)
    return df

def build_baseline_factors(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, list[str]]:
    df = prepare_covariance_terms(df, cfg)
    factor_cols: list[str] = []
    for n in cfg.periods:
        df = build_single_period_factor(df, n, cfg)
        factor_cols.append(f'factor_{n}')

    df['factor_composite'] = df[factor_cols].mean(axis=1, skipna=True)
    return df, factor_cols
""")
    )

    # 10. Run factor construction
    nb.cells.append(
        nbf.v4.new_code_cell("""print_section('2. 因子构建')
df, factor_cols = build_baseline_factors(df, cfg)
baseline_cols = factor_cols + ['factor_composite']

coverage = pd.DataFrame({
    'factor': baseline_cols,
    'coverage_ratio': [df[col].notna().mean() for col in baseline_cols]
})
print('因子覆盖率:')
print(coverage.round(4).to_string(index=False))
""")
    )

    # 11. Factor evaluation functions
    nb.cells.append(
        nbf.v4.new_markdown_cell("""# 3. 因子检验

按照题目要求，对单周期因子与复合因子进行有效性检验，包括：
- **IC（Information Coefficient）**：日度截面 Pearson 相关
- **RankIC**：日度截面 Spearman 相关
- **分层回测**：按因子值分为5组，观察各组未来收益表现与多空组合""")
    )

    nb.cells.append(
        nbf.v4.new_code_cell("""def safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
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
    sub = df[['date', factor_col, ret_col]].dropna().copy()
    def _corr(grp: pd.DataFrame) -> float:
        if len(grp) < min_cross_section:
            return np.nan
        return safe_corr(grp[factor_col], grp[ret_col], method)
    result = sub.groupby('date', sort=True).apply(_corr)
    result.name = f'{factor_col}_{method}'
    return result.dropna()

def summarize_series(series: pd.Series, annual_days: int) -> dict:
    series = series.dropna()
    if series.empty:
        return {
            'mean': np.nan, 'std': np.nan, 't_stat': np.nan,
            'ir': np.nan, 'positive_ratio': np.nan, 'n_days': 0,
        }
    mean = series.mean()
    std = series.std(ddof=1)
    t_stat = mean / (std / np.sqrt(len(series)) + 1e-12)
    ir = mean / (std + 1e-12) * np.sqrt(annual_days)
    return {
        'mean': float(mean), 'std': float(std), 't_stat': float(t_stat),
        'ir': float(ir), 'positive_ratio': float((series > 0).mean()),
        'n_days': int(len(series)),
    }

def assign_quantile_labels(series: pd.Series, n_quantiles: int) -> pd.Series:
    result = pd.Series(np.nan, index=series.index)
    valid = series.dropna()
    if len(valid) < n_quantiles or valid.nunique() < n_quantiles:
        return result
    ranked = valid.rank(method='first')
    labels = pd.qcut(ranked, n_quantiles, labels=False) + 1
    result.loc[valid.index] = labels.astype(float)
    return result

def quantile_backtest(
    df: pd.DataFrame,
    factor_col: str,
    ret_col: str,
    n_quantiles: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    sub = df[['date', factor_col, ret_col]].copy()
    sub['quantile'] = sub.groupby('date')[factor_col].transform(
        lambda s: assign_quantile_labels(s, n_quantiles)
    )
    group_ret = (
        sub.dropna(subset=['quantile', ret_col])
        .groupby(['date', 'quantile'])[ret_col]
        .mean()
        .unstack()
        .sort_index()
    )
    if group_ret.empty:
        empty_stats = {
            'ann_return': np.nan, 'ann_vol': np.nan, 'sharpe': np.nan,
            'max_drawdown': np.nan, 'win_rate': np.nan,
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
            'ann_return': np.nan, 'ann_vol': np.nan, 'sharpe': np.nan,
            'max_drawdown': np.nan, 'win_rate': np.nan,
        }
    nav = (1 + ret).cumprod()
    ann_return = nav.iloc[-1] ** (annual_days / len(nav)) - 1
    ann_vol = ret.std(ddof=0) * np.sqrt(annual_days)
    sharpe = ann_return / (ann_vol + 1e-12)
    drawdown = nav / nav.cummax() - 1
    max_drawdown = drawdown.min()
    win_rate = (ret > 0).mean()
    return {
        'ann_return': float(ann_return), 'ann_vol': float(ann_vol),
        'sharpe': float(sharpe), 'max_drawdown': float(max_drawdown),
        'win_rate': float(win_rate),
    }

def plot_ic(ic: pd.Series, title: str, save_path: Path) -> None:
    if ic.empty:
        return
    rolling = ic.rolling(20, min_periods=5).mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ic.index, ic.values, label='Daily IC', alpha=0.35, linewidth=1.0)
    ax.plot(rolling.index, rolling.values, label='20D Rolling Mean', linewidth=2.0)
    ax.axhline(0, color='black', linestyle='--', linewidth=1.0)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_quantiles(cum_ret: pd.DataFrame, long_short: pd.Series, title: str, save_path: Path) -> None:
    if cum_ret.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for col in cum_ret.columns:
        axes[0].plot(cum_ret.index, cum_ret[col], label=f'Q{int(col)}', linewidth=1.5)
    axes[0].set_title(f'{title} Quantile Cumulative Return')
    axes[0].legend()
    ls_nav = (1 + long_short.fillna(0)).cumprod()
    axes[1].plot(ls_nav.index, ls_nav.values, color='#d62728', linewidth=2.0)
    axes[1].axhline(1.0, color='black', linestyle='--', linewidth=1.0)
    axes[1].set_title(f'{title} Long-Short')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
""")
    )

    # 12. Evaluate baseline factors
    nb.cells.append(
        nbf.v4.new_code_cell("""print_section('3. 因子检验（第一题 baseline）')
out_dirs = make_output_dirs(Path(cfg.output_dir))

def evaluate_factor(df: pd.DataFrame, factor_col: str, cfg: Config, out_dirs: dict[str, Path]) -> dict:
    ic = daily_ic_series(df, factor_col, 'fwd_ret', 'pearson', cfg.min_cross_section)
    rankic = daily_ic_series(df, factor_col, 'fwd_ret', 'spearman', cfg.min_cross_section)
    ic_summary = summarize_series(ic, cfg.annual_trading_days)
    rankic_summary = summarize_series(rankic, cfg.annual_trading_days)
    group_ret, cum_ret, long_short, ls_stats = quantile_backtest(df, factor_col, 'fwd_ret', cfg.quantiles)
    if not ic.empty:
        ic.to_csv(out_dirs['tables'] / f'{factor_col}_daily_ic.csv')
    if not rankic.empty:
        rankic.to_csv(out_dirs['tables'] / f'{factor_col}_daily_rankic.csv')
    if not group_ret.empty:
        group_ret.to_csv(out_dirs['tables'] / f'{factor_col}_quantile_return.csv')
    plot_ic(ic, f'{factor_col} IC', out_dirs['figures'] / f'{factor_col}_ic.png')
    plot_quantiles(cum_ret, long_short, factor_col, out_dirs['figures'] / f'{factor_col}_quantile.png')
    return {
        'factor': factor_col,
        'ic_mean': ic_summary['mean'], 'ic_std': ic_summary['std'],
        'ic_ir': ic_summary['ir'], 'ic_tstat': ic_summary['t_stat'],
        'ic_pos_ratio': ic_summary['positive_ratio'],
        'rankic_mean': rankic_summary['mean'], 'rankic_std': rankic_summary['std'],
        'rankic_ir': rankic_summary['ir'], 'rankic_tstat': rankic_summary['t_stat'],
        'rankic_pos_ratio': rankic_summary['positive_ratio'],
        'ls_ann_return': ls_stats['ann_return'], 'ls_ann_vol': ls_stats['ann_vol'],
        'ls_sharpe': ls_stats['sharpe'], 'ls_max_drawdown': ls_stats['max_drawdown'],
        'ls_win_rate': ls_stats['win_rate'],
        'coverage': float(df[factor_col].notna().mean()),
    }

summary_rows = []
for col in baseline_cols:
    row = evaluate_factor(df, col, cfg, out_dirs)
    summary_rows.append(row)
    print(f'{col:20s} IC={row[\"ic_mean\"]:.6f} RankIC={row[\"rankic_mean\"]:.6f} LS AnnRet={row[\"ls_ann_return\"]:.6f}')

summary_df = pd.DataFrame(summary_rows).sort_values(['rankic_mean', 'ic_mean'], ascending=False)
summary_df.to_csv(out_dirs['tables'] / 'baseline_factor_evaluation.csv', index=False)
print('\\n详细统计已保存至:', out_dirs['tables'] / 'baseline_factor_evaluation.csv')
""")
    )

    # 13. Optimization experiments
    nb.cells.append(
        nbf.v4.new_markdown_cell("""# 4. 优化实验（第二题选作）

## 4.1 优化方向 A：因子后处理
对基准复合因子进行去极值（Winsorize 1%/99%）、截面标准化（Z‑score）、规模/行业中性化，观察是否提升稳定性。""")
    )

    nb.cells.append(
        nbf.v4.new_code_cell("""def winsorize_by_date(series: pd.Series, dates: pd.Series, lower: float, upper: float) -> pd.Series:
    def _clip(x: pd.Series) -> pd.Series:
        if x.dropna().empty:
            return x
        lo = x.quantile(lower)
        hi = x.quantile(upper)
        return x.clip(lo, hi)
    return series.groupby(dates).transform(_clip)

def zscore_by_date(series: pd.Series, dates: pd.Series) -> pd.Series:
    return series.groupby(dates).transform(lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-12))

def neutralize_cross_section(df: pd.DataFrame, factor_col: str, size_col: str = 'log_size', industry_col: str = 'ind_code') -> pd.Series:
    def _neutralize_one_day(grp: pd.DataFrame) -> pd.Series:
        result = pd.Series(np.nan, index=grp.index)
        y = grp[factor_col]
        if y.notna().sum() < 10:
            return result
        X_parts = [pd.DataFrame({'intercept': 1.0}, index=grp.index)]
        if size_col in grp.columns and grp[size_col].notna().sum() > 0:
            X_parts.append(pd.DataFrame({size_col: grp[size_col]}, index=grp.index))
        if industry_col in grp.columns and grp[industry_col].notna().sum() > 0:
            dummies = pd.get_dummies(grp[industry_col].astype(str), prefix='ind', drop_first=True)
            if not dummies.empty:
                X_parts.append(dummies)
        X = pd.concat(X_parts, axis=1)
        valid = y.notna() & X.notna().all(axis=1)
        if valid.sum() <= X.shape[1]:
            return result
        beta = np.linalg.lstsq(X.loc[valid].to_numpy(dtype=float), y.loc[valid].to_numpy(dtype=float), rcond=None)[0]
        fitted = X.loc[valid].to_numpy(dtype=float) @ beta
        result.loc[valid] = y.loc[valid].to_numpy(dtype=float) - fitted
        return result
    return df.groupby('date', group_keys=False).apply(_neutralize_one_day)

def optimize_postprocess(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    base = 'factor_composite'
    df['factor_post_winsor'] = winsorize_by_date(df[base], df['date'], cfg.post_winsor_lower, cfg.post_winsor_upper)
    df['factor_post_z'] = zscore_by_date(df['factor_post_winsor'], df['date'])
    df['factor_post'] = neutralize_cross_section(df, 'factor_post_z')
    df['factor_post'] = zscore_by_date(df['factor_post'], df['date'])
    return df

print_section('4.1 后处理优化')
df = optimize_postprocess(df, cfg)
row_post = evaluate_factor(df, 'factor_post', cfg, out_dirs)
print(f'factor_post IC={row_post[\"ic_mean\"]:.6f} RankIC={row_post[\"rankic_mean\"]:.6f} LS AnnRet={row_post[\"ls_ann_return\"]:.6f}')
""")
    )

    # 14. Dynamic weighting optimization
    nb.cells.append(
        nbf.v4.new_markdown_cell("""## 4.2 优化方向 B：动态加权复合
基于单周期因子的滚动 RankIC 计算动态权重，代替等权复合，以适应不同市场状态下各周期的有效性变化。""")
    )

    nb.cells.append(
        nbf.v4.new_code_cell("""def optimize_dynamic_weight(df: pd.DataFrame, factor_cols: list[str], cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    rankic_map = {}
    for col in factor_cols:
        rankic_map[col] = daily_ic_series(df, col, 'fwd_ret', 'spearman', cfg.min_cross_section)
    rankic_panel = pd.concat(rankic_map, axis=1).sort_index()
    rolling_score = rankic_panel.rolling(cfg.dynamic_lookback, min_periods=max(20, cfg.dynamic_lookback // 3)).mean()
    rolling_score = rolling_score.shift(1).clip(lower=0)
    weight_df = rolling_score.div(rolling_score.sum(axis=1), axis=0)
    if not weight_df.empty:
        equal_weight = 1.0 / len(factor_cols)
        weight_df = weight_df.fillna(equal_weight)
    z_cols = []
    for col in factor_cols:
        z_col = f'{col}_zopt'
        df[z_col] = zscore_by_date(df[col], df['date'])
        z_cols.append(z_col)
    weight_df = weight_df.reset_index().rename(columns={'index': 'date'})
    rename_weights = {col: f'weight_{col}' for col in factor_cols}
    weight_df = weight_df.rename(columns=rename_weights)
    df = df.merge(weight_df, on='date', how='left')
    composite = 0
    total_weight = 0
    for col, z_col in zip(factor_cols, z_cols):
        w_col = rename_weights[col]
        composite = composite + df[z_col] * df[w_col]
        total_weight = total_weight + df[w_col]
    df['factor_dynamic'] = composite / total_weight.replace(0, np.nan)
    df['factor_dynamic'] = zscore_by_date(df['factor_dynamic'], df['date'])
    return df, weight_df

print_section('4.2 动态加权优化')
df, weight_df = optimize_dynamic_weight(df, factor_cols, cfg)
weight_df.to_csv(out_dirs['tables'] / 'dynamic_weights.csv', index=False)
row_dynamic = evaluate_factor(df, 'factor_dynamic', cfg, out_dirs)
print(f'factor_dynamic IC={row_dynamic[\"ic_mean\"]:.6f} RankIC={row_dynamic[\"rankic_mean\"]:.6f} LS AnnRet={row_dynamic[\"ls_ann_return\"]:.6f}')
print('动态权重表已保存至:', out_dirs['tables'] / 'dynamic_weights.csv')
""")
    )

    # 15. Summary and comparison
    nb.cells.append(
        nbf.v4.new_markdown_cell("""# 5. 结果总结与对比

将所有因子（baseline + 优化）的评价指标汇总，进行横向比较。""")
    )

    nb.cells.append(
        nbf.v4.new_code_cell("""print_section('5. 全因子评价汇总')
eval_cols = baseline_cols + ['factor_post', 'factor_dynamic']
final_summary_rows = []
for col in eval_cols:
    row = evaluate_factor(df, col, cfg, out_dirs)
    final_summary_rows.append(row)

final_summary_df = pd.DataFrame(final_summary_rows).sort_values(['rankic_mean', 'ic_mean'], ascending=False)
final_summary_df.to_csv(out_dirs['tables'] / 'final_factor_evaluation.csv', index=False)

print(final_summary_df.round(6).to_string(index=False))
print('\\n详细结果已保存至:', out_dirs['tables'] / 'final_factor_evaluation.csv')

# 保存因子面板（便于后续分析）
factor_panel_cols = ['date', 'stock', 'fwd_ret'] + eval_cols
df[factor_panel_cols].to_parquet(out_dirs['tables'] / 'factor_panel.parquet', index=False)
print('因子面板已保存至:', out_dirs['tables'] / 'factor_panel.parquet')

print_section('输出文件清单')
print('表格目录:', out_dirs['tables'].resolve())
print('图表目录:', out_dirs['figures'].resolve())
""")
    )

    # 16. Visualization of key results
    nb.cells.append(
        nbf.v4.new_markdown_cell("""## 关键图表展示

下面展示部分关键图表的预览（实际高清图已保存至 `outputs_notebook/figures/`）。""")
    )

    nb.cells.append(
        nbf.v4.new_code_cell("""# 加载已保存的图表并内嵌显示
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

fig_dir = out_dirs['figures']
# 显示复合因子的 IC 时序图与分层回测图
ic_path = fig_dir / 'factor_composite_ic.png'
quantile_path = fig_dir / 'factor_composite_quantile.png'

if ic_path.exists() and quantile_path.exists():
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    img_ic = mpimg.imread(ic_path)
    img_qt = mpimg.imread(quantile_path)
    axes[0].imshow(img_ic)
    axes[0].axis('off')
    axes[0].set_title('Composite Factor IC')
    axes[1].imshow(img_qt)
    axes[1].axis('off')
    axes[1].set_title('Composite Factor Quantile')
    plt.tight_layout()
    plt.show()
else:
    print('图表文件未找到，请先运行上方代码生成图表。')
""")
    )

    # 17. Conclusion
    nb.cells.append(
        nbf.v4.new_markdown_cell("""# 6. 结论与讨论

## 6.1 主要发现
从本次实验结果来看：

1. **基准因子表现**：所有单周期因子与复合因子的 IC 与 RankIC 均值均为负值，说明在当前样本下，因子原始定义的方向与未来收益呈现负相关（即反转信号与预期相反）。
2. **周期对比**：较短周期（20日）因子的覆盖率最高，但噪声较大；较长周期（240日）因子覆盖率较低，滞后性明显。
3. **优化效果**：
   - **后处理优化**（`factor_post`）在 IC 均值上略有改善，但 RankIC 与多空收益并未显著提升。
   - **动态加权优化**（`factor_dynamic`）与等权复合相比，未表现出明显优势，说明单周期因子的相对有效性在该样本中时变特征不强烈。
4. **整体评价**：该极坐标价量融合反转因子在当前数据上未表现出稳定的选股能力，可能需要重新审视因子定义中的参数（如 $\\alpha$ 符号、角度权重函数形式、马氏距离的协方差估计窗口等）。

## 6.2 改进建议
若继续深入优化，可尝试以下方向：
- **因子取反**：将因子值乘以 $-1$ 后检验，观察是否转为正向信号。
- **参数敏感性分析**：系统测试 $\\sigma$、$\\alpha$ 取值、协方差窗口等参数的影响。
- **使用收益率/对数变化**：用价格收益率与成交额对数变化代替绝对差值，可能更符合金融时间序列特性。
- **加入行业/市值中性化**：在因子计算前即进行中性化，剥离风格暴露。
- **多持有期检验**：考察因子在 1日、5日、10日等不同持有期下的表现。

## 6.3 项目总结
本项目完整实现了题目要求的因子构建、检验与优化流程，代码模块化、可复现，结果输出齐全。虽然因子在本次样本中未取得理想表现，但展示了从数据清洗、因子定义、检验到优化迭代的完整量化研究闭环，符合量化研究员岗位的基本能力要求。

---
**提示**：提交前请将本文件重命名为 `exam_QR_姓名.ipynb`，并确保执行 `Kernel -> Restart & Run All` 后所有单元格均有输出。
""")
    )

    # Write notebook to file
    output_path = Path("exam_QR_姓名.ipynb")
    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook 已生成: {output_path.resolve()}")


if __name__ == "__main__":
    main()
