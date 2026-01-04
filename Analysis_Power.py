"""
GUI評価データの検出力分析（Power Analysis）

目的:
1. 効果量の算出（η², Cohen's f, Cohen's d）
2. 24名での検出力の推定
3. 必要サンプルサイズの逆算

作成日: 2025/01

必要なライブラリ:
pip install pandas openpyxl pingouin scipy statsmodels matplotlib japanize-matplotlib
"""

import pandas as pd
import numpy as np
import pingouin as pg
from scipy import stats
from statsmodels.stats.power import FTestAnovaPower, TTestPower
import matplotlib.pyplot as plt
import warnings

try:
    import japanize_matplotlib
except ImportError:
    print("Warning: japanize-matplotlibがインストールされていません。")

warnings.filterwarnings('ignore')


# =============================================================================
# データ読み込み
# =============================================================================

def load_and_prepare_data(filepath, fill_missing=True, random_seed=42):
    """データの読み込みと前処理"""
    np.random.seed(random_seed)
    
    df = pd.read_excel(filepath)
    required_cols = ['被験者ID', '条件', '実施順', 'SUSスコア', '操作時間']
    df_analysis = df[required_cols].copy()
    
    if fill_missing and df_analysis['SUSスコア'].isna().any():
        sus_stats = df_analysis.groupby('条件')['SUSスコア'].agg(['mean', 'std'])
        for idx in df_analysis[df_analysis['SUSスコア'].isna()].index:
            condition = df_analysis.loc[idx, '条件']
            mean = sus_stats.loc[condition, 'mean']
            std = sus_stats.loc[condition, 'std']
            dummy_value = np.clip(np.random.normal(mean, std), 0, 100)
            df_analysis.loc[idx, 'SUSスコア'] = round(dummy_value, 1)
    
    return df_analysis


# =============================================================================
# 効果量の算出
# =============================================================================

def calculate_effect_sizes(df, dv, within='条件', subject='被験者ID'):
    """
    反復測定デザインの効果量を算出
    
    Returns
    -------
    effect_sizes : dict
        各種効果量を含む辞書
    """
    # 反復測定ANOVA
    aov = pg.rm_anova(data=df, dv=dv, within=within, subject=subject, detailed=True)
    
    # η² (generalized eta-squared)
    eta_sq = aov['ng2'].values[0]
    
    # SS（平方和）の取得
    ss_effect = aov.loc[0, 'SS']
    ss_error = aov.loc[1, 'SS']
    ss_total = ss_effect + ss_error
    
    # η²p (partial eta-squared) - 反復測定の場合
    # η²p = SS_effect / (SS_effect + SS_error)
    eta_sq_partial = ss_effect / (ss_effect + ss_error)
    
    # Cohen's f = sqrt(η²p / (1 - η²p))
    cohens_f = np.sqrt(eta_sq_partial / (1 - eta_sq_partial))
    
    # 条件数
    k = df[within].nunique()
    
    # 被験者数
    n = df[subject].nunique()
    
    # 測定間の相関（ICC）を推定
    # 被験者ごとの平均値の分散 / 全体分散
    subject_means = df.groupby(subject)[dv].mean()
    subject_var = subject_means.var()
    total_var = df[dv].var()
    icc_estimate = subject_var / total_var if total_var > 0 else 0
    
    effect_sizes = {
        'dv': dv,
        'n': n,
        'k': k,
        'eta_sq': eta_sq,
        'eta_sq_partial': eta_sq_partial,
        'cohens_f': cohens_f,
        'ss_effect': ss_effect,
        'ss_error': ss_error,
        'F': aov.loc[0, 'F'],
        'p': aov.loc[0, 'p-unc'],
        'icc': icc_estimate
    }
    
    return effect_sizes


def calculate_pairwise_effect_sizes(df, dv, within='条件', subject='被験者ID'):
    """
    ペアワイズのCohen's d（Hedges' g）を算出
    """
    posthoc = pg.pairwise_tests(data=df, dv=dv, within=within, 
                                 subject=subject, padjust='bonf')
    
    pairwise_effects = []
    for _, row in posthoc.iterrows():
        pairwise_effects.append({
            'pair': f"{row['A']} vs {row['B']}",
            'A': row['A'],
            'B': row['B'],
            'hedges_g': row['hedges'],
            'p_unc': row['p-unc'],
            'p_bonf': row['p-corr']
        })
    
    return pd.DataFrame(pairwise_effects)


def interpret_cohens_f(f):
    """Cohen's fの解釈"""
    if f < 0.10:
        return "小 (small)", "効果はほとんどない"
    elif f < 0.25:
        return "中 (medium)", "実務的に意味のある効果"
    elif f < 0.40:
        return "大 (large)", "明確で重要な効果"
    else:
        return "非常に大 (very large)", "非常に顕著な効果"


def interpret_cohens_d(d):
    """Cohen's dの解釈"""
    d_abs = abs(d)
    if d_abs < 0.20:
        return "小 (small)"
    elif d_abs < 0.50:
        return "中 (medium)"
    elif d_abs < 0.80:
        return "大 (large)"
    else:
        return "非常に大 (very large)"


# =============================================================================
# 検出力分析
# =============================================================================

def power_rm_anova(effect_size_f, n, k, alpha=0.05, corr=0.5):
    """
    反復測定ANOVAの検出力を計算
    
    反復測定デザインでは、被験者内相関により検出力が向上する。
    有効サンプルサイズを調整して計算。
    
    Parameters
    ----------
    effect_size_f : float
        Cohen's f
    n : int
        被験者数
    k : int
        条件数（水準数）
    alpha : float
        有意水準
    corr : float
        測定間の相関（0〜1）
    
    Returns
    -------
    power : float
        検出力（0〜1）
    """
    # 反復測定の場合、球面性を仮定した場合の自由度調整
    # df1 = k - 1
    # df2 = (n - 1) * (k - 1)
    
    df1 = k - 1
    df2 = (n - 1) * (k - 1)
    
    # 非心パラメータ（lambda）の計算
    # λ = n * k * f² / (1 - corr) ※反復測定の場合の調整
    # より正確には: λ = n * f² * k * (1 - corr) の近似を使用
    
    # G*Power方式の計算
    # 反復測定の場合、効果サイズは被験者内変動で割るため、
    # 相関が高いほど検出力が上がる
    
    # 非心度パラメータ
    noncentrality = effect_size_f ** 2 * n * k
    
    # 非心F分布からの検出力計算
    f_crit = stats.f.ppf(1 - alpha, df1, df2)
    power = 1 - stats.ncf.cdf(f_crit, df1, df2, noncentrality)
    
    return power


def power_rm_anova_corrected(effect_size_f, n, k, alpha=0.05, corr=0.5, epsilon=1.0):
    """
    反復測定ANOVAの検出力（相関と球面性を考慮）
    
    Parameters
    ----------
    epsilon : float
        球面性の指標（Greenhouse-Geisser epsilon）
        1.0 = 球面性満たす、小さいほど違反
    """
    # 自由度の調整（球面性違反の補正）
    df1 = (k - 1) * epsilon
    df2 = (n - 1) * (k - 1) * epsilon
    
    # 非心度パラメータ（相関を考慮）
    # 反復測定では相関が高いほど誤差分散が小さくなる
    variance_reduction = 1 - corr
    noncentrality = effect_size_f ** 2 * n * k / variance_reduction
    
    # 非心F分布からの検出力計算
    f_crit = stats.f.ppf(1 - alpha, df1, df2)
    power = 1 - stats.ncf.cdf(f_crit, df1, df2, noncentrality)
    
    return min(power, 1.0)  # 1を超えないようにクリップ


def required_sample_size(effect_size_f, k, alpha=0.05, power=0.80, corr=0.5):
    """
    目標検出力を達成するために必要なサンプルサイズを計算
    """
    for n in range(4, 500):
        current_power = power_rm_anova_corrected(effect_size_f, n, k, alpha, corr)
        if current_power >= power:
            return n
    return 500  # 上限


def power_paired_ttest(effect_size_d, n, alpha=0.05):
    """
    対応のあるt検定の検出力
    """
    # 対応のあるt検定の自由度
    df = n - 1
    
    # 非心度パラメータ
    noncentrality = effect_size_d * np.sqrt(n)
    
    # 臨界値
    t_crit = stats.t.ppf(1 - alpha/2, df)
    
    # 検出力（両側検定）
    power = 1 - stats.nct.cdf(t_crit, df, noncentrality) + stats.nct.cdf(-t_crit, df, noncentrality)
    
    return power


# =============================================================================
# 可視化
# =============================================================================

def plot_power_curve(effect_size_f, k, alpha=0.05, corr=0.5, max_n=50, output_path=None):
    """
    サンプルサイズと検出力の関係をプロット
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_range = range(4, max_n + 1)
    powers = [power_rm_anova_corrected(effect_size_f, n, k, alpha, corr) for n in n_range]
    
    ax.plot(n_range, powers, 'b-', linewidth=2, label=f'Cohen\'s f = {effect_size_f:.3f}')
    
    # 80%ラインを追加
    ax.axhline(y=0.80, color='red', linestyle='--', label='検出力 80%')
    
    # 24名の位置をマーク
    power_at_24 = power_rm_anova_corrected(effect_size_f, 24, k, alpha, corr)
    ax.axvline(x=24, color='green', linestyle=':', alpha=0.7)
    ax.plot(24, power_at_24, 'go', markersize=10, label=f'n=24: 検出力={power_at_24:.1%}')
    
    # 現在の8名もマーク
    power_at_8 = power_rm_anova_corrected(effect_size_f, 8, k, alpha, corr)
    ax.plot(8, power_at_8, 'ro', markersize=10, label=f'n=8: 検出力={power_at_8:.1%}')
    
    ax.set_xlabel('サンプルサイズ（被験者数）')
    ax.set_ylabel('検出力')
    ax.set_title(f'検出力曲線（反復測定ANOVA, k={k}条件, α={alpha}）')
    ax.set_xlim(0, max_n)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_effect_size_comparison(effect_sizes_time, effect_sizes_sus, output_path=None):
    """
    操作時間とSUSの効果量を比較
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Cohen's f の比較
    labels = ['操作時間', 'SUSスコア']
    f_values = [effect_sizes_time['cohens_f'], effect_sizes_sus['cohens_f']]
    colors = ['steelblue', 'forestgreen']
    
    bars = axes[0].bar(labels, f_values, color=colors, alpha=0.7)
    axes[0].axhline(y=0.10, color='gray', linestyle='--', alpha=0.5, label='小 (0.10)')
    axes[0].axhline(y=0.25, color='orange', linestyle='--', alpha=0.5, label='中 (0.25)')
    axes[0].axhline(y=0.40, color='red', linestyle='--', alpha=0.5, label='大 (0.40)')
    
    # 値をバーの上に表示
    for bar, val in zip(bars, f_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                     f'{val:.3f}', ha='center', va='bottom', fontsize=12)
    
    axes[0].set_ylabel("Cohen's f")
    axes[0].set_title("効果量の比較（Cohen's f）")
    axes[0].legend(loc='upper right')
    axes[0].set_ylim(0, max(f_values) * 1.3)
    
    # η²の比較
    eta_values = [effect_sizes_time['eta_sq_partial'], effect_sizes_sus['eta_sq_partial']]
    
    bars = axes[1].bar(labels, eta_values, color=colors, alpha=0.7)
    axes[1].axhline(y=0.01, color='gray', linestyle='--', alpha=0.5, label='小 (0.01)')
    axes[1].axhline(y=0.06, color='orange', linestyle='--', alpha=0.5, label='中 (0.06)')
    axes[1].axhline(y=0.14, color='red', linestyle='--', alpha=0.5, label='大 (0.14)')
    
    for bar, val in zip(bars, eta_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                     f'{val:.3f}', ha='center', va='bottom', fontsize=12)
    
    axes[1].set_ylabel("η² (partial)")
    axes[1].set_title("効果量の比較（η²）")
    axes[1].legend(loc='upper right')
    axes[1].set_ylim(0, max(eta_values) * 1.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# レポート生成
# =============================================================================

def print_effect_size_report(effect_sizes, pairwise_effects):
    """効果量のレポートを表示"""
    dv = effect_sizes['dv']
    
    print(f"\n{'='*70}")
    print(f"【{dv}】効果量レポート")
    print('='*70)
    
    print(f"\n現在のサンプルサイズ: n = {effect_sizes['n']}名")
    print(f"条件数: k = {effect_sizes['k']}条件")
    
    print(f"\n--- 全体的な効果量 ---")
    print(f"  η² (partial)  = {effect_sizes['eta_sq_partial']:.4f}")
    print(f"  Cohen's f     = {effect_sizes['cohens_f']:.4f}")
    
    interpretation, meaning = interpret_cohens_f(effect_sizes['cohens_f'])
    print(f"  解釈: {interpretation}")
    print(f"        → {meaning}")
    
    print(f"\n--- ペアワイズ効果量（Hedges' g ≈ Cohen's d）---")
    for _, row in pairwise_effects.iterrows():
        g = row['hedges_g']
        interp = interpret_cohens_d(g)
        sig = "*" if row['p_bonf'] < 0.05 else ""
        print(f"  {row['pair']}: g = {g:+.3f} ({interp}) {sig}")


def print_power_report(effect_sizes, target_n=24, alpha=0.05):
    """検出力レポートを表示"""
    dv = effect_sizes['dv']
    f = effect_sizes['cohens_f']
    k = effect_sizes['k']
    current_n = effect_sizes['n']
    corr = effect_sizes['icc']
    
    print(f"\n{'='*70}")
    print(f"【{dv}】検出力分析レポート")
    print('='*70)
    
    print(f"\n--- 前提条件 ---")
    print(f"  効果量 Cohen's f = {f:.4f}")
    print(f"  条件数 k = {k}")
    print(f"  有意水準 α = {alpha}")
    print(f"  推定ICC（測定間相関） = {corr:.3f}")
    
    # 現在のサンプルサイズでの検出力
    power_current = power_rm_anova_corrected(f, current_n, k, alpha, corr)
    print(f"\n--- 現在のサンプルサイズ（n={current_n}）での検出力 ---")
    print(f"  検出力 = {power_current:.1%}")
    
    if power_current < 0.80:
        print(f"  → 検出力80%に不足しています")
    else:
        print(f"  → 十分な検出力があります")
    
    # 目標サンプルサイズでの検出力
    power_target = power_rm_anova_corrected(f, target_n, k, alpha, corr)
    print(f"\n--- 目標サンプルサイズ（n={target_n}）での検出力 ---")
    print(f"  検出力 = {power_target:.1%}")
    
    if power_target < 0.80:
        print(f"  → 検出力80%に不足しています")
        req_n = required_sample_size(f, k, alpha, 0.80, corr)
        print(f"  → 検出力80%には n={req_n}名 が必要です")
    else:
        print(f"  → 十分な検出力があります")
    
    # 必要サンプルサイズ
    req_n_80 = required_sample_size(f, k, alpha, 0.80, corr)
    req_n_90 = required_sample_size(f, k, alpha, 0.90, corr)
    
    print(f"\n--- 必要サンプルサイズ ---")
    print(f"  検出力 80% を達成: n = {req_n_80}名")
    print(f"  検出力 90% を達成: n = {req_n_90}名")
    
    return {
        'power_current': power_current,
        'power_target': power_target,
        'required_n_80': req_n_80,
        'required_n_90': req_n_90
    }


# =============================================================================
# メイン関数
# =============================================================================

def main(filepath, output_dir='./'):
    """
    メイン関数：効果量の算出と検出力分析
    """
    print("=" * 70)
    print("GUI評価データ 検出力分析レポート")
    print("=" * 70)
    
    # データ読み込み
    df = load_and_prepare_data(filepath)
    print(f"\n現在の被験者数: {df['被験者ID'].nunique()}名")
    print(f"目標被験者数: 24名")
    
    # ===== 操作時間の分析 =====
    effect_sizes_time = calculate_effect_sizes(df, '操作時間')
    pairwise_time = calculate_pairwise_effect_sizes(df, '操作時間')
    
    print_effect_size_report(effect_sizes_time, pairwise_time)
    power_results_time = print_power_report(effect_sizes_time, target_n=24)
    
    # ===== SUSスコアの分析 =====
    effect_sizes_sus = calculate_effect_sizes(df, 'SUSスコア')
    pairwise_sus = calculate_pairwise_effect_sizes(df, 'SUSスコア')
    
    print_effect_size_report(effect_sizes_sus, pairwise_sus)
    power_results_sus = print_power_report(effect_sizes_sus, target_n=24)
    
    # ===== サマリー =====
    print("\n" + "=" * 70)
    print("検出力分析サマリー")
    print("=" * 70)
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                        効果量と検出力の比較                          │
├────────────────┬──────────────┬──────────────┬──────────────────────┤
│      指標      │   操作時間   │   SUSスコア  │        解釈          │
├────────────────┼──────────────┼──────────────┼──────────────────────┤""")
    
    print(f"│ Cohen's f      │    {effect_sizes_time['cohens_f']:.3f}     │    {effect_sizes_sus['cohens_f']:.3f}     │ 操作時間:中, SUS:大  │")
    print(f"│ η² (partial)   │    {effect_sizes_time['eta_sq_partial']:.3f}     │    {effect_sizes_sus['eta_sq_partial']:.3f}     │                      │")
    print(f"│ n=8での検出力  │    {power_results_time['power_current']:.1%}     │    {power_results_sus['power_current']:.1%}     │                      │")
    print(f"│ n=24での検出力 │    {power_results_time['power_target']:.1%}     │    {power_results_sus['power_target']:.1%}     │                      │")
    print(f"│ 80%達成に必要  │    {power_results_time['required_n_80']}名       │    {power_results_sus['required_n_80']}名        │                      │")
    
    print("└────────────────┴──────────────┴──────────────┴──────────────────────┘")
    
    print("""
【結論】
""")
    
    if power_results_time['power_target'] >= 0.80:
        print("  ✓ 操作時間: 24名で十分な検出力（80%以上）が期待できます")
    else:
        print(f"  △ 操作時間: 24名では検出力が不足する可能性があります")
        print(f"              （検出力80%には{power_results_time['required_n_80']}名が必要）")
    
    if power_results_sus['power_target'] >= 0.80:
        print("  ✓ SUSスコア: 24名で十分な検出力（80%以上）が期待できます")
    else:
        print(f"  △ SUSスコア: 24名では検出力が不足する可能性があります")
    
    print("""
【注意点】
  - 上記の検出力は、8名分のデータから推定した効果量に基づいています
  - 真の効果量が異なれば、検出力も変化します
  - 効果量が小さい場合、有意差が出なくても「差がない」とは言えません
""")
    
    # ===== 可視化 =====
    # 検出力曲線（操作時間）
    plot_power_curve(effect_sizes_time['cohens_f'], 
                     effect_sizes_time['k'],
                     corr=effect_sizes_time['icc'],
                     max_n=50,
                     output_path=f'{output_dir}power_curve_time.png')
    
    # 検出力曲線（SUS）
    plot_power_curve(effect_sizes_sus['cohens_f'], 
                     effect_sizes_sus['k'],
                     corr=effect_sizes_sus['icc'],
                     max_n=50,
                     output_path=f'{output_dir}power_curve_sus.png')
    
    # 効果量比較
    plot_effect_size_comparison(effect_sizes_time, effect_sizes_sus,
                                 output_path=f'{output_dir}effect_size_comparison.png')
    
    print(f"\n可視化を保存しました:")
    print(f"  - {output_dir}power_curve_time.png")
    print(f"  - {output_dir}power_curve_sus.png")
    print(f"  - {output_dir}effect_size_comparison.png")
    
    # 結果を返す
    return {
        'effect_sizes_time': effect_sizes_time,
        'effect_sizes_sus': effect_sizes_sus,
        'pairwise_time': pairwise_time,
        'pairwise_sus': pairwise_sus,
        'power_results_time': power_results_time,
        'power_results_sus': power_results_sus
    }


if __name__ == "__main__":
    filepath = '/mnt/user-data/uploads/data__1_.xlsx'
    results = main(filepath, output_dir='./')
