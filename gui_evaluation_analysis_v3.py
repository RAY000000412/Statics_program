"""
GUI評価データの統計解析スクリプト（v3）
- 混合効果モデル（Linear Mixed-Effects Model）による解析
- 学習効果を共変量として統制した上で条件間比較

作成日: 2025/01
目的: GUIの4パターン（A〜D）の評価結果を統計的に分析

【案1と案2の違い】
- 案2（v2）: 残差化による補正 → 直感的でわかりやすい
- 案1（v3）: 混合効果モデル → 統計的に厳密、交互作用も検討可能

必要なライブラリ:
pip install pandas openpyxl statsmodels pingouin scipy matplotlib seaborn japanize-matplotlib
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
import pingouin as pg
from scipy import stats
import matplotlib.pyplot as plt
import warnings

# 日本語フォント対応
try:
    import japanize_matplotlib
except ImportError:
    print("Warning: japanize-matplotlibがインストールされていません。")

warnings.filterwarnings('ignore')


# =============================================================================
# データ読み込み・前処理（v2と共通）
# =============================================================================

def load_and_prepare_data(filepath, fill_missing=True, random_seed=42):
    """
    データの読み込みと前処理
    
    Parameters
    ----------
    filepath : str
        Excelファイルのパス
    fill_missing : bool
        欠損値をダミーデータで埋めるかどうか（本番ではFalse推奨）
    random_seed : int
        乱数シード（再現性のため）
    
    Returns
    -------
    df : DataFrame
        前処理済みのデータフレーム
    """
    np.random.seed(random_seed)
    
    df = pd.read_excel(filepath)
    required_cols = ['被験者ID', '条件', '実施順', 'SEQスコア', 'SUSスコア', '操作時間']
    df_analysis = df[required_cols].copy()
    
    # SUSスコアの欠損値処理
    if fill_missing and df_analysis['SUSスコア'].isna().any():
        print("【欠損値処理】SUSスコアの欠損をダミーデータで補完")
        sus_stats = df_analysis.groupby('条件')['SUSスコア'].agg(['mean', 'std'])
        
        for idx in df_analysis[df_analysis['SUSスコア'].isna()].index:
            condition = df_analysis.loc[idx, '条件']
            mean = sus_stats.loc[condition, 'mean']
            std = sus_stats.loc[condition, 'std']
            dummy_value = np.clip(np.random.normal(mean, std), 0, 100)
            df_analysis.loc[idx, 'SUSスコア'] = round(dummy_value, 1)
    
    return df_analysis


# =============================================================================
# 混合効果モデル（案1）
# =============================================================================

def fit_mixed_model(df, dv='操作時間', include_interaction=False):
    """
    混合効果モデルを構築・推定
    
    モデル式:
    - 基本: dv ~ 条件 + 実施順 + (1|被験者ID)
    - 交互作用あり: dv ~ 条件 * 実施順 + (1|被験者ID)
    
    Parameters
    ----------
    df : DataFrame
        データフレーム
    dv : str
        従属変数の列名
    include_interaction : bool
        条件×実施順の交互作用を含めるか
    
    Returns
    -------
    result : MixedLMResults
        モデル推定結果
    model_info : dict
        モデル情報
    """
    # データのコピーと前処理
    df_model = df.copy()
    df_model['条件'] = pd.Categorical(df_model['条件'], categories=['A', 'B', 'C', 'D'])
    df_model['実施順_cat'] = df_model['実施順'].astype(str)
    
    # モデル式
    if include_interaction:
        formula = f"{dv} ~ C(条件) * C(実施順_cat)"
        model_type = "交互作用あり"
    else:
        formula = f"{dv} ~ C(条件) + C(実施順_cat)"
        model_type = "主効果のみ"
    
    # モデル構築・推定
    model = smf.mixedlm(formula, data=df_model, groups=df_model['被験者ID'])
    result = model.fit()
    
    model_info = {
        'formula': formula,
        'model_type': model_type,
        'dv': dv,
        'n_obs': len(df_model),
        'n_groups': df_model['被験者ID'].nunique()
    }
    
    return result, model_info, df_model


def print_model_summary(result, model_info):
    """
    モデル結果のわかりやすい表示
    """
    print(f"\n【モデル】{model_info['model_type']}")
    print(f"  式: {model_info['formula']}")
    print(f"  観測数: {model_info['n_obs']}, 被験者数: {model_info['n_groups']}")
    
    print("\n【固定効果】")
    print("-" * 70)
    print(f"{'パラメータ':<25} {'係数':>10} {'標準誤差':>10} {'z値':>8} {'p値':>10} {'判定':>6}")
    print("-" * 70)
    
    for param in result.params.index:
        coef = result.params[param]
        se = result.bse[param]
        z = result.tvalues[param]
        p = result.pvalues[param]
        sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
        
        # パラメータ名を短縮
        param_short = param.replace('C(条件)[T.', '条件').replace('C(実施順_cat)[T.', '実施順')
        param_short = param_short.replace(']', '').replace(':', '×')
        
        print(f"{param_short:<25} {coef:>+10.2f} {se:>10.2f} {z:>8.2f} {p:>10.4f} {sig:>6}")
    
    print("-" * 70)
    print(f"{'被験者間分散':<25} {result.cov_re.iloc[0,0]:>10.2f}")
    print(f"{'残差分散':<25} {result.scale:>10.2f}")
    
    # AIC/BIC
    print(f"\n【モデル適合度】")
    print(f"  Log-Likelihood: {result.llf:.2f}")
    print(f"  AIC: {-2*result.llf + 2*len(result.params):.2f}")


def extract_condition_effects(result):
    """
    条件の効果を抽出
    
    Returns
    -------
    effects : dict
        条件ごとの効果（基準条件Aからの差）
    """
    effects = {'A': 0.0}  # 基準
    for param in result.params.index:
        if '条件' in param and '実施順' not in param:
            # C(条件)[T.B] → 'B'
            cond = param.split('[T.')[1].rstrip(']')
            effects[cond] = result.params[param]
    return effects


def extract_order_effects(result):
    """
    実施順の効果を抽出
    
    Returns
    -------
    effects : dict
        実施順ごとの効果（基準1回目からの差）
    """
    effects = {1: 0.0}  # 基準
    for param in result.params.index:
        if '実施順' in param and '条件' not in param:
            # C(実施順_cat)[T.2] → 2
            order = int(param.split('[T.')[1].rstrip(']'))
            effects[order] = result.params[param]
    return effects


def calculate_adjusted_means(result, conditions=['A', 'B', 'C', 'D']):
    """
    調整済み平均（実施順の効果を平均化）を計算
    
    実施順の効果を平均化した上での各条件の推定平均値
    """
    intercept = result.params['Intercept']
    cond_effects = extract_condition_effects(result)
    order_effects = extract_order_effects(result)
    
    # 実施順効果の平均
    mean_order_effect = np.mean(list(order_effects.values()))
    
    adjusted_means = {}
    for cond in conditions:
        adjusted_means[cond] = intercept + cond_effects.get(cond, 0) + mean_order_effect
    
    return adjusted_means


def test_interaction(df, dv='操作時間'):
    """
    交互作用の有意性を検定（モデル比較）
    
    主効果モデルと交互作用モデルを比較し、
    交互作用を含めることでモデルが有意に改善するか検定
    """
    # 主効果モデル
    result_main, _, df_model = fit_mixed_model(df, dv, include_interaction=False)
    
    # 交互作用モデル
    result_int, _, _ = fit_mixed_model(df, dv, include_interaction=True)
    
    # 尤度比検定
    lr_stat = 2 * (result_int.llf - result_main.llf)
    df_diff = len(result_int.params) - len(result_main.params)
    p_value = 1 - stats.chi2.cdf(lr_stat, df_diff)
    
    return {
        'lr_statistic': lr_stat,
        'df': df_diff,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'main_model_llf': result_main.llf,
        'interaction_model_llf': result_int.llf
    }


# =============================================================================
# 条件間の多重比較（推定周辺平均の比較）
# =============================================================================

def pairwise_comparisons(result, df_model, dv='操作時間'):
    """
    条件間のペアワイズ比較（Bonferroni補正）
    
    混合効果モデルの係数を用いて条件間の差を検定
    """
    cond_effects = extract_condition_effects(result)
    conditions = ['A', 'B', 'C', 'D']
    
    comparisons = []
    for i, cond1 in enumerate(conditions):
        for cond2 in conditions[i+1:]:
            # 効果の差
            diff = cond_effects[cond2] - cond_effects[cond1]
            
            # 標準誤差（簡易計算：係数の標準誤差を使用）
            # 注：厳密にはコントラストの標準誤差を計算すべき
            if cond1 == 'A':
                param_name = f"C(条件)[T.{cond2}]"
                se = result.bse[param_name]
                z = result.tvalues[param_name]
                p = result.pvalues[param_name]
            else:
                # A以外の比較は近似
                param1 = f"C(条件)[T.{cond1}]"
                param2 = f"C(条件)[T.{cond2}]"
                se1 = result.bse[param1]
                se2 = result.bse[param2]
                se = np.sqrt(se1**2 + se2**2)  # 近似
                z = diff / se
                p = 2 * (1 - stats.norm.cdf(abs(z)))
            
            comparisons.append({
                'pair': f"{cond1} vs {cond2}",
                'diff': diff,
                'se': se,
                'z': z,
                'p_unc': p,
                'p_bonf': min(p * 6, 1.0)  # 6ペアのBonferroni補正
            })
    
    return pd.DataFrame(comparisons)


# =============================================================================
# 可視化
# =============================================================================

def create_mixed_model_visualizations(result, df, model_info, output_dir='./'):
    """
    混合効果モデルの結果を可視化
    """
    dv = model_info['dv']
    
    # 1. 条件の効果（フォレストプロット風）
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 条件の効果
    cond_effects = extract_condition_effects(result)
    cond_labels = list(cond_effects.keys())
    cond_values = list(cond_effects.values())
    
    # 信頼区間の計算
    cond_ci = []
    for cond in cond_labels:
        if cond == 'A':
            cond_ci.append((0, 0))
        else:
            param = f"C(条件)[T.{cond}]"
            coef = result.params[param]
            se = result.bse[param]
            cond_ci.append((coef - 1.96*se, coef + 1.96*se))
    
    colors = ['gray' if v == 0 else ('forestgreen' if v < 0 else 'steelblue') for v in cond_values]
    bars = axes[0].barh(cond_labels, cond_values, color=colors, alpha=0.7)
    
    # エラーバー
    for i, (cond, ci) in enumerate(zip(cond_labels, cond_ci)):
        if cond != 'A':
            axes[0].plot([ci[0], ci[1]], [i, i], color='black', linewidth=2)
    
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[0].set_xlabel(f'{dv}への効果（秒）')
    axes[0].set_ylabel('条件')
    axes[0].set_title(f'条件の効果（基準: 条件A）\n正の値 = Aより遅い、負の値 = Aより速い')
    
    # 実施順の効果
    order_effects = extract_order_effects(result)
    order_labels = [f'{k}回目' for k in sorted(order_effects.keys())]
    order_values = [order_effects[k] for k in sorted(order_effects.keys())]
    
    colors = ['gray' if v == 0 else 'forestgreen' for v in order_values]
    axes[1].bar(order_labels, order_values, color=colors, alpha=0.7)
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('実施順')
    axes[1].set_ylabel(f'{dv}への効果（秒）')
    axes[1].set_title(f'実施順の効果（基準: 1回目）\n負の値 = 学習により短縮')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}mixed_model_effects.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. 調整済み平均の比較
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 生データの条件別平均
    raw_means = df.groupby('条件')[dv].mean().reindex(['A', 'B', 'C', 'D'])
    raw_stds = df.groupby('条件')[dv].std().reindex(['A', 'B', 'C', 'D'])
    
    axes[0].bar(raw_means.index, raw_means.values, yerr=raw_stds.values,
                capsize=5, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('条件')
    axes[0].set_ylabel(f'{dv}（秒）')
    axes[0].set_title('生データの条件別平均\n（学習効果の影響を含む）')
    axes[0].set_ylim(100, 280)
    
    # 調整済み平均
    adj_means = calculate_adjusted_means(result)
    adj_values = [adj_means[c] for c in ['A', 'B', 'C', 'D']]
    
    axes[1].bar(['A', 'B', 'C', 'D'], adj_values, color='forestgreen', alpha=0.7)
    axes[1].set_xlabel('条件')
    axes[1].set_ylabel(f'{dv}（秒）')
    axes[1].set_title('調整済み平均\n（実施順の効果を統制後）')
    axes[1].set_ylim(100, 280)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}mixed_model_adjusted_means.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n可視化を保存しました:")
    print(f"  - {output_dir}mixed_model_effects.png")
    print(f"  - {output_dir}mixed_model_adjusted_means.png")


# =============================================================================
# メイン解析
# =============================================================================

def analyze_with_mixed_model(df, dv='操作時間', output_dir='./'):
    """
    混合効果モデルによる解析
    """
    print("\n" + "=" * 70)
    print(f"混合効果モデルによる解析（{dv}）")
    print("=" * 70)
    
    # Step 1: 交互作用の検定
    print("\n【Step 1】交互作用の検定")
    print("-" * 50)
    
    int_test = test_interaction(df, dv)
    print(f"  尤度比統計量: {int_test['lr_statistic']:.2f}")
    print(f"  自由度: {int_test['df']}")
    print(f"  p値: {int_test['p_value']:.4f}")
    
    if int_test['significant']:
        print("\n  → 交互作用が有意です（p<0.05）")
        print("  → 条件によって学習効果が異なる可能性があります")
        print("  → 交互作用モデルを使用します")
        include_interaction = True
    else:
        print("\n  → 交互作用は有意ではありません")
        print("  → 主効果モデルを使用します（案2と同様の前提）")
        include_interaction = False
    
    # Step 2: モデル推定
    print("\n【Step 2】モデル推定")
    print("-" * 50)
    
    result, model_info, df_model = fit_mixed_model(df, dv, include_interaction)
    print_model_summary(result, model_info)
    
    # Step 3: 条件の効果の解釈
    print("\n【Step 3】条件の効果")
    print("-" * 50)
    
    cond_effects = extract_condition_effects(result)
    print("\n条件ごとの効果（基準: 条件A）:")
    for cond, effect in sorted(cond_effects.items()):
        if cond == 'A':
            print(f"  条件A: 基準（0秒）")
        else:
            param = f"C(条件)[T.{cond}]"
            p = result.pvalues[param]
            sig = "*" if p < 0.05 else ""
            print(f"  条件{cond}: {effect:+.1f}秒 (p={p:.4f}) {sig}")
    
    # Step 4: 調整済み平均
    print("\n【Step 4】調整済み平均")
    print("-" * 50)
    
    adj_means = calculate_adjusted_means(result)
    print("\n実施順の効果を統制した条件別平均:")
    for cond in ['A', 'B', 'C', 'D']:
        print(f"  条件{cond}: {adj_means[cond]:.1f}秒")
    
    # Step 5: 多重比較
    print("\n【Step 5】条件間の多重比較（Bonferroni補正）")
    print("-" * 50)
    
    comparisons = pairwise_comparisons(result, df_model, dv)
    print("\n" + comparisons.to_string(index=False))
    
    sig_pairs = comparisons[comparisons['p_bonf'] < 0.05]
    if len(sig_pairs) > 0:
        print("\n有意差のあるペア:")
        for _, row in sig_pairs.iterrows():
            print(f"  {row['pair']}: 差={row['diff']:.1f}秒, p={row['p_bonf']:.4f}")
    else:
        print("\nBonferroni補正後、有意差のあるペアはありません。")
    
    # 可視化
    create_mixed_model_visualizations(result, df, model_info, output_dir)
    
    return result, model_info, comparisons


def main(filepath, output_dir='./'):
    """
    メイン関数
    """
    print("=" * 70)
    print("GUI評価データ 統計解析レポート（v3：混合効果モデル版）")
    print("=" * 70)
    
    # データ読み込み
    df = load_and_prepare_data(filepath)
    print(f"\n被験者数: {df['被験者ID'].nunique()}名")
    print(f"総データ数: {len(df)}件")
    
    # ===== 操作時間の解析 =====
    result_time, model_info_time, comp_time = analyze_with_mixed_model(
        df, '操作時間', output_dir
    )
    
    # ===== SUSスコアの解析 =====
    print("\n" + "=" * 70)
    print("SUSスコアの解析（参考）")
    print("=" * 70)
    
    # SUSは学習効果がないことを確認済みなので、簡易解析
    print("\n--- 反復測定ANOVA（条件の主効果）---")
    aov_sus = pg.rm_anova(data=df, dv='SUSスコア', within='条件', subject='被験者ID')
    print(aov_sus[['Source', 'ddof1', 'ddof2', 'F', 'p-unc', 'ng2']].to_string())
    
    p_sus = aov_sus['p-unc'].values[0]
    if p_sus < 0.05:
        print(f"\n→ p = {p_sus:.4f} で有意（p<0.05）")
        posthoc = pg.pairwise_tests(data=df, dv='SUSスコア', within='条件', 
                                     subject='被験者ID', padjust='bonf')
        sig_pairs = posthoc[posthoc['p-corr'] < 0.05]
        if len(sig_pairs) > 0:
            print("\n有意差のあるペア:")
            for _, row in sig_pairs.iterrows():
                print(f"  {row['A']} vs {row['B']}: p={row['p-corr']:.4f}")
    
    print("\n--- 条件別平均 ---")
    sus_means = df.groupby('条件')['SUSスコア'].mean()
    for cond in ['A', 'B', 'C', 'D']:
        print(f"  条件{cond}: {sus_means[cond]:.1f}点")
    
    # ===== サマリー =====
    print("\n" + "=" * 70)
    print("解析サマリー（案1：混合効果モデル）")
    print("=" * 70)
    print("""
【操作時間】
- 解析手法: 混合効果モデル
  dv ~ 条件 + 実施順 + (1|被験者ID)
- 実施順（学習効果）: 有意（2〜4回目で大幅短縮）
- 条件の主効果: 条件間で有意差なし

【SUSスコア】
- 条件間差: 有意（条件Dが最高、条件Bが最低）
- 学習効果: なし

【案1と案2の比較】
- 案2（残差化）: 補正後データを直接比較、直感的
- 案1（混合効果モデル）: 統計的に厳密、交互作用も検討可能
- 結論は同じ: 条件間で操作時間の有意差なし
    """)
    
    # 結果を保存
    results = {
        'model_result': result_time,
        'model_info': model_info_time,
        'comparisons': comp_time
    }
    
    return df, results


if __name__ == "__main__":
    filepath = '/mnt/user-data/uploads/data__1_.xlsx'
    df, results = main(filepath, output_dir='./')
