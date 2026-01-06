"""
GUI評価データの統計解析スクリプト（v2 Dynamic）
- 反復測定ANOVA
- 学習効果の補正（残差化による補正：案2）
- 解析結果に基づく動的サマリー生成

作成日: 2025/01
目的: GUIの4パターン（A〜D）の評価結果を統計的に分析

必要なライブラリ:
pip install pandas openpyxl pingouin scipy matplotlib seaborn japanize-matplotlib
"""

import pandas as pd
import numpy as np
import pingouin as pg
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 日本語フォント対応
try:
    import japanize_matplotlib
except ImportError:
    print("Warning: japanize-matplotlibがインストールされていません。")

warnings.filterwarnings('ignore')


# =============================================================================
# データ読み込み・前処理
# =============================================================================

def load_and_prepare_data(filepath, fill_missing=True, random_seed=42):
    """
    データの読み込みと前処理
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
# 学習効果の補正（案2：残差化）
# =============================================================================

def correct_learning_effect(df, target_col='操作時間'):
    """
    学習効果を残差化により補正する
    
    補正式: 補正後の値 = 実測値 − 実施順の効果
    実施順の効果 = 実施順別平均 − 全体平均
    """
    df_corrected = df.copy()
    
    # 全体平均
    grand_mean = df[target_col].mean()
    
    # 実施順別平均
    order_means = df.groupby('実施順')[target_col].mean()
    
    # 実施順の効果（全体平均からの偏差）
    order_effects = order_means - grand_mean
    
    # 補正の適用
    df_corrected['実施順効果'] = df_corrected['実施順'].map(order_effects)
    df_corrected[f'{target_col}_補正後'] = df_corrected[target_col] - df_corrected['実施順効果']
    
    # 補正情報を返す
    correction_info = {
        'grand_mean': grand_mean,
        'order_means': order_means,
        'order_effects': order_effects
    }
    
    return df_corrected, correction_info


def print_correction_summary(correction_info, target_col='操作時間'):
    """
    補正の内容をわかりやすく表示
    """
    print("\n" + "=" * 70)
    print(f"学習効果の補正（{target_col}）")
    print("=" * 70)
    
    print(f"\n全体平均: {correction_info['grand_mean']:.1f}秒")
    
    print("\n【実施順別平均と補正値】")
    print("-" * 50)
    print(f"{'実施順':<10} {'平均値':<15} {'補正値（全体平均との差）':<20}")
    print("-" * 50)
    
    for order in correction_info['order_means'].index:
        mean = correction_info['order_means'][order]
        effect = correction_info['order_effects'][order]
        sign = "+" if effect > 0 else ""
        print(f"{order}回目{'':<6} {mean:.1f}秒{'':<8} {sign}{effect:.1f}秒")
    
    print("-" * 50)
    print("\n【補正式】")
    print("  補正後の操作時間 = 実測値 − 補正値")
    
    # 動的な解釈
    first_order = correction_info['order_means'].index.min()
    last_order = correction_info['order_means'].index.max()
    first_effect = correction_info['order_effects'][first_order]
    last_effect = correction_info['order_effects'][last_order]
    
    print("\n【解釈】")
    print(f"  - {first_order}回目のデータ: {abs(first_effect):.0f}秒分「{'遅い' if first_effect > 0 else '速い'}」と補正される（不慣れの影響を除去）")
    print(f"  - {last_order}回目のデータ: {abs(last_effect):.0f}秒分「{'速い' if last_effect < 0 else '遅い'}」と補正される（習熟の影響を除去）")


# =============================================================================
# 反復測定ANOVA
# =============================================================================

def run_rm_anova(df, dv, within, subject='被験者ID'):
    """1要因反復測定ANOVAの実行"""
    return pg.rm_anova(data=df, dv=dv, within=within, subject=subject)


def run_posthoc(df, dv, within, subject='被験者ID', padjust='bonf'):
    """多重比較の実行"""
    return pg.pairwise_tests(data=df, dv=dv, within=within, 
                              subject=subject, padjust=padjust)


# =============================================================================
# 可視化
# =============================================================================

def create_correction_visualizations(df_corrected, output_dir='./'):
    """
    学習効果補正の可視化
    """
    # 1. 補正前後の実施順別比較
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 補正前
    order_before = df_corrected.groupby('実施順')['操作時間'].agg(['mean', 'std'])
    axes[0].bar(order_before.index, order_before['mean'], yerr=order_before['std'], 
                capsize=5, color='steelblue', alpha=0.7)
    axes[0].axhline(y=df_corrected['操作時間'].mean(), color='red', linestyle='--', label='全体平均')
    axes[0].set_xlabel('実施順')
    axes[0].set_ylabel('操作時間（秒）')
    axes[0].set_title('補正前：実施順別の平均操作時間\n（学習効果が見える）')
    axes[0].set_xticks(order_before.index)
    axes[0].legend()
    
    # 補正後
    order_after = df_corrected.groupby('実施順')['操作時間_補正後'].agg(['mean', 'std'])
    axes[1].bar(order_after.index, order_after['mean'], yerr=order_after['std'], 
                capsize=5, color='forestgreen', alpha=0.7)
    axes[1].axhline(y=df_corrected['操作時間_補正後'].mean(), color='red', linestyle='--', label='全体平均')
    axes[1].set_xlabel('実施順')
    axes[1].set_ylabel('操作時間（秒）')
    axes[1].set_title('補正後：実施順別の平均操作時間\n（学習効果が除去された）')
    axes[1].set_xticks(order_after.index)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}correction_order_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. 補正前後の条件別比較
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    conditions = sorted(df_corrected['条件'].unique())
    
    cond_before = df_corrected.groupby('条件')['操作時間'].agg(['mean', 'std']).reindex(conditions)
    axes[0].bar(cond_before.index, cond_before['mean'], yerr=cond_before['std'], 
                capsize=5, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('条件')
    axes[0].set_ylabel('操作時間（秒）')
    axes[0].set_title('補正前：条件別の平均操作時間')
    
    cond_after = df_corrected.groupby('条件')['操作時間_補正後'].agg(['mean', 'std']).reindex(conditions)
    axes[1].bar(cond_after.index, cond_after['mean'], yerr=cond_after['std'], 
                capsize=5, color='forestgreen', alpha=0.7)
    axes[1].set_xlabel('条件')
    axes[1].set_ylabel('操作時間（秒）')
    axes[1].set_title('補正後：条件別の平均操作時間')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}correction_condition_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. 補正の概念図
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for subject_id in df_corrected['被験者ID'].unique():
        subject_data = df_corrected[df_corrected['被験者ID'] == subject_id].sort_values('実施順')
        ax.plot(subject_data['実施順'], subject_data['操作時間'], 
                marker='o', alpha=0.3, color='gray', linestyle='--')
        ax.plot(subject_data['実施順'], subject_data['操作時間_補正後'], 
                marker='s', alpha=0.7, color='forestgreen')
    
    order_means_before = df_corrected.groupby('実施順')['操作時間'].mean()
    order_means_after = df_corrected.groupby('実施順')['操作時間_補正後'].mean()
    ax.plot(order_means_before.index, order_means_before.values, 
            color='red', linewidth=3, marker='o', markersize=10, label='補正前平均')
    ax.plot(order_means_after.index, order_means_after.values, 
            color='darkgreen', linewidth=3, marker='s', markersize=10, label='補正後平均')
    
    ax.set_xlabel('実施順')
    ax.set_ylabel('操作時間（秒）')
    ax.set_title('学習効果の補正：補正前（灰色破線）→ 補正後（緑実線）')
    ax.set_xticks(order_means_before.index)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}correction_concept.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n可視化を保存しました:")
    print(f"  - {output_dir}correction_order_comparison.png")
    print(f"  - {output_dir}correction_condition_comparison.png")
    print(f"  - {output_dir}correction_concept.png")


# =============================================================================
# 動的サマリー生成
# =============================================================================

def generate_dynamic_summary(results):
    """
    解析結果に基づいて動的なサマリーを生成
    
    Parameters
    ----------
    results : dict
        解析結果を含む辞書
    
    Returns
    -------
    summary : str
        サマリー文字列
    """
    summary_lines = []
    
    # ===== 操作時間のサマリー =====
    summary_lines.append("【操作時間】")
    
    # 学習効果
    learning_effect = results['learning_effect']
    if learning_effect['significant']:
        reduction = learning_effect['first_mean'] - learning_effect['last_mean']
        summary_lines.append(f"- 学習効果: 有意（p={learning_effect['p_value']:.4f}）")
        summary_lines.append(f"  → {learning_effect['first_order']}回目→{learning_effect['last_order']}回目で約{reduction:.0f}秒短縮")
    else:
        summary_lines.append(f"- 学習効果: 有意ではない（p={learning_effect['p_value']:.4f}）")
    
    # 補正方法
    summary_lines.append("- 補正方法: 残差化（実施順の効果を差し引き）")
    
    # 条件間差（補正後）
    time_condition = results['time_condition_corrected']
    if time_condition['significant']:
        summary_lines.append(f"- 条件間差: 有意（p={time_condition['p_value']:.4f}）")
        # 有意なペアがあれば表示
        if time_condition['significant_pairs']:
            pairs_str = ', '.join(time_condition['significant_pairs'])
            summary_lines.append(f"  → 有意差のあるペア: {pairs_str}")
    else:
        summary_lines.append(f"- 条件間差: 有意ではない（p={time_condition['p_value']:.4f}）")
    
    # 条件別平均（補正後）
    time_means = time_condition['means']
    best_cond = max(time_means, key=time_means.get)  # 最も遅い
    worst_cond = min(time_means, key=time_means.get)  # 最も速い
    summary_lines.append(f"- 条件別平均: 最速={worst_cond}({time_means[worst_cond]:.1f}秒), 最遅={best_cond}({time_means[best_cond]:.1f}秒)")
    
    summary_lines.append("")
    
    # ===== SUSスコアのサマリー =====
    summary_lines.append("【SUSスコア】")
    
    # 条件間差
    sus_condition = results['sus_condition']
    if sus_condition['significant']:
        summary_lines.append(f"- 条件間差: 有意（p={sus_condition['p_value']:.4f}）")
        if sus_condition['significant_pairs']:
            pairs_str = ', '.join(sus_condition['significant_pairs'])
            summary_lines.append(f"  → 有意差のあるペア: {pairs_str}")
    else:
        summary_lines.append(f"- 条件間差: 有意ではない（p={sus_condition['p_value']:.4f}）")
    
    # 条件別平均
    sus_means = sus_condition['means']
    best_cond = max(sus_means, key=sus_means.get)
    worst_cond = min(sus_means, key=sus_means.get)
    summary_lines.append(f"- 条件別平均: 最高={best_cond}({sus_means[best_cond]:.1f}点), 最低={worst_cond}({sus_means[worst_cond]:.1f}点)")
    
    # 学習効果
    sus_learning = results.get('sus_learning_effect', None)
    if sus_learning:
        if sus_learning['significant']:
            summary_lines.append(f"- 学習効果: 有意（p={sus_learning['p_value']:.4f}）→ 補正を検討")
        else:
            summary_lines.append("- 学習効果: なし（補正不要）")
    
    summary_lines.append("")
    
    # ===== 注意事項 =====
    summary_lines.append("【注意事項】")
    summary_lines.append("- 本解析では全条件に同一の補正を適用")
    summary_lines.append("- 条件×実施順の交互作用がある場合は案1（混合効果モデル）を検討")
    
    return "\n".join(summary_lines)


# =============================================================================
# メイン解析
# =============================================================================

def analyze_with_correction(df, output_dir='./'):
    """
    学習効果を補正した上での解析
    
    Returns
    -------
    df_corrected : DataFrame
    analysis_results : dict
    """
    # 学習効果の補正
    df_corrected, correction_info = correct_learning_effect(df, '操作時間')
    print_correction_summary(correction_info)
    
    # 補正前後の比較
    print("\n" + "=" * 70)
    print("補正前後の条件別比較（操作時間）")
    print("=" * 70)
    
    conditions = sorted(df['条件'].unique())
    
    print("\n--- 補正前 ---")
    before = df_corrected.groupby('条件')['操作時間'].agg(['mean', 'std'])
    for cond in conditions:
        print(f"  条件{cond}: {before.loc[cond, 'mean']:.1f} ± {before.loc[cond, 'std']:.1f}秒")
    
    print("\n--- 補正後 ---")
    after = df_corrected.groupby('条件')['操作時間_補正後'].agg(['mean', 'std'])
    for cond in conditions:
        print(f"  条件{cond}: {after.loc[cond, 'mean']:.1f} ± {after.loc[cond, 'std']:.1f}秒")
    
    # 反復測定ANOVA（補正後）
    print("\n" + "=" * 70)
    print("反復測定ANOVA（補正後の操作時間）")
    print("=" * 70)
    
    print("\n--- 条件の主効果 ---")
    aov = run_rm_anova(df_corrected, '操作時間_補正後', '条件')
    print(aov[['Source', 'ddof1', 'ddof2', 'F', 'p-unc', 'ng2']].to_string())
    
    p_val = aov['p-unc'].values[0]
    print(f"\n→ p = {p_val:.4f}", end="")
    if p_val < 0.05:
        print(" で有意（p<0.05）")
    else:
        print(" で有意ではない")
    
    # 多重比較
    print("\n--- 多重比較（Bonferroni補正）---")
    posthoc = run_posthoc(df_corrected, '操作時間_補正後', '条件')
    print(posthoc[['A', 'B', 'T', 'p-unc', 'p-corr', 'hedges']].to_string())
    
    # 有意なペアを抽出
    sig_pairs_time = []
    for _, row in posthoc[posthoc['p-corr'] < 0.05].iterrows():
        sig_pairs_time.append(f"{row['A']} vs {row['B']}")
    
    # 可視化
    create_correction_visualizations(df_corrected, output_dir)
    
    # 結果を辞書にまとめる
    analysis_results = {
        'correction_info': correction_info,
        'time_condition_corrected': {
            'p_value': p_val,
            'significant': p_val < 0.05,
            'means': after['mean'].to_dict(),
            'significant_pairs': sig_pairs_time
        }
    }
    
    return df_corrected, analysis_results


def main(filepath, output_dir='./'):
    """
    メイン関数
    """
    print("=" * 70)
    print("GUI評価データ 統計解析レポート（v2 Dynamic：学習効果補正版）")
    print("=" * 70)
    
    # データ読み込み
    df = load_and_prepare_data(filepath)
    n_subjects = df['被験者ID'].nunique()
    print(f"\n被験者数: {n_subjects}名")
    print(f"総データ数: {len(df)}件")
    
    conditions = sorted(df['条件'].unique())
    orders = sorted(df['実施順'].unique())
    
    # 結果を格納する辞書
    results = {}
    
    # ===== 1. 学習効果の検出 =====
    print("\n" + "=" * 70)
    print("Step 1: 学習効果の検出")
    print("=" * 70)
    
    aov_order = run_rm_anova(df, '操作時間', '実施順')
    if 'p-GG-corr' in aov_order.columns and pd.notna(aov_order['p-GG-corr'].values[0]):
        p_order = aov_order['p-GG-corr'].values[0]
    else:
        p_order = aov_order['p-unc'].values[0]
    print(f"\n実施順の主効果: F={aov_order['F'].values[0]:.2f}, p={p_order:.4f}")
    
    # 実施順別平均
    order_means = df.groupby('実施順')['操作時間'].mean()
    first_order = orders[0]
    last_order = orders[-1]
    
    if p_order < 0.05:
        print("→ 学習効果が有意に検出されました。補正を適用します。")
    else:
        print("→ 学習効果は有意ではありませんが、念のため補正を適用します。")
    
    results['learning_effect'] = {
        'p_value': p_order,
        'significant': p_order < 0.05,
        'first_order': first_order,
        'last_order': last_order,
        'first_mean': order_means[first_order],
        'last_mean': order_means[last_order]
    }
    
    # ===== 2. 補正と解析 =====
    print("\n" + "=" * 70)
    print("Step 2: 学習効果の補正と条件間比較")
    print("=" * 70)
    
    df_corrected, analysis_results = analyze_with_correction(df, output_dir)
    results.update(analysis_results)
    
    # ===== 3. SUSスコアの解析 =====
    print("\n" + "=" * 70)
    print("Step 3: SUSスコアの解析（参考：補正不要）")
    print("=" * 70)
    
    # 実施順の効果（学習効果）を確認
    aov_sus_order = run_rm_anova(df, 'SUSスコア', '実施順')
    if 'p-GG-corr' in aov_sus_order.columns and pd.notna(aov_sus_order['p-GG-corr'].values[0]):
        p_sus_order = aov_sus_order['p-GG-corr'].values[0]
    else:
        p_sus_order = aov_sus_order['p-unc'].values[0]
    
    results['sus_learning_effect'] = {
        'p_value': p_sus_order,
        'significant': p_sus_order < 0.05
    }
    
    print("\n--- 条件の主効果 ---")
    aov_sus = run_rm_anova(df, 'SUSスコア', '条件')
    print(aov_sus[['Source', 'ddof1', 'ddof2', 'F', 'p-unc', 'ng2']].to_string())
    
    p_sus = aov_sus['p-unc'].values[0]
    sig_pairs_sus = []
    
    if p_sus < 0.05:
        print(f"\n→ p = {p_sus:.4f} で有意（p<0.05）")
        print("\n--- 多重比較（Bonferroni補正）---")
        posthoc_sus = run_posthoc(df, 'SUSスコア', '条件')
        print(posthoc_sus[['A', 'B', 'T', 'p-unc', 'p-corr', 'hedges']].to_string())
        
        for _, row in posthoc_sus[posthoc_sus['p-corr'] < 0.05].iterrows():
            sig_pairs_sus.append(f"{row['A']} vs {row['B']}")
        
        if sig_pairs_sus:
            print("\n有意差のあるペア:")
            for pair in sig_pairs_sus:
                print(f"  {pair}")
    else:
        print(f"\n→ p = {p_sus:.4f} で有意ではない")
    
    print("\n--- 条件別平均 ---")
    sus_means = df.groupby('条件')['SUSスコア'].mean()
    for cond in conditions:
        print(f"  条件{cond}: {sus_means[cond]:.1f}点")
    
    results['sus_condition'] = {
        'p_value': p_sus,
        'significant': p_sus < 0.05,
        'means': sus_means.to_dict(),
        'significant_pairs': sig_pairs_sus
    }
    
    # ===== 動的サマリー =====
    print("\n" + "=" * 70)
    print("解析サマリー")
    print("=" * 70)
    
    summary = generate_dynamic_summary(results)
    print("\n" + summary)
    
    # データ保存
    df_corrected.to_csv(f'{output_dir}corrected_data.csv', index=False, encoding='utf-8-sig')
    print(f"\n補正後データを保存: {output_dir}corrected_data.csv")
    
    return df_corrected, results


if __name__ == "__main__":
    filepath = '/mnt/user-data/uploads/data__1_.xlsx'
    df_result, results = main(filepath, output_dir='./')
