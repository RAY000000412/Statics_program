# GUI評価データ統計解析ツール

GUIユーザビリティ評価実験のデータを統計的に分析するためのPythonスクリプト集です。

## 概要

4種類のGUIレイアウト（条件A〜D）を24名の被験者が評価する反復測定デザインの実験データを解析します。

### 主な機能

- 反復測定ANOVA（Repeated Measures ANOVA）
- 学習効果の検出と補正
- 混合効果モデル（Linear Mixed-Effects Model）
- 効果量の算出と検出力分析
- 結果の可視化

## ファイル構成

```
├── gui_evaluation_analysis.py      # v1: 基本解析（反復測定ANOVA）
├── gui_evaluation_analysis_v2.py   # v2: 学習効果補正版（残差化）
├── gui_evaluation_analysis_v3.py   # v3: 混合効果モデル版
├── Analysis_Power.py               # 効果量・検出力分析
└── README.md                       # このファイル
```

## 各スクリプトの説明

### 1. gui_evaluation_analysis.py（v1）

**目的**: 基本的な反復測定ANOVAによる解析

**機能**:
- 条件（A/B/C/D）と実施順（1〜4回目）の主効果を検定
- 球面性の検定とGreenhouse-Geisser補正
- 多重比較（Bonferroni補正）
- 基本的な可視化（箱ひげ図、学習効果の推移）

**出力**:
- `data_distribution.png`: データ分布の可視化
- `learning_effect.png`: 学習効果の推移グラフ

---

### 2. gui_evaluation_analysis_v2.py（v2）

**目的**: 学習効果を補正した上での条件間比較（案2：残差化による補正）

**機能**:
- 実施順の効果を算出し、各データから差し引く
- 補正後のデータで反復測定ANOVAを実行
- 補正前後の比較可視化

**補正方法**:
```
補正後の操作時間 = 実測値 − 実施順の効果
実施順の効果 = 実施順別平均 − 全体平均
```

**出力**:
- `corrected_data.csv`: 補正後のデータ
- `correction_order_comparison.png`: 補正前後の実施順別比較
- `correction_condition_comparison.png`: 補正前後の条件別比較
- `correction_concept.png`: 補正の概念図

**使用場面**:
- 発表でわかりやすく説明したい場合
- 条件×実施順の交互作用がない（または小さい）場合

---

### 3. gui_evaluation_analysis_v3.py（v3）

**目的**: 混合効果モデルによる厳密な解析（案1）

**機能**:
- 実施順を共変量として含めた混合効果モデル
- 条件×実施順の交互作用の検定
- 調整済み平均（Adjusted Means）の算出
- ペアワイズ比較

**モデル式**:
```
操作時間 ~ 条件 + 実施順 + (1|被験者ID)           # 主効果モデル
操作時間 ~ 条件 * 実施順 + (1|被験者ID)           # 交互作用モデル
```

**出力**:
- `mixed_model_effects.png`: 条件・実施順の効果プロット
- `mixed_model_adjusted_means.png`: 生データ vs 調整済み平均の比較

**使用場面**:
- 統計的に厳密な解析が必要な場合
- 条件×実施順の交互作用が疑われる場合
- 査読論文などで報告する場合

---

### 4. Analysis_Power.py

**目的**: 効果量の算出と検出力分析

**機能**:
- 効果量の算出（η², Cohen's f, Cohen's d / Hedges' g）
- 現在のサンプルサイズでの検出力推定
- 目標サンプルサイズでの検出力推定
- 必要サンプルサイズの逆算（80%, 90%検出力）
- 検出力曲線の可視化

**出力**:
- `power_curve_time.png`: 操作時間の検出力曲線
- `power_curve_sus.png`: SUSスコアの検出力曲線
- `effect_size_comparison.png`: 効果量の比較図

**使用場面**:
- サンプルサイズの妥当性を説明する場合
- 「なぜ24名なのか」と聞かれた場合の根拠として

---

## 必要なライブラリ

```bash
pip install pandas openpyxl numpy scipy pingouin statsmodels matplotlib seaborn japanize-matplotlib
```

### ライブラリ一覧

| ライブラリ | バージョン | 用途 |
|-----------|-----------|------|
| pandas | >= 1.3.0 | データ操作 |
| openpyxl | >= 3.0.0 | Excelファイル読み込み |
| numpy | >= 1.20.0 | 数値計算 |
| scipy | >= 1.7.0 | 統計検定 |
| pingouin | >= 0.5.0 | 反復測定ANOVA、多重比較 |
| statsmodels | >= 0.13.0 | 混合効果モデル |
| matplotlib | >= 3.4.0 | 可視化 |
| seaborn | >= 0.11.0 | 可視化 |
| japanize-matplotlib | >= 1.1.0 | 日本語フォント対応 |

---

## 使い方

### 入力データ形式

Excelファイル（.xlsx）で以下の列が必要です：

| 列名 | 説明 | 例 |
|------|------|-----|
| 被験者ID | 被験者の識別番号 | 1, 2, 3, ... |
| 条件 | GUIの条件（A〜D） | A, B, C, D |
| 実施順 | 何回目に実施したか | 1, 2, 3, 4 |
| SUSスコア | System Usability Scale | 0〜100 |
| 操作時間 | タスク完了時間（秒） | 120, 180, ... |
| SEQスコア | Single Ease Question（オプション） | 1〜7 |

### 基本的な実行方法

```python
# v1: 基本解析
python gui_evaluation_analysis.py

# v2: 学習効果補正版
python gui_evaluation_analysis_v2.py

# v3: 混合効果モデル版
python gui_evaluation_analysis_v3.py

# 検出力分析
python Analysis_Power.py
```

### ファイルパスの変更

各スクリプトの末尾にある `filepath` を自分のデータファイルに変更してください：

```python
if __name__ == "__main__":
    filepath = '/path/to/your/data.xlsx'  # ← ここを変更
    main(filepath, output_dir='./')
```

### 出力ディレクトリの指定

`output_dir` パラメータで出力先を指定できます：

```python
main(filepath, output_dir='./results/')
```

---

## 推奨ワークフロー

```
1. データ収集（8名分など）
       ↓
2. gui_evaluation_analysis.py で基本解析
   → 学習効果の有無を確認
       ↓
3. Analysis_Power.py で検出力分析
   → 目標サンプルサイズで十分な検出力があるか確認
       ↓
4. データ収集完了（24名分）
       ↓
5. gui_evaluation_analysis_v2.py で補正解析（発表用）
       ↓
6. gui_evaluation_analysis_v3.py で厳密解析（必要に応じて）
```

---

## v2とv3の使い分け

| 観点 | v2（残差化） | v3（混合効果モデル） |
|------|-------------|---------------------|
| **わかりやすさ** | ◎ 直感的 | △ やや複雑 |
| **統計的厳密さ** | ○ 十分 | ◎ 最も厳密 |
| **交互作用** | × 考慮しない | ◎ 考慮可能 |
| **発表向け** | ◎ 推奨 | △ 専門家向け |
| **論文向け** | ○ 可 | ◎ 推奨 |

**推奨**:
- まずv2で解析・発表
- 質問されたらv3の結果も提示できるよう準備

---

## 効果量の解釈基準

### Cohen's f（ANOVA用）

| 値 | 解釈 |
|----|------|
| < 0.10 | 小さい（small） |
| 0.10 - 0.25 | 中程度（medium） |
| 0.25 - 0.40 | 大きい（large） |
| > 0.40 | 非常に大きい（very large） |

### Cohen's d / Hedges' g（2群比較用）

| 値 | 解釈 |
|----|------|
| < 0.20 | 小さい（small） |
| 0.20 - 0.50 | 中程度（medium） |
| 0.50 - 0.80 | 大きい（large） |
| > 0.80 | 非常に大きい（very large） |

---

## 注意事項

1. **ダミーデータ**: SUSスコアに欠損がある場合、条件別の平均・標準偏差から乱数生成で補完します。本番解析では `fill_missing=False` に設定してください。

2. **球面性の仮定**: 反復測定ANOVAでは球面性の仮定を検定し、違反している場合はGreenhouse-Geisser補正を適用します。

3. **多重比較**: Bonferroni補正を使用しています。比較数が多い場合は保守的になりすぎる可能性があります。

4. **検出力分析の前提**: 8名分のデータから推定した効果量に基づいています。真の効果量が異なれば検出力も変化します。

---

## ライセンス

MIT License

---

## 更新履歴

| 日付 | バージョン | 内容 |
|------|-----------|------|
| 2025/01 | v1.0 | 初版作成 |
