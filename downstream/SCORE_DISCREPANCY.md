# スコア乖離調査レポート：論文値 0.592 vs 我々の F1-mean ~0.52

> 作成日: 2026-06-12  
> 対象チェックポイント: `neurips_best` = `vit3d_ordered_pruning_light_finetune_epoch_50_20260423_221908_0.02523.pth`

---

## 1. 概要 / TL;DR

我々の full-waveform（無圧縮）ベースライン F1-mean は **~0.52**（3-class signal mean; object/glass/ghost）であり、Ghost-FWL / FWL-ToPM 論文の報告値 **≈ 0.592** と約 0.07 の乖離がある。スコア計算アルゴリズム（metric 関数・予測ロジック・dataset 前処理・テストセット・重み・divide・crop-seed）は一通り精査した結果、いずれも乖離の原因ではないことが確認された。最有力の残余説明は**チェックポイントのバリアント違い**である：`neurips_best` はリポジトリ内で "baseline" として位置づけられるチェックポイントであり、論文の 0.592 はおそらく同リポジトリ内の別の augmentation バリアント（`rotate+noise`、`cutmix0.2_*` 等）で学習した上位チェックポイントによるものとみられる。我々のパイプライン自体は正常であり、圧縮手法間の相対比較は有効である。

---

## 2. 背景：論文値とのギャップ

### 各スコアの定義

| 表記 | 定義 | 値 |
|------|------|----|
| **論文 F1-mean** | object / glass / ghost の per-class F1 の平均（3-class signal mean） | ≈ **0.592** |
| **我々の F1-mean（3-class）** | 同上、`neurips_best` チェックポイント、divide=1（全 1427 フレーム） | **0.5209** |
| **我々の macro F1（4-class）** | noise を含む 4 クラス macro（repo の `macro_f1` フィールド相当） | **0.638** |

- **Convention の違いに注意**：noise クラスを confusion matrix から除外（`ignore_visualize_labels=[0]`）すると F1 が ~0.14 膨らむ（no-compression で **0.66 masked vs 0.52 unmasked**）。論文・リポジトリともに `ignore_visualize_labels=[]`（noise を competing class として残す）のが正規 convention であり、以下の数値はすべてこの convention による。
- 論文の 0.592 は 3-class mean であり、noise 込みの 4-class macro（~0.64）ではない。

---

## 3. 検証項目と結論の一覧

| 項目 | 検証方法 | 結論（乖離の原因か？） |
|------|----------|----------------------|
| **Metric 関数** | `run_eval.py` がリポジトリの `calculate_metrics_from_confusion_matrix` を import して呼ぶことをソース確認（F1=2TP/(2TP+FP+FN)、macro=valid_classes の平均） | **原因でない**（完全一致） |
| **予測ロジック** | ignore labels を -1e9 マスク → softmax → threshold 0.5 → class 0 fallback → argmax；`test_vit3d_model` と行単位で照合 | **原因でない**（完全一致） |
| **Dataset・前処理** | `VoxelDatasetWithToMe` の構築引数（target_size / divide / y_crop_top/bottom=88 / z_crop_front=25 / z_crop_back=375）、モデル/アーキ パラメータ（`use_threshold_prediction` 含む）を repo の split2 test config と対照 | **原因でない**（完全一致） |
| **テストセット** | split2 TEST 3 シーン（36build/22build/14build_7floor、30 hist dirs、1427 フレーム）を byte-level で set-equal と確認 | **原因でない** |
| **重み** | `neurips_best` pth が repo の `test/baseline/*.yaml` が参照するファイルと同一であることをユーザーが確認 | **原因でない** |
| **divide（フレームサブサンプリング）** | divide=1/3/10 で実測比較（§5 参照） | **原因でない**（差 ≤0.003） |
| **Random crop の seed** | シード 42/43/44/45 でスイープ（§4 参照） | **原因でない**（std=0.0024） |
| **チェックポイントのバリアント** | `neurips_best` = repo "baseline" ckpt；repo には aug-variant 上位 ckpt が複数存在（§6 参照） | **最有力の残余原因** |

---

## 4. ランダムクロップの発見

### クロップの仕組み

前処理の y/z クロップ適用後のグリッドサイズは **(X=400, Y=336, Z=350)** であるが、`target_size` は **(200, 168, 300)** である。3 軸すべてで `np.random.randint` ベースのランダムクロップが有効になっており、各フレームは元ボリュームの約 **43 %** の部分領域で評価される。

### Seeding の差異

- リポジトリ (`test_vit3d_model`): `DataLoader` の `worker_init_fn=seed_worker` + `generator=manual_seed(config.seed+2)` によって再現性を確保
- 我々のハーネス: `set_seed(config.seed)` のみ（seeding スキームが異なる）

### シードスイープ結果（neurips_best、divide=3、~475 フレーム）

| seed | object F1 | glass F1 | ghost F1 | **3-class mean** |
|------|-----------|----------|----------|-----------------|
| 42 | 0.6943 | 0.3289 | 0.5474 | **0.5235** |
| 43 | 0.6998 | 0.3265 | 0.5503 | **0.5255** |
| 44 | 0.6968 | 0.3236 | 0.5504 | **0.5236** |
| 45 | 0.7019 | 0.3289 | 0.5577 | **0.5295** |
| **mean** | **0.6982** | **0.3270** | **0.5514** | **0.5255** |
| **std** | 0.0030 | 0.0024 | 0.0040 | **0.0024** |

**結論**：std=0.0024、range=0.5235–0.5295 であり、crop-seed は **0.07 の乖離を説明しない**。なお、5 フレーム程度の極小サブセットではクロップ/フレームノイズが大きくなる（glass F1 で ±0.05 超）が、1427 フレーム規模では完全に平均化される。また固定 seed での 2 回実行は bit-for-bit 一致しており、我々のパイプラインは決定的であり、圧縮手法間の相対比較はすべて有効である。

---

## 5. divide=1 / divide=3 / divide=10 のベースライン数値

すべて `neurips_best` チェックポイント、no-compression、`ignore_visualize_labels=[]`。

| divide | フレーム数 | object F1 | glass F1 | ghost F1 | **3-class mean** | 4-class macro |
|--------|-----------|-----------|----------|----------|-----------------|---------------|
| 1（全件） | 1427 | 0.6948 | 0.3241 | 0.5437 | **0.5209** | 0.638 |
| 3（~1/3） | ~475 | 0.6943 | 0.3289 | 0.5474 | **0.5235** | 0.640 |
| 10（~1/10） | ~143 | 0.695 | 0.360 | 0.550 | **0.5349** | — |

divide=1 と divide=3 の差は **0.003 以内**（per-class std ≤0.004）であり、divide によるスコア変動は実質無視できる。divide=10 のわずかな upward bias（glass +0.04）は小サンプルのノイズである。

---

## 6. 結論と次のステップ

### 結論

スコア計算**アルゴリズムはリポジトリ実装と byte-identical** である（metric 関数・予測ロジック・dataset・テストセット・重みいずれも一致）。divide・crop-seed も乖離の原因ではない。

最有力の残余説明：

> **`neurips_best` はリポジトリの "baseline" FWL-ToPM チェックポイント（val loss ≈ 0.02523）であり、論文の 0.592 はリポジトリ内の別の augmentation バリアント（`aug-only-cutmix0.2_*`・`rotate+noise` 等）の上位チェックポイントによるものとみられる。**

実際、以前に使用した `cutmix0.2_0.8`（val loss ≈ 0.02485）チェックポイントでの 3-class F1 は **0.517**（`evalA_noignore.json`）であり、これも `neurips_best` の 0.520–0.524 と同水準に留まる。リポジトリには `paper/pruning0.7merging0.9-vit3d*/…aug-*/` 以下に多数のバリアントが存在し、より高い val-loss（val score）を示す上位バリアントが 0.592 を達成している可能性が高い。

### 副次的確認：repo 純正 run_test.py、divide=10 の結果

`/tmp/repo_d10.log` に "Overall Detailed Metrics" ブロックが含まれていたため、参考値として記載する（repo 純正 `run_test.py`、`neurips_best` ckpt、divide=10）：

| Class | Precision | Recall | **F1-Score** |
|-------|-----------|--------|-------------|
| unknown (noise) | 0.9929 | 0.9860 | **0.9894** |
| object | 0.7145 | 0.6735 | **0.6934** |
| glass | 0.2639 | 0.4997 | **0.3454** |
| ghost | 0.4492 | 0.7318 | **0.5567** |
| **Macro avg** | 0.6051 | 0.7228 | **0.6462** |

3-class signal mean（object/glass/ghost） = (0.6934 + 0.3454 + 0.5567) / 3 = **0.5318**

これは我々の divide=10 での 0.5349 と差 0.003 以内で一致する。**repo 純正ツールと我々のハーネスは同じ数値を出す**ことが直接確認された。

なお、repo の "Macro avg" F1（0.6462）は noise を含む 4-class macro であり、我々の `macro_f1_4class`（~0.640）と対応する。論文の 0.592 はこの "4-class macro" ではなく "3-class signal mean" の値である。

### 次のステップ

1. **上位 aug-variant チェックポイントの特定**：リポジトリ内の `…/paper/pruning0.7merging0.9-vit3d*/…aug-*/` チェックポイント群を同じハーネスで評価し、どのバリアントが 3-class F1 ≈ 0.592 を再現するかを確認する。
2. （オプション）特定できた場合、そのチェックポイントを `neurips_best` に代えて下流の圧縮評価を再実行し、ベースライン F1 を論文値に揃えた上で compression degradation の曲線を更新する。
