# スコア乖離調査レポート：論文値 0.592 vs 我々の F1-mean ~0.52

> 作成日: 2026-06-12 / **解決追記: 2026-06-13**  
> 対象チェックポイント: `neurips_best` = `vit3d_ordered_pruning_light_finetune_epoch_50_20260423_221908_0.02523.pth`（= リポジトリの "baseline" FWL-ToPM、augmentation なし）

---

## 1. 概要 / TL;DR

> **【解決済み 2026-06-13】乖離の原因は「採点する母集団」の違いだった。論文の 0.592 は peak-level F1（検出ピーク位置のみで採点）、我々が出していた ~0.52 は voxel-level F1（3D格子の全ボクセルを採点）。同一の baseline 重み・同一テストセットで repo 純正 `run_test.py` を回すと、voxel-level=0.532／peak-level=0.599 が同時に出力され、peak-level が論文の 0.592 と一致する。**

経緯：我々の full-waveform（無圧縮）ベースライン F1-mean は **~0.52**（3-class signal mean; object/glass/ghost）であり、論文の報告値 **≈ 0.592** と約 0.07 の乖離があった。スコア計算アルゴリズム（metric 関数・予測ロジック・dataset 前処理・テストセット・重み・divide・crop-seed）を精査した結果、いずれも乖離の原因ではないことが確認された。当初は「チェックポイントのバリアント違い」を残余仮説としたが（§6、後に**棄却**）、ユーザーより「論文の 0.592 は augmentation なしの baseline 値」との指摘を受け再調査。repo 純正 `run_test.py` が **voxel-level と peak-level の2種類の F1 を出力する**ことを突き止め、**論文値 = peak-level（0.599）**、**我々の値 = voxel-level（0.532）** であると確定した（§7）。スコア計算アルゴリズム・重み・データ・閾値・ρ はすべて同一であり、我々のパイプラインは正常。圧縮手法間の相対比較（すべて voxel-level で統一）は有効である。

---

## 2. 背景：論文値とのギャップ

### 各スコアの定義

| 表記 | 定義 | 値 |
|------|------|----|
| **論文 F1-mean** | object / glass / ghost の per-class F1 の平均（3-class signal mean）。**実体は peak-level**（§7 で確定） | ≈ **0.592** |
| **我々の F1-mean（voxel, 3-class）** | 全ボクセルを採点した 3-class mean、`neurips_best`、divide=1（全 1427 フレーム） | **0.5209** |
| **我々の macro F1（voxel, 4-class）** | noise を含む 4 クラス voxel macro（repo の `macro_f1` フィールド相当） | **0.638** |
| **peak-level F1（3-class）** | 検出ピーク位置のみで採点（§7、repo 純正 divide=10） | **0.599** ≈ 論文 |

- **2つの「採点母集団」**：repo は同じ予測から **voxel-level**（3D格子の全ボクセル ~1000万個を採点）と **peak-level**（`find_peaks` で検出した返り波ピーク位置のボクセルのみ採点）の2種の F1 を出す。voxel-level は背景・裾の曖昧ボクセルも全部効くので辛く（~0.53）、peak-level は信号位置だけ採点するので高く出る（~0.60）。**論文の 0.592 は peak-level**（§7）。
- **Convention の違いにも注意**：noise クラスを confusion matrix から除外（`ignore_visualize_labels=[0]`）すると F1 が ~0.14 膨らむ。論文・リポジトリともに `ignore_visualize_labels=[]`（noise を competing class として残す）が正規 convention で、本レポートの数値はすべてこの convention による。
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
| **チェックポイントのバリアント** | repo "baseline" ckpt と aug-variant の比較（§6） | **棄却**（論文値は baseline 由来とユーザー確認） |
| **採点母集団（voxel vs peak）** | repo 純正 `run_test.py` が出す peak-level F1 を確認（§7） | **これが原因**（peak-level=0.599 ≈ 論文 0.592） |

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

## 6. 〔棄却された仮説〕チェックポイントのバリアント違い

当初（2026-06-12 時点）、乖離の最有力説明として「`neurips_best` は repo の "baseline" ckpt で、論文の 0.592 は別の augmentation バリアントの上位 ckpt によるもの」と考えた。しかし **2026-06-13 にユーザーより「論文の 0.592 は augmentation なしの baseline FWL-ToPM の値」との指摘**を受け、この仮説は**棄却**された（baseline = `neurips_best` 自身であり、同じ重みで 0.592 が出るはずなのに我々は 0.52 しか出ていなかった）。

参考に、val-loss 最小の2本（`cutmix0.2_0.8` 0.02485 → 3-class **0.517**、`baseline` 0.02523 → **0.521**）はどちらも ~0.52 で、val-loss は F1 を予測しないことも確認済み。→ 真因は §7。

### 副次的確認：repo 純正 run_test.py（voxel-level）は我々と一致

repo 純正 `run_test.py`（`neurips_best`、divide=10）の **voxel-level** "Overall Detailed Metrics"：

| Class | Precision | Recall | **F1-Score** |
|-------|-----------|--------|-------------|
| unknown (noise) | 0.9929 | 0.9860 | **0.9894** |
| object | 0.7145 | 0.6735 | **0.6934** |
| glass | 0.2639 | 0.4997 | **0.3454** |
| ghost | 0.4492 | 0.7318 | **0.5567** |
| **Macro avg (4-class)** | 0.6051 | 0.7228 | **0.6462** |

voxel-level 3-class mean = (0.6934 + 0.3454 + 0.5567) / 3 = **0.5318** → 我々の divide=10 値 0.5349 と差 0.003 以内で一致。**repo 純正ツールと我々のハーネスは同じ voxel-level 数値を出す**ことが直接確認された（パイプライン忠実性の証明）。

---

## 7. 〔解決〕真因は voxel-level vs peak-level の採点母集団の違い

同じ `run_test.py` 実行ログ（`/tmp/repo_d10.log`）には、voxel-level とは別に **peak-level F1**（`detect_peaks_in_voxel` で `find_peaks` 検出したピーク位置のボクセルのみで採点）も出力されていた。"Average Peak Metrics Across All Build_IDs"：

| Class | Peak_Precision | Peak_Recall | **Peak_F1** |
|-------|----------------|-------------|-------------|
| unknown (noise) | 0.6320 | 0.6537 | **0.6427** |
| object | 0.8057 | 0.6881 | **0.7423** |
| glass | 0.3076 | 0.5160 | **0.3854** |
| ghost | 0.6646 | 0.6730 | **0.6687** |
| **Macro avg (4-class)** | 0.6025 | 0.6327 | **0.6172** |

**peak-level 3-class signal mean** = (0.7423 + 0.3854 + 0.6687) / 3 = **0.5988 ≈ 0.599**

これは **論文の 0.592 とクロップ seed 分散の範囲内で一致**する。

### 結論

| 指標（同一 baseline 重み・同一テストセット・repo 純正 run_test.py, divide=10） | object | glass | ghost | **3-class mean** |
|---|--:|--:|--:|--:|
| **voxel-level**（全ボクセルを採点。我々のヘッドライン） | 0.693 | 0.345 | 0.557 | **0.532** |
| **peak-level**（検出ピーク位置のみ採点。論文の指標） | 0.742 | 0.385 | 0.669 | **0.599** |

- **論文の "F1-mean 0.592" は peak-level**。我々が出していた ~0.52 は voxel-level。アルゴリズム・重み・データ・閾値・ρ_low(0.7)/merge(0.9) はすべて同一で、**唯一の違いは「採点する母集団」**（3D格子の全ボクセル vs 返り波ピーク位置のみ）。
- 我々の `run_eval.py` は計算コストの高い per-pixel `find_peaks` を意図的にスキップして voxel-level のみ算出していた（README にも明記）。これが ~0.52 vs ~0.59 の正体。
- 我々の event 圧縮スイープはすべて voxel-level で統一されているため、**相対比較（taw vs ta、K 依存、full vs 圧縮）はすべて有効**。

### 次のステップ

1. （論文と絶対値を揃える場合）`run_eval.py` に peak-level F1 を追加し、headline 構成（full / taw K=2,3 等）のみ peak-level で測り直す。per-pixel peak 検出は重い（~2h/config）ため主要構成に限定するのが現実的。
2. voxel-level のまま相対比較を続ける場合は追加作業不要（現行の結論はすべて有効）。
