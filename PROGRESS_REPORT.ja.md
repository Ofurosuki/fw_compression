# 研究経過報告 — Full-Waveform LiDAR 波形圧縮 / スパース・トランスポートイベント表現

> 作成日: 2026-06-24 / ブランチ `feature/remove_falsepositive`
> 既存ドキュメントの統合・要約版。詳細は各ドキュメントへリンク:
> - 圧縮 AE / イベント表現の本編: `downstream/RESULTS.md`
> - スコア乖離（voxel vs peak）調査: `downstream/SCORE_DISCREPANCY.md`
> - アーキテクチャ統制 retrain: `downstream/RETRAIN_RESULTS.md`, `downstream/SESSION_PROGRESS_2026-06-18.md`
> - 学習済みイベントネット: `FW_Event_Net/RESULTS.md`, `FW_Event_Net/EXPERIMENT_LOG.md`, `FW_Event_Net/SCENE_GEOMETRY.md`
> - 初期計画: `initial_research_plan.md`, `event_aware_experiment_plan.md`

---

## 1. 背景 (Background)

### 1.1 問題設定
Full-Waveform (FW) LiDAR は各画素で時間方向 `T=700` の密な波形 `x[t]` を取得する。これは
通常のマルチエコー LiDAR（各リターンの距離＋強度のリスト）より遥かに情報量が多く、
帯域・蓄積・伝送コストが高い。本研究の中心的問いは:

> **FW 波形をどこまで圧縮できるか — そして密な波形を、スパースな
> top-K トランスポートイベント表現 `{(t_i, a_i, w_i)}`（ピーク位置・強度・幅）で
> 置き換えられるか。**

応用上の判定軸は「**下流のゴースト検出タスクの性能をどれだけ保てるか**」。
ガラス・ゴースト・マルチパスといった**トランスポート（光の伝播）構造**を保てるかが鍵で、
これは深度のみを保つ既存圧縮（CVPR2022 Compressive SP 3D Cameras / ICCV2023 Learned
Compressive Representations）とは目的が異なる。

### 1.2 評価の枠組み — frozen judge（凍結審判）
下流モデルは **FWL-ToPM**（`vit3d_ordered_pruning_light`, 8.72 M, 論文 "Towards Real-Time
FWL Transformers…"）。これを**再学習せず固定の審判**として使い、
「圧縮→復元した擬似波形」を入力したときの voxel/peak ごとの
4 クラス分類 `{noise, object, glass, ghost}` の F1 で圧縮品質を測る。

- **F1-mean = signal 3 クラス {object, glass, ghost} の per-class F1 平均**
  （noise は競合クラスとして CM に残すが平均からは除外、`ignore_visualize_labels=[]`）。
- データは repo の **split2 TEST**（`36build` / `22build` / `14build_7floor`、
  30 hist dir・1427 frame）。下流モデルと AE の学習に使わない held-out。
- 圧縮は下流の crop/normalize の**前段**に monkey-patch で挿入
  （per-pixel max 正規化 → 変換 → de-正規化、"T=700-first"）。repo はread-only。

### 1.3 ⚠️ 報告する指標は論文の指標と母集団が違う（重要）
同じ予測から 2 種類の F1 が出る:
- **voxel-level**（`run_eval.py` の `macro_f1`）— 全 ~1000 万 voxel を採点。ベースライン ≈ **0.532**。
- **peak-level**（`peak_macro_f1`）— `find_peaks` の返り波ピーク位置のみ採点。ベースライン ≈ **0.599**。
  **これが論文の「F1-mean ≈ 0.592」に対応**（同一重み・データ・閾値、母集団だけが違う。
  `SCORE_DISCREPANCY.md` で 2026-06-13 に決着）。

スイープは速度のため voxel-level（peak 検出は ~2h/config なので原則スキップ）。
voxel-level は**相対比較には完全に自己整合**だが、論文値と直接比較するときは peak-level を使う。

---

## 2. 手法 (Method) と実験系統

パイプラインの基本形:

```
raw 波形 x[T=700]
  → (圧縮 / イベント抽出)        ← ここを差し替えて比較
  → 擬似波形 x_hat[T=700]
  → frozen FWL-ToPM（または retrain した同アーキ）
  → object / glass / ghost の F1
```

これまでに 5 系統の実験を積み上げてきた。

### 2.1 オートエンコーダ圧縮（1D / spatial 4×4）
- `compression/encoders.py`（coarse_binning / random_projection / DCT / learnable_linear）+
  MLP デコーダ。K∈{8,16,32,64,128}、圧縮率 `T/K` = 6×〜88×。
- ICCV2023 に倣った **spatial 4×4 separable coding**（`spatial_coding.py`）も実装。
  近傍 16 画素を共同符号化（`C_k = c^t_k ⊗ c^s_k`）。
- split2 train（7 scene）で学習、split2 test で下流評価。

### 2.2 Anti-hallucination loss（偽ピーク抑制）
復元波形は真のピークを回復する一方、背景に**偽ピークを幻覚**して下流の誤検出を生む。
これに対し `reconstruction_loss`（`compression/autoencoder.py`）へ非ピーク bin だけにかかる 2 項を追加:
- `bg_weight·‖relu(x̂−x)‖²` — 背景オーバーシュート抑制（真値より**上**に作った信号のみ罰する非対称項）
- `fp_weight·relu(slope_L)·relu(slope_R)` — 微分可能な局所最大（偽ピーク）罰則

評価側にも spurious-peak 指標（`false_ghost_rate` ほか）を `compression/utils/metrics.py` に追加。
ブランチ `feature/remove_falsepositive`。

### 2.3 top-K トランスポートイベント表現
波形を**スパースなイベント列** `{(t_i, a_i, w_i)}` に置き換え、Gaussian パルスで擬似波形に合成して
同じ frozen FWL-ToPM に入れる（密な `x[t]` が必要か、ピーク・パラメータだけで足りるかの検証）。
- 抽出: `compression/event_extraction.py`（scipy 参照実装 ＋ GPU バッチ抽出 ~200ms/frame）。
- 合成: `compression/event_synthesis.py`、`x̂[t]=Σ a_i·exp(−(t−t_i)²/2σ_i²)`。
- アブレーション `representation ∈ {t, ta, tw, taw}`（位置のみ / +強度 / +幅 / 全部）×
  `K∈{1,2,3,4,6,8}`。`ta` K=4 は**従来のマルチエコー LiDAR の類似物**（距離＋強度・幅なし）。

### 2.4 FW_Event_Net（スクラッチ学習のイベントネット）
frozen judge とは別系統で、**top-K イベントテンソルを直接入力**して各イベントを
`{noise, object, glass, ghost}` に分類する**スクラッチ学習ネット**（`eventnet/`）。
- V1: event MLP + rank 埋め込み → channel-flatten → 2-level U-Net（1.9 M）。
- V2: **イベント横断 attention**（per-pixel transformer, K 個のリターンを相互参照）+ 3-level U-Net（7.85 M）。
- メトリクスは**論文準拠の peak-level F1**（repo の `detect_peaks_in_voxel`/`evaluate_peaks` を再利用、
  raw 波形ピークを採点母集団に固定 → 論文 0.592 / frozen 0.599 と直接比較可）。

### 2.5 アーキテクチャ統制 retrain（PI の方法論ピボット, 2026-06-17〜）
> frozen judge と FW_Event_Net は**表現とアーキ/学習を同時に**変えていて、表現単独の価値を切り分けられない。

そこで **ToPM アーキを固定**し、各表現を T=700 擬似波形に持ち上げて
**ToPM を同一レシピでスクラッチ再学習**する。表現の有効性 = F1(rep で再学習した ToPM) / F1(full 波形で再学習した ToPM)。
- 基盤: `downstream/run_retrain.py`（repo の `train_vit3d` を monkey-patch で駆動）、
  `cache_repr.py`（イベント/AE 表現を uint16 でキャッシュ、~6min/epoch）、
  `run_eval_peak.py`（peak-level 評価）。
- レシピ統一: AdamW lr1e-4 / focal+dice / cosine / 50ep / no-aug / divide=3。voxel と peak の両方で報告。

---

## 3. これまでの主要結果 (Key Findings)

### 3.1 スパースイベントは下流性能の大半を保つ（frozen judge）
`taw` K=2 ≈ **0.426（full 波形 0.524 の 81%）@ 117× 圧縮**。`taw` は K=2〜8 で平坦。
→ **Ghost-FWL の知覚は密な波形忠実度ではなく、スパースなトランスポートイベントで概ね決まる。**

### 3.2 強度も幅も効くが、frozen では「幅 > 強度」に見えた
位置のみ `t` ≈ 0.13–0.16、`ta` ≈ 0.18、`tw` ≈ 0.30、`taw` ≈ 0.41。
`ta`→`taw` の +0.23（主に object）は「マルチエコー上に乗る FW のパルス形状情報」の価値…
**に見えた**が、§3.5 で覆る。

### 3.3 anti-hallucination loss は MSE が悪化しても下流 F1 を上げる
ほぼ全 config で Δ +0.01〜+0.07（高圧縮ほど効く。1D 88× で +0.066）。
AH モデルは復元 val-MSE が**高い**のに下流 F1 が**良い** → **MSE は正しい代理指標ではない**。
この frozen-judge ハーネスはまさにそのために存在する。

### 3.4 FW_Event_Net: アーキテクチャが表現の結論を変える
- **V2 `taw` K=4 = 0.555（2-seed）= 0.599 天井の 93%**、>100× 小さい入力・同等のモデルサイズ。
- **width はアーキ依存**: V1 では marginal、V2（attention）では効く（`taw`−`ta`=+0.021、glass 0.27→0.31）。
- **Δt は attention 下で冗長/有害**（attention がリターン間タイミングを学習）。最良表現は `taw`。
- ghost は**二次リターン**で **K≥2 が必須**（K=1 で ghost F1 ≈ 0.06）。

### 3.5 アーキ統制 retrain: 「密な波形は要らない」/ frozen のギャップは大半が**ドメインシフト**
- **適応すればスパースイベントはほぼ無損失**: retrain `taw` K=4 = **0.531 vox / 0.582 peak ≈ full 0.533 / 0.595**。
- frozen のギャップ（taw +0.12, ta +0.34）は**情報損失ではなくドメインシフト**
  （frozen ToPM は合成 Gaussian/固定幅パルスを未学習で OOD 扱いしていた）。
- **width の真の正味価値は小さい**: アーキ固定＋適応下で `taw−ta` = +0.016 vox / +0.008 peak、
  **2-seed では正味 ≈ 0（peak で符号反転）**。マルチエコー `ta` だけで full の 96–97% に到達。
  → §3.2 の「幅が大きく効く / FW ≫ マルチエコー」は frozen のドメインシフト由来で、過大評価だった。
- **密な AE 復元はスパースイベントに負ける**（retrain 下で frozen のランキングが**逆転**）。
  MSE 最適な密復元はトランスポート構造をぼかす。AH は retrain 下でも +0.03 で有効。

### 3.6 glass は representation/domain-gap 問題（一貫した結論）
glass はどの系統でも頭打ち（frozen `taw` ≈0.25、FW_Event_Net V2 ≈0.31 vs ToPM 0.385）。
glass を狙った**特徴量 6 種**（width / behind_energy / direct-indirect 分解 / radiometric 補正 /
NeRF 透過プロファイル）と**DG 学習（V-REx）/ loss 調整 / spatial attention** が**すべて transfer しない**。
共通原因: glass の「背後に何があるか」の手掛かりは**シーン幾何依存で符号反転**（train scene でも反転）。
per-ray のスカラ/形状要約では転移しない。

### 3.7 横断的な教訓
- **アーキ > 特徴量**（V1→V2 +0.03 だけが頑健に効いた。bolt-on 特徴/DG loss は効かない）。
- **「単独で分離的 ≠ タスクに有用」**（width, EMG kernel, behind_energy, MSE が同じ罠）。
- **±0.03 の run/cache/seed variance は実在** → 必ず within-run control ＋ multi-seed。単発 <0.03 はノイズ。

---

## 4. 今後の流れ・展望 (Future Directions)

### 4.1 残された唯一の本命レバー — 微分可能トランスポート表現（Option B）
glass の頭打ちは**特徴量でも学習側でも閉じない**ことが 6 方向から確認済み。
残るのは表現そのものを変える方向:
> ピーク列 + bolt-on スカラではなく、**σ/T（密度・透過率）と direct/indirect の分離を
> end-to-end で学習する微分可能トランスポート表現**（transient-NeRF 系）。
幾何を**要約**するのではなく**モデル化**する。単一 ray からは劣決定なので
**spatial context / prior** が要る大きめのビルド。これが glass の 0.555→0.599 を埋める本命。

### 4.2 主張を論文比較可能にする（peak-level の徹底）
- frozen スイープの**ヘッドライン config を peak-level でも採点**して論文 0.592 と並べる
  （現状 frozen 側は voxel が主）。
- retrain 系は full peak 0.595 ≈ 論文 0.592 で**既に paper-comparable**。これを軸に表を統一。

### 4.3 統計的頑健性の底上げ
- retrain 系の K-sweep / `tw` / AE 再評価は**多くが single seed**。±0.003〜0.03 の variance に対し
  **2-seed 以上**で width margin・taw≈full・K の境界（K=4 で 58× は保つが K=2 で緩む）を確定。
- イベント検出パラメータ（smooth_σ, min_height, min_distance, fixed_width）と
  AE の bg/fp 重みは未スイープ。

### 4.4 表現の拡張（glass / 密残差）
- `taw_bg`（背景フロア）や **dense-residual / tail トークン**を足す**ハイブリッド**
  （スパースイベント + 小さな密残差チャネル）で glass と残差構造を拾えるか。
- ghost は depth 依存で**近距離（早期・主ピーク近傍）ゴーストが落ちる**ことが診断済み
  → spatial context で近距離ゴーストを補強する方向。

### 4.5 実機・運用面
- 圧縮の勝ちは**帯域/データ量（>100×）であって速度ではない**（V2 end-to-end ≈ 39 FPS で ToPM と同オーダー）。
  V2 の attention がレイテンシ支配（forward の 75%）なので、軽量化（fp16 / heads 削減）で ~5ms まで詰められる余地。
- Evaluator B（pruning/merging なしの素の ViT3D）は未実行（対応 ckpt が未配置）。
- スケール: 現状は事実上少数シーン。**brightness-diverse / 多シーン学習**が cross-scene 頑健性の前提。

### 4.6 直近の意思決定ポイント
1. Option B（微分可能トランスポート）に踏み込むか、それともハイブリッド残差トークン（4.4）で
   安価に glass を突くかの優先順位付け。
2. ヘッドライン表を **peak-level に統一**して対外発表用に整える。
3. width / taw≈full の **2-seed 確定**（主張の頑健性の最終確認）。

---

*関連メモリ: `fwc-eventnet`, `fwc-topm-retrain`, `fwc-peak-vs-voxel-f1`,
`fwc-behind-energy-litreview`, `fwc-aw-joint-why-ta-eq-taw`,
`fwc-false-peak-suppression`, `fwc-spatial-coding`, `fwc-real-data-domain-gap`.*
</content>
</invoke>
