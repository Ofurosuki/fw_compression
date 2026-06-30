# fw_compression — プロジェクト概要（ChatGPTディスカッション用）

> このドキュメントは、外部（ChatGPT）と議論するために現状を1枚にまとめた自己完結サマリ。
> 一次資料: `CLAUDE.md`（建て付け）, `downstream/RESULTS.md`（フローズン評価の全結果）,
> `downstream/RETRAIN_RESULTS.md`（アーキ固定の再学習）, `SPARSE_EVENT_TOPM.md`（トークンネイティブ）。

---

## 1. 何の研究か（目的）

**Full-waveform (FW) LiDAR の波形圧縮研究。** 1ピクセルあたり `T=700` の時間波形を持つFW LiDARに対して:

- **コア問い:** 波形を圧縮すると、**下流のゴースト検出タスク**がどれだけ劣化するか?
- **サブ問い:** 密な波形 `x[t]` の代わりに、**スパースな top-K 輸送イベント表現** `{(t_i, a_i, w_i)}`
  （ピーク位置・強度・幅）で下流性能を代替できるか?

**評価の肝（フローズン・ジャッジ方式）:** 下流のゴースト検出モデル（**FWL-ToPM**, 3クラス
voxelセグメンテーション {object, glass, ghost}, ＋noise）を **凍結した固定審判** として使い、
「圧縮→復元した波形」を食わせて F1 がどれだけ落ちるかを測る。下流モデルは再学習しない（当初は）。

### なぜフローズン・ジャッジか
**MSE が下流品質の正しい代理指標ではない**ことが繰り返し確認されているから。再構成MSEが悪化しても
下流F1が改善する事例（anti-hallucination loss, EMG非対称カーネル）が複数あり、「再構成の良さ」ではなく
「タスクにとっての良さ」を直接測る必要がある → そのためにこの凍結審判ハーネスが存在する。

---

## 2. 計測の前提（重要な落とし穴）

- **報告するF1は論文の指標と違う。** 同じ予測から2種類のF1が出て ~0.07 ずれる:
  - **voxel-level**（全 ~10M voxel を採点, 背景/裾含む）: `run_eval.py` の `macro_f1`。ベースライン ≈ **0.532**。
  - **peak-level**（scipy `find_peaks` のピーク位置だけ採点）: ベースライン ≈ **0.599** = 論文の "F1-mean ≈ 0.592"。
  - 我々はpeak検出を基本スキップ（遅い, ~2h/config）。**voxel-levelは論文非互換だが、圧縮手法間の相対比較には自己整合。**
- **F1-mean = {object, glass, ghost} の3クラス平均**（noiseは平均から除外、ただしCM上は競合クラスとして残す）。
  noiseを含む4クラスmacro（F1≈0.99）とは別物（混同注意）。
- **テストセット:** SPLIT2 test の3シーン（`36build`, `22build`, `14build_7floor`）= 30 dirs / 1427 frames。
  下流モデルもAEもこの3シーンは見ていない（held-out）。sweepは `divide=3`（~475 frames, 全解像度と≤0.003一致）。
- **圧縮の挿入位置:** 下流モデル自身のcrop/normalizeの**前**に、`VoxelDataset._load_voxel_grid` を
  monkey-patchして圧縮を差し込む（"T=700-first": ピクセルごとmax正規化 → 変換 → 逆正規化）。

---

## 3. これまでにやったこと（時系列・スレッド別）

### スレッドA: AE圧縮 × フローズン評価（`RESULTS.md`）
- **1D AE**（per-pixel linear encoder + MLP decoder）と **spatial 4×4 AE**（4×4ブロック共有）と
  naive coarse-binning を、圧縮比 6×〜88× でスイープ。
- **結果:** spatial 4×4 が最も頑健（88×でも 0.426 = ベースライン0.524の~82%）。1D は中間、naive は崩壊。
  **glass が一貫してボトルネッククラス**（圧縮なしでも~0.30）。
- **anti-hallucination loss**（背景オーバーシュート抑制＋微分可能な偽ピーク罰）を導入 → ほぼ全configで
  下流F1改善（Δ +0.01〜+0.07, 高圧縮ほど効く）。**再構成val-MSEは悪化するのにF1は改善** → MSE-is-wrong-proxyの実証。

### スレッドB: top-K 輸送イベント表現（`RESULTS.md` §6, フローズン）
- 波形を `{(t_i,a_i,w_i)}` のスパースリストに置換 → Gaussianパルスで擬似波形合成 → 同じ凍結ToPMに投入。
- **アブレーション:** `t`（位置のみ）/ `ta`（+強度）/ `tw`（+幅）/ `taw`（全部）。
- **結果（フローズン, neurips_best）:** `taw` K=2 ≈ 0.426（full 0.524の81%）@ 117×圧縮。
  `t`≈0.13-0.16, `ta`≈0.18, `tw`≈0.30, `taw`≈0.41。**幅 > 強度**。`ta`→`taw` の +0.23 は
  「マルチエコーLiDAR に対するFW（パルス形状）の価値」と解釈。
- **glass崩壊・ghost診断:** ghostは2次返り（中央値rank2）でK≥2必要、recall律速、近距離ghostが落ちる（depth依存, AUROC 0.71）。

### スレッドC: ⚠️ アーキ固定の再学習（`RETRAIN_RESULTS.md`）— **重要な方向転換（PI指摘）**
- **PI指摘:** フローズン実験とFW_Event_Netは「表現」と「アーキ/学習」を同時に変えていて、表現単体の価値を分離できない。
- **方法:** アーキを **ToPMに固定**し、入力表現だけ変えて（各repをT=700擬似波形にリフト）**ToPMをゼロから再学習**。
  表現効果 = F1(rep再学習) / F1(full波形再学習)。
- **結果（2-seed, voxel/peak）:**
  - full 0.536/0.598, **taw K4 0.523/0.575, ta K4 0.514/0.574** → **taw ≈ ta ≈ full（96-97.5%）**。
  - **フローズンが表現の価値を大幅に過小評価していた。差はinformation lossではなくdomain shiftだった**
    （再学習で taw +0.12, ta +0.34 回復）。
  - **幅(width)のネット価値は ~0**（taw−ta = +0.009 voxel / +0.001 peak, seedで符号反転）。
    フローズンの +0.23 は純粋にdomain shiftだった。
  - **per-classでは width は ghost を助け glass を害す**（再分配、ネットは動かない）。
    機構: ghost幅は転移する（ghostは一貫して細い）が、glass幅は**シーン依存で符号反転**し非転移。
- **takeaway:** アーキ固定＋適応下では**密な全波形は不要**。スパース top-K `(t,a,w)`（12数/pixel, ~58×小）で互角、
  プレーンなマルチエコー `(t,a)` でも97%。「幅が大きく効く / FW≫マルチエコー」という以前の読みは**訂正**。

### スレッドD: FW_Event_Net（スクラッチ学習イベントネット, メモリ参照）
- イベントテンソル `{(t,Δt,a,w,m)}` をスクラッチ学習。V2（cross-event attention+deep UNet）taw K=4 = **0.555**（93%）。
- 多数のネガティブ（DG/V-REx, NeRF transmittance, 直接/間接チャネル分解, radiometric補正）→
  **glass天井は表現問題**との結論。

### スレッドE: SparseEventToPM — トークンネイティブ（`SPARSE_EVENT_TOPM.md`）— **最新**
- **PI仮説:** EventNet V2(0.555) → ToPM-retrain(0.582) のギャップは、EventNetが K events を
  spatial mixing前にchannelへflattenするせい? → events を `(h,w,k)` トークンのまま保つネットで検証。
- **結果:** 3変種すべて ≈0.545 ≈ V2、ToPM 0.582 には届かず。flatten は律速ではなかった。
  - hierarchy(PatchMergeK)が v1 のglass崩壊を修復（glass 0.234→0.267 ≈ ToPM）。
  - 残ギャップは**全部ghost**（0.61 vs 0.72）。global attention は val↑/test flat = **非転移**（本プロジェクト恒例のシグネチャ）。
- **UPDATE(2026-06-26):** ghostギャップは dense-vs-sparse ではなく **precision/loss問題**だった（finding 5撤回）。
  - recallはToPMと同一（同じ入力events, 両者~72%検出）。ギャップは100%precision（false ghost過剰予測）。
  - 根因はEventNet由来のloss重み `[0.2,1,2,2]` の偏り。focal `[1,1,2,1]` で ghost precision 0.54→0.64,
    headline 0.545→**0.554 ≈ EventNet V2 0.555**、object 0.774（denseの天井超え）。
  - **しかしloss調整はprec/recトレードオフ上を滑るだけ。** dice追加も val↑/test↓で非転移。
    **全loss変種が held-out ghost F1 ~0.60-0.62 で頭打ち、ToPMの0.72/0.72には届かない。**

---

## 4. 現状の到達点（数字サマリ）

| 表現 / モデル | 方式 | peak-level F1-mean | メモ |
|---|---|--:|---|
| full waveform | フローズンToPM | 0.595/0.599 | 論文0.592 ≈ これ |
| full waveform | ToPM再学習(天井) | **0.598** | アーキ固定の上限 |
| taw K4 | ToPM再学習 | 0.575 | **ほぼfull（96%）** |
| ta K4（マルチエコー相当） | ToPM再学習 | 0.574 | **fullの96%, widthのネット価値~0** |
| taw K4 | FW_Event_Net V2 | 0.555 | スクラッチ学習イベントネット |
| taw K4 | SparseEventToPM v2+focal | 0.554 | トークンネイティブ, V2と互角 |

- **per-class傾向（全スレで一貫）:** object は天井に到達しやすい。**ghost** は K≥2 必要・ToPM(dense 3D conv)が最強。
  **glass がボトルネック**（fullでも~0.30, 圧縮で最初に崩れる, 幅は非転移で助けにならない）。
- **run分散 ±0.03**, 多くが single/2-seed。

---

## 5. 確立した主要な知見

1. **MSE は下流品質の代理指標ではない**（anti-hallucination, EMGで実証） → 凍結審判ハーネスの存在理由。
2. **フローズン評価は表現の価値を過小評価する**（domain shift混入）。アーキ固定＋再学習が公平な比較。
3. **密な全波形は（適応下では）ほぼ不要** — スパース top-K `(t,a,w)` で互角、マルチエコー `(t,a)` でも97%。
4. **パルス幅のネット価値は ~0**（ghost↔glassの再分配のみ）。「FW≫マルチエコー」は当初の過大評価。
5. **glass天井は表現問題**（多数のDG/物理補正がネガティブ）。glassはストレステストであって成功基準にすべきでない。
6. **ghostギャップは情報の天井ではなくprecision/loss問題**。ただしToPMの dense 3D conv のghost汎化(0.72/0.72)に
   トークンネイティブは loss調整では届かない（held-out 0.60-0.62で頭打ち, val↑/test↓非転移）。

---

## 6. 今ぶつかっている壁 / オープンな論点（← ここをChatGPTと議論したい）

- **glass がどのアプローチでも動かない。** 表現問題と結論づけたが、どんな表現/inductive biasなら glass の
  透明物体キューを転移可能な形で捉えられる? （裾/残差? 空間文脈? 偏光的な何か?）
- **ghostの held-out天井（0.60-0.62 vs ToPM 0.72）。** トークンネイティブ＋loss調整では届かない。
  ToPMの dense 3D conv が secondary returnsを「denoise」して汎化する優位を、スパース系で再現する手は?
- **val↑ / held-out test flat の非転移シグネチャが恒例化。** global attention, dice, DG手法すべてこの形。
  本質的にシーン間でoverfitしやすい何か。これを破る方向性は?
- **方向転換の整理:** 「密波形ほぼ不要」が確定した今、研究のストーリー/貢献をどう立てるか。
  圧縮（実用）軸 vs 表現分析（科学）軸のどちらを主にするか。

---

## 7. キーファイル早見表

| ファイル | 内容 |
|---|---|
| `CLAUDE.md` | プロジェクトの建て付け・環境・実行方法・findings overview |
| `downstream/RESULTS.md` | フローズン評価の全結果（AE圧縮 + top-Kイベント） |
| `downstream/RETRAIN_RESULTS.md` | アーキ固定ToPM再学習（表現効果の分離） |
| `SPARSE_EVENT_TOPM.md` | トークンネイティブ SparseEventToPM |
| `downstream/SCORE_DISCREPANCY.md` | voxel vs peak F1 の0.07差調査 |
| `downstream/AW_JOINT_WHY_TA_EQ_TAW.md` | なぜ ta ≈ taw か（(a,w)分布分析） |
| `eventnet_v4.md` / `event_aware_experiment_plan.md` | イベントネット系の実験計画 |
| `compression/event_extraction.py` / `event_synthesis.py` | top-K抽出 + Gaussian合成 |
