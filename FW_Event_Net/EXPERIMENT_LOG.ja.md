# FW_Event_Net — 実験ログ集約（V1 → 現在）

イベントネット調査の全実験を 1 ページに集約した台帳。スコアはすべて **論文準拠 peak-level F1**
（SPLIT2 テスト＝held-out 3 シーン）、論文「F1-mean」と同一の指標・母集団。詳細: `RESULTS.md`、
シーン別幾何: `SCENE_GEOMETRY.ja.md`。

## 参照上限（凍結 full-waveform Ghost-FWL / FWL-ToPM、同一テスト・同一指標）
- **peak-level F1-mean 0.599**（= 論文 0.592）、voxel-level 0.532。クラス別 peak：object 0.742、
  glass 0.385、ghost 0.669。モデル規模 **8.72 M** params。

## ヘッドライン
- **V2 `taw` K=4 = 0.555（2-seed）= 上限 0.599 の 93 %**。入力は **>100× 小さい**
  （≤4 数値 × 4 イベント vs 700 サンプル）、**7.85 M** params（判定器 8.72 M とほぼ同規模）。
- object はすでに full-waveform を**上回る**（0.75 vs 0.742）。残る差は glass（≈0.31 vs 0.385）で、
  これは **表現/ドメインギャップ** の問題（「教訓」参照）。

## 推論速度（cuda:2 ~Blackwell、B=1、fp32；warmup + 計測 iters + CUDA同期）
| 構成 | 入力 | params | ms/fwd | FPS |
|---|---|--:|--:|--:|
| FWL-ToPM（full-waveform） | crop 300×168×200（ToMe枝刈り後） | 8.72 M | 9.8 | 102 |
| V2 `taw` forward | full 400×336×K4×F4 | 7.85 M | 15.6 | 64 |
| V2 イベント抽出（taw, bare） | raw → events | — | 10 | 100 |
| **V2 `taw` end-to-end**（抽出+fwd） | raw → logits | 7.85 M | **25.6** | **39** |

- forward 単体は ToPM が ~1.6× 速いが、**カバー範囲は 43%**（200×168 crop vs V2 は全 400×336 面）。
  ToPM は ToMe トークン併合 + 強度枝刈りで実効入力が小さく速い。
- **全フレーム同一カバーなら同等**（各 ~24 ms：ToPM は ~2.4 crops、V2 は 25.6 ms で全面）。V2 end-to-end ≈ **39 FPS** で実時間可。
- **圧縮の利点は帯域/データ量（>100×）であって速度ではない** — 推論速度は同オーダー。
  （V3/V4 の 12 列分解抽出は 394 ms だが、`taw` は 10 ms の bare 抽出のみで足りる。）
- **V2 forward 内訳**（パラメータ≠レイテンシ）：cross-event attention **11.6 ms / 75%**（ほぼ無パラだが
  B·H·W = 13.4万本の per-pixel × K=4 シーケンス → memory/launch-bound・低稼働率）、U-Net 3.2 ms / 21%
  （パラの大半を持つが計算密度高く高速）、event-MLP 0.7 ms。→「パラの割に遅い」のは**ボトルネックが
  ほぼ無パラの per-pixel attention** だから。ここを軽くすれば（軽い関係演算/fp16/ヘッド削減）forward は
  ~5 ms まで縮みうる。ToPM はパラ多いが ToMe 併合+枝刈りで実効トークンが減るので速い。

---

## マスター結果表

### A. V1 — `EventTensorNet`（event MLP + rank-emb → channel-flatten → 2層 U-Net、1.9 M）
特徴量 ablation（K=4、単一 seed）：

| feature | F1-mean | object | glass | ghost |
|---|--:|--:|--:|--:|
| t_only | 0.500 | 0.720 | 0.265 | 0.514 |
| t_dt | 0.509 | 0.720 | 0.283 | 0.522 |
| ta | 0.525 | 0.738 | 0.281 | 0.555 |
| taw | 0.525 | 0.741 | 0.274 | 0.561 |
| tdta | 0.529 | 0.756 | 0.265 | 0.565 |
| **tdtaw** | **0.534** | 0.754 | 0.273 | 0.576 |

K スイープ（tdtaw）：K1 **0.357**（ghost 0.058!）、K2 0.497、**K4 0.534**、K8 0.523。
→ 強度が主レバー（t_dt→tdta +0.020）、V1 では**幅は限界的**（taw−ta +0.00）、ghost は **K≥2** 必要、K=4 最良。

### B. V2 — `EventTensorNetV2`（+ cross-event attention + 3層 U-Net、GELU、7.85 M）
特徴量 ablation（K=4、2-seed 平均；seed42/seed43）：

| feature | F1-mean | glass | 備考 |
|---|--:|--:|---|
| ta | 0.534 | 0.268 | 0.531 / 0.536 |
| **taw** | **0.555** | **0.307** | 0.565 / 0.544 — **ヘッドライン** |
| tdta | 0.524 | 0.252 | 0.536 / 0.511 |
| tdtaw | 0.525 | 0.257 | 0.509 / 0.542 |

→ V2 で taw が V1 比 +0.03。**幅が効くようになる**（taw−ta +0.021、両 seed）。**Δt は冗長/有害化**
（taw > tdtaw）— attention がリターン間の時間関係を学習。最良表現 = `taw`。

### C. V2 `taw` 上での特徴量実験（glass の差を埋める狙い）— すべて NEGATIVE
| 実験 | 構成 | taw control 比 | 判定 |
|---|---|---|---|
| **behind_energy**（生の透過E） | taE / tdtaE / tdtaEw (s42) | ta 0.525→taE **0.498**（glass −0.066） | 悪化；深さ/シーン交絡で符号反転 |
| **直達/間接 分解** | tawD / tawI / tawi (s42、in-run base 0.536) | tawD 0.500、tawI 0.521、tawi 0.535 | ≈/悪化；glass 動かず |
| **放射(距離)補正** | 診断のみ（ρ=a·R²） | per-scene AUC が依然反転 | 転移しない（学習未実施） |
| **NeRF 透過率プロファイル** | tawT (2-seed) | taw 0.538 → tawT **0.536**（glass +0.007） | ≈ taw、±0.03 内；形状も転移せず |

共通原因：glass の「奥に何があるか」手掛かりが**幾何/シーン依存**（train シーン含め符号反転）
→ per-ray のスカラー/形状要約は転移しない（`SCENE_GEOMETRY.ja.md`）。

### D. V2 `taw` 上の学習側レバー
| 実験 | 構成 | 結果 | 判定 |
|---|---|---|---|
| **V-REx（ドメイン汎化）** | β=1/10/30、scene=env | 2-seed erm 0.531 vs β1 **0.539**；β10/30 悪化 | ≈ ERM（Gulrajani-LopezPaz）；実質無効 |
| **損失: glass class-weight** | ×1.5 / ×3 | 2-seed glass +0.018（seed非頑健）、**F1 横ばい** | precision/recall トレード、底上げなし |
| **損失: focal (γ=2)** | seed42 | F1 0.535、glass 0.296 | class-weight と同じトレード |

### E. アーキテクチャ: spatial attention（`v2sa`、+1.06 M）
| run | レシピ | F1 (2-seed) | ghost | 判定 |
|---|---|--:|--:|---|
| v2sa 初回 | 後付け（lr1e-3、warmup なし、40ep） | 0.523 | 0.201 | 人工物 — *under-training*（val も↓、過学習ではない） |
| base（公正） | lr5e-4、warmup5、50ep | 0.536 | 0.583 | control |
| **v2sa（公正）** | lr5e-4、warmup5、50ep | **0.541** | 0.612 | **中立** — ΔF1 +0.005、per-seed 符号反転（+0.029/−0.019）；seed42 の ghost +0.08 は再現せず（s43 −0.022） |

→ 公正レシピで under-training は解消（v2sa val ≈ base）するが、test では v2sa **≈ base**（seed 依存、±0.03 内）。
spatial attention は**中立**でヘッドラインではない。F1-mean 0.541 < ToPM 0.599。

---

## 時系列（一行判定）
1. **V1 ablation** → sparse events は機能；強度≫幅；ghost は K≥2；tdtaw K4 = 0.534（89 %）。
2. **V1 K-sweep** → K=4 最適；K=1 で ghost 崩壊。
3. **V2（attention）** → +0.03 → **taw 0.555（93 %）**；幅が効く；Δt 冗長。**ヘッドライン。**
4. **behind_energy** → 単体では最強特徴だが **悪化**（深さ/シーン交絡）。
5. **per-scene + 深さ層別 + 放射補正 診断** → 交絡は **幾何**；split 由来でない（train シーンでも反転）。
6. **V3 分解（直達/間接）** → 転移せず；glass 動かず。
7. **V4 NeRF 透過率プロファイル** → 転移せず（同じシーンで形状反転）。
8. **V-REx DG** → ≈ ERM。
9. **損失スイープ（glass-weight / focal）** → glass は動くが precision/recall トレード；F1 横ばい・seed非頑健。
10. **spatial attention** → 初回は under-training 人工物；**公正レシピ 2-seed = 中立**（ΔF1 +0.005、seed反転；seed42 の ghost +0.08 は非再現）。ヘッドラインにならず。

---

## 横断的教訓
- **アーキ > 特徴量。** ヘッドラインを確実に押し上げたのは V2 アーキのみ（V1→V2 +0.03）。
  後付け特徴や DG 損失はいずれも効かず。
- **「分離するが転移しない（separable but not transferable）」。** 拡散系 4 特徴（width, behind_energy,
  分解, NeRF-T）＋放射補正は単体/訓練上は discriminative に見えるが held-out シーンで転移しない
  — glass 手掛かりが幾何/シーン依存だから。
- **幅の価値はアーキ依存**（V1 で限界的、V2 で実効）。
- **Δt は attention があれば冗長。**
- **glass の壁は表現/ドメインギャップの問題**。6 角度（4 特徴 + DG 損失 + 損失調整）で確認、
  さらに容量増（spatial attn）が後付けでは悪化することが補強。残る有望レバーは別表現
  = **transport/NeRF を end-to-end**（Option B、大規模実装）のみ。
- **±0.03 の run/cache/seed 分散は実在**（同一 taw/V2/seed42 が 4 列キャッシュ 0.565 vs 7 列 0.536）
  → 必ず within-run control + multi-seed；単発の <0.03 差はノイズ。

## 確定 vs 未決
- **確定：** sparse events は判定器とほぼ同規模で上限の ~93 % に到達；object は full-waveform に
  匹敵/凌駕；glass が律速で、per-ray 特徴・DG 損失・損失調整では埋まらない。
- **未決/進行中：** spatial attention の公正再検証（改良レシピ、GPU-2）；未試行のアーキ/学習ノブ
  （EMA、強 aug、別バックボーン）；**Option B** transport/NeRF 表現（残る主レバー）。

*(詳細手法・表: `RESULTS.md`。シーン別幾何: `SCENE_GEOMETRY.md` / `SCENE_GEOMETRY.ja.md`。
memory: `fwc-eventnet`, `fwc-behind-energy-litreview`。)*
