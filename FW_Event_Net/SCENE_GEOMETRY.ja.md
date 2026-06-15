# シーンごとの object / glass / ghost の幾何分布（Ghost-FWL データセット）

FW_Event_Net 調査で実施した **シーン別の幾何/transport 分析** の集約版（英語版: `SCENE_GEOMETRY.md`）。
glass がなぜ最難クラスなのか、そして per-ray の「奥に何があるか」系の手掛かり（behind_energy・
分解・透過率・放射補正）が**なぜシーンを越えて転移しないか**を説明する：**クラスの深さ順序そのものが
シーン依存**であり、そこから導く手掛かりは held-out シーンで符号反転するため。

> TL;DR — glass は **多くのシーンで object より手前にあるが、`gym_build`（train）と
> `14build_7floor`（test）では奥にある**。「透過/後段エネルギー」は機構的に深さに連動するので、
> glass-vs-object の識別**方向がこれらのシーンで反転**する。反転は **train シーンでも**起きるため、
> これは train/test の分け方の問題ではなく**特徴量に内在**する。下流への帰結は FW_Event_Net/RESULTS.md。

---

## 1. データセット・split・測定対象

- **10 シーン**（建物）。リポジトリ **SPLIT2**：held-out **TEST** 3 シーン
  (`36build`, `22build`, `14build_7floor`)、残りが **TRAIN**
  (`11build`, `14build_2floor`, `16build`, `16buildA_large`, `16buildA_mid`,
  `gym_build`, `34build`)。
- 以下の per-scene 表は、各シーンで **最初の 3 つの `hist` dir × 各先頭 4 フレーム** を標本とし、
  下流の y/z クロップ後（T=300）に top-K イベント抽出（`eventnet.events.extract_frame_events`）
  したもの。ラベル = 各イベントのピーク bin の annotation 値。`34build` は表に無い
  （標本フレームで glass イベントが閾値 <50 未満）— 知見ではなく標本抽出の都合。
- **符号付き AUC の規約**：特徴 `f` について AUC(glass vs object) = P(f_glass > f_object)。
  **>0.5 = glass の方が大きい / <0.5 = 符号反転（object の方が大きい）**、0.5 = 分離なし。

---

## 2. クラス分布

- **イベントレベル**（キャッシュ済み top-K の有効イベント）：noise **76 %**、object **12 %**、
  glass **4 %**、ghost **1.3 %**。glass+ghost で約 5 % → 少数・高分散クラス。
- **ボクセルレベル**（代表的な `36build` 1 フレーム、生 (400,512,700)）：背景 ~140.5 M、
  object 2.30 M、glass 0.17 M、ghost 0.36 M ボクセル。glass が最も稀な信号。

---

## 3. クラス別の特徴量中央値（混合シーン標本）

各クラスの返りがどこに位置し、パルス形状がどうか（有効イベントの中央値）：

| 特徴量 | object | glass | ghost | 読み方 |
|---|--:|--:|--:|---|
| 振幅 `a`（ピーク高、max正規化） | 1.00 | 0.93 | 0.26 | ghost = 暗い二次リターン |
| 幅 `w`（FWHM, bins） | 10 | 10 | 8 | ほぼ同一 → 幅はほとんど分離しない |
| `behind_energy`（Σwn[t:]/Σ） | 0.57 | 0.77 | 0.09 | glass は後段にエネルギー、ghost は最後尾 |
| `D_after`（後段の直達質量） | 0.50 | **0.82** | 0.07 | glass = 奥に実物体がある |
| `I_after`（後段の間接/拡散質量） | 0.04 | 0.07 | 0.01 | 小（本データの ghost は離散ピーク） |
| 透過率 `Tp(δ)` δ=0/8/16/32/64 | .49/.08/.04/.02/.004 | **.73/.56/.25/.10/.03** | .07/.01/.005/0/0 | glass = 緩やかな減衰（半透過） |

単体では glass に対し非常に discriminative に見える（特に `D_after`、`Tp` 減衰形）。
だが以下の per-scene 表が、その分離が**汎化しない**理由を示す。

---

## 4. シーン別：後段エネルギーと深さ（中核の幾何テーブル）

`E_*` = クラス別 behind_energy 中央値、`t_*` = ピーク深さ bin 中央値（0–299、大きいほど遠い）。
`AUC g|o` = glass-vs-object の behind_energy（>0.5 = glass が大きい）。

| scene | set | E_glass | E_obj | E_ghost | t_glass | t_obj | glass と obj の深さ | AUC g\|o | AUC gh\|o |
|---|---|--:|--:|--:|--:|--:|---|--:|--:|
| 11build | train | 0.661 | 0.587 | 0.093 | 46 | 55 | glass が手前 | 0.668 | 0.109 |
| 14build_2floor | train | 0.657 | 0.576 | **0.891** | 34 | 86 | glass が手前（大） | 0.739 | 0.814 |
| 16build | train | 0.652 | 0.590 | 0.142 | 23 | 33 | glass が手前 | 0.723 | 0.105 |
| 16buildA_large | train | 0.789 | 0.632 | 0.380 | 55 | 53 | ほぼ同じ | 0.671 | 0.212 |
| 16buildA_mid | train | 0.772 | 0.641 | 0.493 | 19 | 29 | glass が手前 | 0.718 | 0.303 |
| gym_build | train | 0.665 | 0.711 | 0.138 | 56 | 32 | **glass が奥** | **0.392** | 0.127 |
| 22build | TEST | 0.725 | 0.653 | 0.094 | 38 | 52 | glass が手前 | 0.710 | 0.022 |
| 36build | TEST | 0.636 | 0.637 | 0.128 | 21 | 21 | ほぼ同じ | 0.553 | 0.040 |
| 14build_7floor | TEST | 0.625 | 0.631 | 0.423 | 42 | 33 | **glass が奥** | 0.485 | 0.191 |

まとめ：behind_energy の glass\|object AUC は **平均 0.63、範囲 [0.39, 0.74]**、
**2/9 シーンで反転（<0.5）** — `gym_build`（train）と `14build_7floor`（test）。

**幾何的な読み**：AUC>0.5（glass の後段エネルギーが大）が成り立つのは **glass が object より手前のとき**
（手前のピークほど後ろに波形が多く残る）。glass が奥のシーン（`gym_build`, `14build_7floor`）では**反転**。
つまり「glass は後ろにエネルギーがある」は「このシーンでは glass がたまたま手前」の代理であり、
glass の物理ではなくシーンのレイアウト。

### ghost の深さに関する注記
ghost の behind_energy はほぼ全シーンで**低い**（0.09–0.49）→ ghost は後段にほとんど何もない遅い返り。
例外は **`14build_2floor`（E_ghost 0.891、AUC gh|o 0.814）** — 近距離 ghost の集団（光線の手前にある
ghost で後ろに多くが残る）。これは既知のシーン間 ghost 明るさ/幾何のドメインギャップと整合
（memory `fwc-real-data-domain-gap` 参照）。

---

## 5. シーン別：深さ層別の後段エネルギー（手掛かりは“ただの深さ”か？）

`raw` = behind_energy の glass\|object AUC、`depth-strat` = **深さビン内で揃えた**同 AUC（深さを統制）。
手掛かりが純粋に幾何なら、層別で 0.5 に近づくはず。

| scene | set | raw AUC | depth-strat AUC |
|---|---|--:|--:|
| 11build | train | 0.667 | 0.543 |
| 14build_2floor | train | 0.736 | 0.574 |
| 16build | train | 0.724 | 0.603 |
| 16buildA_large | train | 0.674 | 0.661 |
| 16buildA_mid | train | 0.722 | 0.606 |
| gym_build | train | 0.394 | 0.398 |
| 22build | TEST | 0.709 | 0.669 |
| 36build | TEST | 0.553 | 0.443 |
| 14build_7floor | TEST | 0.488 | 0.555 |
| **平均** | | **0.63** | **0.56** |

**見かけの glass 信号の約半分は純粋に深さ**（平均 0.63 → 0.56）。弱い残差（0.56）は残るが
**依然シーン非一貫**（2/9 反転：gym_build 0.398、36build 0.443；14build_7floor の raw 反転は 0.555 に*回復*）。
→ 学習した幾何減衰の残差でも救えない。

---

## 6. シーン別：距離補正した振幅（放射補正）

振幅を材質後方散乱の代理 `ρ = a·R(t)²`（距離補正、R∝t+c0）に直すと glass の手掛かりがシーン一貫に
なるか、の検証。物理仮説：glass（半透過）→ 材質後方散乱が一貫して*低い*（全シーンで AUC <0.5）。
`a` = 生振幅の glass\|object AUC、`ρ` は距離ゼロ点 c0 を 3 通り、最終列は深さ層別 ρ。

| scene | set | raw `a` | ρ(c0=25) | ρ(c0=125) | ρ(c0=325) | ρ depth-strat |
|---|---|--:|--:|--:|--:|--:|
| 11build | train | 0.473 | 0.370 | 0.379 | 0.393 | 0.479 |
| 14build_2floor | train | 0.565 | 0.221 | 0.244 | 0.263 | 0.511 |
| 16build | train | 0.640 | 0.290 | 0.308 | 0.319 | 0.580 |
| 16buildA_large | train | 0.569 | 0.534 | 0.529 | 0.542 | 0.470 |
| 16buildA_mid | train | 0.564 | 0.246 | 0.266 | 0.324 | 0.390 |
| gym_build | train | 0.676 | 0.672 | 0.684 | 0.687 | 0.728 |
| 22build | TEST | 0.549 | 0.283 | 0.293 | 0.302 | 0.349 |
| 36build | TEST | 0.664 | 0.505 | 0.524 | 0.523 | 0.744 |
| 14build_7floor | TEST | 0.639 | 0.687 | 0.659 | 0.640 | 0.543 |

距離補正しても一貫しない：ρ 平均 ~0.43 だが **範囲 [0.22, 0.69]、符号は依然反転（5/9 <0.5、4/9 >0.5）**、
c0 に頑健、ばらつきは生より*拡大*。反転シーン（`gym_build`, `14build_7floor`）はやはり glass が奥の
シーンで、×R² が遠い glass を過剰増幅し反転を*増幅*。交絡は相対的な手前/奥の**幾何**であって、
放射的な距離だけではない。

---

## 7. シーン別：透過率減衰プロファイル（NeRF 風）

`Tp0` = ピーク位置の透過率 survival（≈ behind_energy）、`Tp8` = peak+8、
`ratio8/0` = Tp8/Tp0（「半透過の plateau」指標）。すべて glass\|object AUC。

| scene | set | Tp0 | Tp8 | ratio8/0 |
|---|---|--:|--:|--:|
| 11build | train | 0.666 | 0.618 | 0.601 |
| 14build_2floor | train | 0.734 | 0.663 | 0.632 |
| 16build | train | 0.731 | 0.696 | 0.681 |
| 16buildA_large | train | 0.683 | 0.654 | 0.632 |
| 16buildA_mid | train | 0.711 | 0.657 | 0.594 |
| gym_build | train | 0.385 | 0.337 | 0.313 |
| 22build | TEST | 0.706 | 0.684 | 0.656 |
| 36build | TEST | 0.529 | 0.430 | 0.397 |
| 14build_7floor | TEST | 0.448 | 0.461 | 0.478 |
| **平均** | | **0.62** | **0.58** | **0.55** |

減衰形の主要指標も同じシーンで反転（gym_build, 14build_7floor、ratio では 36build も）→
透過率プロファイルは behind_energy 以上には転移しない。

---

## 8. 幾何に関する横断的結論

1. **クラスの深さ順序がシーン固有。** glass は 6–7/9 シーンで object より手前、`gym_build` と
   `14build_7floor` では奥。この一つの幾何事実が、あらゆる「後段/透過/距離」系の手掛かりの
   符号反転を駆動する。
2. **glass の手掛かりの失敗は幾何由来であって放射由来ではない。** 深さ層別で約半分が消え、
   距離補正でも直らず（反転が残る/悪化）、残差はシーン非一貫。
3. **ghost は幾何的に単純だが 1 シーンだけ外れ値。** ghost = どこでも低い後段エネルギー（遅い返り）、
   例外は `14build_2floor`（近距離 ghost）でシーン間 ghost ドメインギャップと一致。
4. **反転は TRAIN シーン（`gym_build`）でも起きる** → 訓練セット自体に矛盾した glass↔幾何関係が
   混在するので、再 split でも domain-generalization 損失（V-REx ≈ ERM）でも不変な glass 手掛かりを
   作り出せない。これは **表現** の限界：per-ray のスカラー/形状による「奥に何があるか」の要約は、
   シーン幾何を越えてクラス不変にならない。

---

## 9. 由来と注意

- probe（ロジックは transcript に保存、出力を本書に転記）：シーン別 behind_energy & 深さ、
  深さ層別 AUC、放射補正 `ρ=a·R²`、透過率プロファイル `Tp(δ)`。クラス別中央値は
  `eventnet.events` の 12 列キャッシュの単体テストから。
- 標本 = 各シーン 最初の 3 hist dir × 先頭 4 フレーム（テストセット全体ではない）。AUC の絶対値は
  ±0.02 程度の標本ノイズを含むが、**符号/反転パターンが頑健な結果**で、4 つの独立 probe で再現する。
- `34build`（train）は未収録（標本閾値）。深さはクロップ後の bin（0–299）。
- これら幾何事実の下流影響（どの特徴/アーキを試し、なぜ失敗したか）は **FW_Event_Net/RESULTS.md**、
  関連 memory：`fwc-eventnet`, `fwc-behind-energy-litreview`, `fwc-real-data-domain-gap`。
