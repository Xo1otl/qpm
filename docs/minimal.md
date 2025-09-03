## Cascaded THGの連立ODE
#### **1. 結合波方程式**
カスケード二次非線形過程（SHG+SFG）による第三高調波発生（THG）は、以下の結合波方程式系で記述される。
$$\frac{d A_1}{dz} = i \kappa(z) \left[ A_2 A_1^* e^{i\Delta k_1 z} + A_3 A_2^* e^{i\Delta k_2 z} \right] \\ \frac{d A_2}{dz} = i \, 2\kappa(z) \left[ \frac{1}{2} A_1^2 e^{-i\Delta k_1 z} + A_3 A_1^* e^{i\Delta k_2 z} \right] \\ \frac{d A_3}{dz} = i \, 3\kappa(z) \left[ A_1 A_2 e^{-i\Delta k_2 z} \right]$$

ここで、$\boldsymbol{A}(z)$ は各波の複素振幅ベクトル、$\kappa(z)$ は結合係数、$\Delta k_j$ は位相不整合量を表す。

振幅$A_j$を「$|A_j|^2 \propto$ 光強度」と定義しているため、エネルギー保存則は単純な和の形式 $\sum |A_j|^2 = I_{const}$ で与えられる。

#### **2. 正準形式**
正準変換により方程式の $z$ 依存性を除去する。
$$\boldsymbol{B}(z) = e^{i\boldsymbol{L}z} \boldsymbol{A}(z) \quad \text{where} \quad \boldsymbol{L} = \begin{pmatrix} 0 & 0 & 0 \\ 0 & \Delta k_1 & 0 \\ 0 & 0 & \Delta k_1 + \Delta k_2 \end{pmatrix}$$

$\boldsymbol{B}$が従う運動方程式はあるハミルトニアン $K$ を用いて以下の正準形式で記述される。

$$\frac{d\boldsymbol{B}}{dz} = i \boldsymbol{J} \nabla_{\boldsymbol{B}^*} K(\boldsymbol{B}, \boldsymbol{B}^*) \quad \text{where} \quad \boldsymbol{J} = \text{diag}(1, 2, 3)$$

ハミルトニアンの具体的な形は以下で与えられる。

$$K(\boldsymbol{B}, \boldsymbol{B}^*) = \underbrace{\frac{\kappa(z)}{2} \left( B_1^2 B_2^* + (B_1^*)^2 B_2 \right) + \kappa(z) \left( B_1 B_2 B_3^* + B_1^* B_2^* B_3 \right)}_{K_{nl}} + \underbrace{\frac{\Delta k_1}{2} |B_2|^2 + \frac{\Delta k_1 + \Delta k_2}{3} |B_3|^2}_{K_{lin}}$$

APPLNでは、位相不整合 $\Delta k$ を補償するため、周期を $\Lambda \approx 2\pi/\Delta k$ の周辺として非線形係数 $\kappa(z)$ の符号を反転させる。$\kappa(z)$ は系全体としては$\Delta k$に近いstiffnessだが、各ドメインでは定数である。

#### **3. 積分形式**
ハミルトニアン $K$ を非線形項を生成する $K_{nl}$ と線形項を生成する $K_{lin}$ に分割すると、運動方程式は次のように書き換えられる。
$$\frac{d\boldsymbol{B}}{dz} = i \boldsymbol{J} \nabla_{\boldsymbol{B}^*} (K_{lin} + K_{nl}) = i \boldsymbol{J} \nabla_{\boldsymbol{B}^*} K_{lin} + i \boldsymbol{J} \nabla_{\boldsymbol{B}^*} K_{nl}$$

ここで、各項を計算すると、
$$i \boldsymbol{J} \nabla_{\boldsymbol{B}^*} K_{lin} = i \begin{pmatrix} 0 & 0 & 0 \\ 0 & \Delta k_1 & 0 \\ 0 & 0 & \Delta k_1 + \Delta k_2 \end{pmatrix} \begin{pmatrix} B_1 \\ B_2 \\ B_3 \end{pmatrix} \equiv i \boldsymbol{L} \boldsymbol{B}$$
$$i \boldsymbol{J} \nabla_{\boldsymbol{B}^*} K_{nl} = i \kappa(z) \begin{pmatrix} B_1^* B_2 + B_2^* B_3 \\ B_1^2 + 2 B_1^* B_3 \\ 3 B_1 B_2 \end{pmatrix} \equiv i \boldsymbol{N}(\boldsymbol{B}, \boldsymbol{B}^*)$$

となり、運動方程式は線形項と非線形項に分離された形式で表現できる。
$$\frac{d\boldsymbol{B}}{dz} = i \left( \boldsymbol{L}\boldsymbol{B} + \boldsymbol{N}(\boldsymbol{B}, \boldsymbol{B}^*) \right)$$

この方程式の厳密解は以下のようになる。

$$\boldsymbol{B}(z_n+h) = e^{i\boldsymbol{L}h} \boldsymbol{B}(z_n) + i \int_0^h e^{i\boldsymbol{L}(h-\tau')} \boldsymbol{N}(\boldsymbol{B}(z_n+\tau'), \boldsymbol{B}^*(z_n+\tau')) d\tau'$$

#### **4. 近似を用いた解析解**
微小である$\kappa$が乗算されている$K_{nl}$を無視して運動方程式を解くことで得られる、$\boldsymbol{B}(z_n + \tau')=e^{i\boldsymbol{L}\tau'}\boldsymbol{B}(z_n)$を用いて非線形項を解析的に積分する近似を用いると、単一ドメインの伝搬を解析的に計算できる。

1.  **積分関数の定義:**
    ドメインパラメータ $P_n$ を用いて、積分関数 $\Phi$ を以下のように定義する。
    $$
    \phi(\omega, h) = \begin{cases}
    \frac{e^{i\omega h} - 1}{i\omega} & (\omega \neq 0) \\
    h & (\omega = 0)
    \end{cases}
    $$

2.  **非線形発展項の計算:**
    ステップ内の非線形効果による状態変化ベクトル $\Delta \boldsymbol{B}_{nl}$ を、以下の解析解を用いて算出する。

$$\Delta B_{nl, 1} = i\kappa_n e^{il_1 h_n} \left[ B_{1n}^* B_{2n} \phi(l_2-l_1-l_1, h_n) + B_{2n}^* B_{3n} \phi(l_3-l_2-l_1, h_n) \right] \\ \Delta B_{nl, 2} = i\kappa_n e^{il_2 h_n} \left[ B_{1n}^2 \phi(2l_1-l_2, h_n) + 2 B_{1n}^* B_{3n} \phi(l_3-l_1-l_2, h_n) \right] \\ \Delta B_{nl, 3} = i \, 3\kappa_n e^{il_3 h_n} \left[ B_{1n} B_{2n} \phi(l_1+l_2-l_3, h_n) \right]$$

3.  **状態の予測:**
    線形発展と非線形発展を組み合わせ、ステップ終了時の状態ベクトル $\boldsymbol{B}_{pred}$ を算出する。

$$\boldsymbol{B}_{pred} = e^{i\boldsymbol{l}h_n} \boldsymbol{B}_n + \Delta \boldsymbol{B}_{nl}$$

## 損失を含むSHGの連立ODE
### 結合波方程式

$$\frac{dA_s}{dz} = -j\sqrt{\eta_o}A_p^2\exp(-j\Delta\beta z) - \frac{\alpha_s}{2}A_s$$

$$\frac{dA_p}{dz} = -j\sqrt{\eta_o}A_p^*A_s\exp(j\Delta\beta z) - \frac{\alpha_p}{2}A_p$$

### 各変数の意味

| 変数 | 意味 |
| :--- | :--- |
| `As`, `Ap` | それぞれ信号光（Signal）、ポンプ光（Pump）の複素振幅 |
| `z` | 光の伝搬距離 |
| `η₀` | 非線形結合係数。波長変換の効率を決定する。 |
| `Δβ` | 位相不整合量。結合する波の位相速度のずれを示す。 |
| `αs`, `αp` | 信号光とポンプ光の伝搬損失係数。デバイス固有の定数。 |

### APPLNのドメインと非線形結合係数 `η₀`

APPLN構造では、結晶の分極の向きを反転させた区画が並んでいます。この**分極の向きが同じ一つの連続した区間**を「ドメイン」と呼びます。

非線形結合係数 `η₀` は、物質の非線形光学定数（`d_eff`）に比例します。ドメイン内では分極の向きが一定であるため、`d_eff` が定数となります。

したがって、**一つのドメイン内では `η₀` は一定の値**をとります。ドメインの境界を越えると分極が反転し、`η₀` の符号が反転します。

# Your Task
上はdeepthinkが発見したExponential Integratorを工夫した計算方法です。3波の結合波方程式において、従来のRK4よりも高速かつ高精度にCWEsを計算できるようになった。

損失を含むSHGのODEにおいても、同様の手法でRK4より良い計算が可能か考えている。
