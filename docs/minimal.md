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

$$K(\boldsymbol{B}, \boldsymbol{B}^*) = \underbrace{\frac{\kappa(z)}{2} \left( B_1^2 B_2^* + (B_1^*)^2 B_2 \right) + \kappa(z) \left( B_1 B_2 B_3^* + B_1^* B_2^* B_3 \right)}_{K_{NL}} + \underbrace{\frac{\Delta k_1}{2} |B_2|^2 + \frac{\Delta k_1 + \Delta k_2}{3} |B_3|^2}_{K_{LIN}}$$

APPLNでは、位相不整合 $\Delta k$ を補償するため、周期を $\Lambda \approx 2\pi/\Delta k$ の周辺として非線形係数 $\kappa(z)$ の符号を反転させる。$\kappa(z)$ は系全体としては$\Delta k$に近いstiffnessだが、各ドメインでは定数である。

#### **3. 積分形式**
ハミルトニアン $K$ を非線形項を生成する $K_{NL}$ と線形項を生成する $K_{LIN}$ に分割すると、運動方程式は次のように書き換えられる。
$$\frac{d\boldsymbol{B}}{dz} = i \boldsymbol{J} \nabla_{\boldsymbol{B}^*} (K_{LIN} + K_{NL}) = i \boldsymbol{J} \nabla_{\boldsymbol{B}^*} K_{LIN} + i \boldsymbol{J} \nabla_{\boldsymbol{B}^*} K_{NL}$$

ここで、各項を計算すると、
$$i \boldsymbol{J} \nabla_{\boldsymbol{B}^*} K_{LIN} = i \begin{pmatrix} 0 & 0 & 0 \\ 0 & \Delta k_1 & 0 \\ 0 & 0 & \Delta k_1 + \Delta k_2 \end{pmatrix} \begin{pmatrix} B_1 \\ B_2 \\ B_3 \end{pmatrix} \equiv i \boldsymbol{L} \boldsymbol{B}$$
$$i \boldsymbol{J} \nabla_{\boldsymbol{B}^*} K_{NL} = i \kappa(z) \begin{pmatrix} B_1^* B_2 + B_2^* B_3 \\ B_1^2 + 2 B_1^* B_3 \\ 3 B_1 B_2 \end{pmatrix} \equiv i \boldsymbol{N}(\boldsymbol{B}, \boldsymbol{B}^*)$$

となり、運動方程式は線形項と非線形項に分離された形式で表現できる。
$$\frac{d\boldsymbol{B}}{dz} = i \left( \boldsymbol{L}\boldsymbol{B} + \boldsymbol{N}(\boldsymbol{B}, \boldsymbol{B}^*) \right)$$

この方程式の厳密解は以下のようになる。

$$\boldsymbol{B}(z_n+h) = e^{i\boldsymbol{L}h} \boldsymbol{B}(z_n) + i \int_0^h e^{i\boldsymbol{L}(h-\tau')} \boldsymbol{N}(\boldsymbol{B}(z_n+\tau'), \boldsymbol{B}^*(z_n+\tau')) d\tau'$$

#### **4. 近似を用いた解析解**
微小である$\kappa$が乗算されている$K_{NL}$を無視して運動方程式を解くことで得られる、$\boldsymbol{B}(z_n + \tau')=e^{i\boldsymbol{L}\tau'}\boldsymbol{B}(z_n)$を用いて非線形項を解析的に積分する近似を用いると、単一ドメインの伝搬を解析的に計算できる。

1.  **積分関数の定義:**
    ドメインパラメータ $P_n$ を用いて、積分関数 $\Phi$ を以下のように定義する。
    $$
    \phi(\omega, h) = \begin{cases}
    \frac{e^{i\omega h} - 1}{i\omega} & (\omega \neq 0) \\
    h & (\omega = 0)
    \end{cases}
    $$

2.  **非線形発展項の計算:**
    ステップ内の非線形効果による状態変化ベクトル $\Delta \boldsymbol{B}_{NL}$ を、以下の解析解を用いて算出する。

$$B_{NL, 1} = i\kappa_n e^{il_1 h_n} \left[ B_{1n}^* B_{2n} \phi(l_2-l_1-l_1, h_n) + B_{2n}^* B_{3n} \phi(l_3-l_2-l_1, h_n) \right] \\ B_{NL, 2} = i\kappa_n e^{il_2 h_n} \left[ B_{1n}^2 \phi(2l_1-l_2, h_n) + 2 B_{1n}^* B_{3n} \phi(l_3-l_1-l_2, h_n) \right] \\ B_{NL, 3} = i \, 3\kappa_n e^{il_3 h_n} \left[ B_{1n} B_{2n} \phi(l_1+l_2-l_3, h_n) \right]$$

3.  **状態の予測:**
    線形発展と非線形発展を組み合わせ、ステップ終了時の状態ベクトル $\boldsymbol{B}_{pred}$ を算出する。

$$\boldsymbol{B}_{pred} = e^{i\boldsymbol{l}h_n} \boldsymbol{B}_n + \boldsymbol{B}_{NL}$$
