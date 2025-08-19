## **研究目的**
APPLNの高精度かつ高速なシミュレーションをする

### **理論**
#### **1. 結合波方程式**
カスケード二次非線形過程（SHG+SFG）による第三高調波発生（THG）は、以下の結合波方程式系で記述される。
$$
\begin{aligned}
\frac{d A_1}{dz} &= i \kappa(z) \left[ A_2 A_1^* e^{i\Delta k_1 z} + A_3 A_2^* e^{i\Delta k_2 z} \right] \\
\frac{d A_2}{dz} &= i \, 2\kappa(z) \left[ \frac{1}{2} A_1^2 e^{-i\Delta k_1 z} + A_3 A_1^* e^{i\Delta k_2 z} \right] \\
\frac{d A_3}{dz} &= i \, 3\kappa(z) \left[ A_1 A_2 e^{-i\Delta k_2 z} \right]
\end{aligned}
$$

ここで、$\boldsymbol{A}(z)$ は各波の複素振幅ベクトル、$\kappa(z)$ は結合係数、$\Delta k_j$ は位相不整合量を表す。

全光強度の保存に対応する$\sum |A_j^2| = I_{const}$が保存量となる。これは、振幅$A_j$を二乗が光強度に比例するように定義しているためである。

#### **2. 正準形式**
正準変換により方程式の $z$ 依存性を除去する。
$$\boldsymbol{B}(z) = e^{i\boldsymbol{L}z} \boldsymbol{A}(z) \quad \text{where} \quad \boldsymbol{L} = \begin{pmatrix} 0 & 0 & 0 \\ 0 & \Delta k_1 & 0 \\ 0 & 0 & \Delta k_1 + \Delta k_2 \end{pmatrix}$$

$\boldsymbol{B}$が従う運動方程式はあるハミルトニアン $K$ を用いて以下の正準形式で記述される。

$$
\frac{d\boldsymbol{B}}{dz} = i \boldsymbol{J} \nabla_{\boldsymbol{B}^*} K(\boldsymbol{B}, \boldsymbol{B}^*) 
\quad \text{where} \quad 
\boldsymbol{J} = \text{diag}(1, 2, 3)
$$

ハミルトニアンの具体的な形は以下で与えられる。

$$K(\boldsymbol{B}, \boldsymbol{B}^*) = \underbrace{\frac{\kappa(z)}{2} \left( B_1^2 B_2^* + (B_1^*)^2 B_2 \right) + \kappa(z) \left( B_1 B_2 B_3^* + B_1^* B_2^* B_3 \right)}_{K_{nl}} + \underbrace{\frac{\Delta k_1}{2} |B_2|^2 + \frac{\Delta k_1 + \Delta k_2}{3} |B_3|^2}_{K_{lin}}$$

APPLNでは、位相不整合 $\Delta k$ を補償するため、周期を $\Lambda \approx 2\pi/\Delta k$ の周辺として非線形係数 $\kappa(z)$ の符号を反転させる。$\kappa(z)$ は系全体としては$\Delta k$に近いstiffnessだが、各ドメインでは定数である。

ドメイン境界を数値計算のステップ境界とするならば、内部では $K$ は $z$ を陽に含まない自律ハミルトニアンとなり、系の保存量（$\frac{dK}{dz}=0$）となる。

#### **3. 積分形式**
ハミルトニアン $K$ を非線形項を生成する $K_{nl}$ と線形項を生成する $K_{lin}$ に分割すると、運動方程式は次のように書き換えられる。
$$\frac{d\boldsymbol{B}}{dz} = i \boldsymbol{J} \nabla_{\boldsymbol{B}^*} (K_{lin} + K_{nl}) = i \boldsymbol{J} \nabla_{\boldsymbol{B}^*} K_{lin} + i \boldsymbol{J} \nabla_{\boldsymbol{B}^*} K_{nl}$$

ここで、各項を計算すると、
$$i \boldsymbol{J} \nabla_{\boldsymbol{B}^*} K_{lin} = i \begin{pmatrix} 0 & 0 & 0 \\ 0 & \Delta k_1 & 0 \\ 0 & 0 & \Delta k_1 + \Delta k_2 \end{pmatrix} \begin{pmatrix} B_1 \\ B_2 \\ B_3 \end{pmatrix} \equiv i \boldsymbol{L} \boldsymbol{B}$$
$$i \boldsymbol{J} \nabla_{\boldsymbol{B}^*} K_{nl} = i \kappa(z) \begin{pmatrix} B_1^* B_2 + B_2^* B_3 \\ B_1^2 + 2 B_1^* B_3 \\ 3 B_1 B_2 \end{pmatrix} \equiv i \boldsymbol{N}(\boldsymbol{B}, \boldsymbol{B}^*)$$

となり、運動方程式は線形項と非線形項に分離された形式で表現できる。
$$
\frac{d\boldsymbol{B}}{dz} = i \left( \boldsymbol{L}\boldsymbol{B} + \boldsymbol{N}(\boldsymbol{B}, \boldsymbol{B}^*) \right)
$$

この方程式の厳密解は以下のようになる。

$$\boldsymbol{B}(z_n+h) = e^{i\boldsymbol{L}h} \boldsymbol{B}(z_n) + i \int_0^h e^{i\boldsymbol{L}(h-\tau')} \boldsymbol{N}(\boldsymbol{B}(z_n+\tau'), \boldsymbol{B}^*(z_n+\tau')) d\tau'$$

## 数値シミュレーション

### **EI Predictor Scheme**

#### **1. 概要**
状態ベクトル $\boldsymbol{B}_n$ をドメイン長 $h_n$ だけ時間発展させ、予測状態 $\boldsymbol{B}_{pred}$ を算出する。

微小である$\kappa$が乗算されている$K_{nl}$を無視して運動方程式を解くことで得られる、$\boldsymbol{B}(z_n + \tau')=e^{i\boldsymbol{L}h}\boldsymbol{B}(z_n)$を用いると、非線形項を解析的に積分できる。

#### **2. 入力**
* ステップ開始時の状態ベクトル: $\boldsymbol{B}_n = (B_{1n}, B_{2n}, B_{3n})^T$
* 現在のドメインパラメータ: $P_n = \{h_n, \kappa_n\}$

#### **3. 出力**
* 予測された状態ベクトル: $\boldsymbol{B}_{pred}$

#### **4. 処理手順**

1.  **積分関数の定義:**
    ドメインパラメータ $P_n$ を用いて、積分関数 $\Phi$ を以下のように定義する。
    $$
    \Phi(\Omega, h) = \begin{cases}
    \frac{e^{i\Omega h} - 1}{i\Omega} & (\Omega \neq 0) \\
    h & (\Omega = 0)
    \end{cases}
    $$

2.  **非線形発展項の計算:**
    ステップ内の非線形効果による状態変化ベクトル $\Delta \boldsymbol{B}_{NL}$ を、以下の解析解を用いて算出する。

    $$
    \begin{aligned}
    \Delta B_{NL, 1} &= i\kappa_n e^{iL_1 h_n} \left[ B_{1n}^* B_{2n} \Phi(L_2-L_1-L_1, h_n) + B_{2n}^* B_{3n} \Phi(L_3-L_2-L_1, h_n) \right] \\
    \Delta B_{NL, 2} &= i\kappa_n e^{iL_2 h_n} \left[ B_{1n}^2 \Phi(2L_1-L_2, h_n) + 2 B_{1n}^* B_{3n} \Phi(L_3-L_1-L_2, h_n) \right] \\
    \Delta B_{NL, 3} &= i \, 3\kappa_n e^{iL_3 h_n} \left[ B_{1n} B_{2n} \Phi(L_1+L_2-L_3, h_n) \right]
    \end{aligned}
    $$

3.  **状態の予測:**
    線形発展と非線形発展を組み合わせ、ステップ終了時の状態ベクトル $\boldsymbol{B}_{pred}$ を算出する。

    $$
    \boldsymbol{B}_{pred} = e^{i\boldsymbol{L}h_n} \boldsymbol{B}_n + \Delta \boldsymbol{B}_{NL}
    $$

# Your Task
上記は今回の研究で発見したcwesの高精度かつ高速なシミュレーションを可能とするschemeです。

正直なんでここまでうまくいくのかよくわかってない。
非線形項を駆動する、まるで重力のような力を一定と仮定していることや、$\kappa$が小さいとき成立しそうであることなど、何となくの感想は抱いているが、本当に合っているのかわからない。しかし結果として、実際に高速かつ高精度の計算に成功している状態です。

計算結果を見ると、**非線形項は$\Delta k$と同じぐらいstiffです**、**B(z)もstiffです**、小刻みに振動する感じになっていることがわかる。これは位相整合してないQPMデバイスの振幅の発展をかんがえてわかる通りです。

**非線形項がstiffであるため、ETDRK4は意味を成しません**、絶対に勘違いしないでください。今回新しいのは、stiffである非線形項で、$B(z_n + \tau')$に対して、**線形項のみで時間発展させるという近似**の結果解析解が導出でき、それも加味して再度ちゃんとB(z_n + h)を計算するという方法で、大成功している点です。

積分形式までは恒等変形だからいいんだけど、そっから線形項のみを考慮したBの発展を用いて非線形項を解析計算することによって、B(z_n + h)がとても高精度に計算できることについて、**エレガントな解説というか理由を考えて**、それを理論的に解説してほしい。数学の答案だと思って、ふわっと伝えるじゃなくて、妥当性を証明してほしい。
