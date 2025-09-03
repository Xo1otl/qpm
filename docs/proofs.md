## **研究目的**
APPLNの高精度かつ高速なシミュレーションをする

## **理論**
#### **1. 結合波方程式**
カスケード二次非線形過程（SHG+SFG）による第三高調波発生（THG）は、以下の結合波方程式系で記述される。
$$\frac{d A_1}{dz} = i \kappa(z) \left[ A_2 A_1^* e^{i\Delta k_1 z} + A_3 A_2^* e^{i\Delta k_2 z} \right] \\ \frac{d A_2}{dz} = i \, 2\kappa(z) \left[ \frac{1}{2} A_1^2 e^{-i\Delta k_1 z} + A_3 A_1^* e^{i\Delta k_2 z} \right] \\ \frac{d A_3}{dz} = i \, 3\kappa(z) \left[ A_1 A_2 e^{-i\Delta k_2 z} \right]$$

ここで、$\boldsymbol{A}(z)$ は各波の複素振幅ベクトル、$\kappa(z)$ は結合係数、$\Delta k_j$ は位相不整合量を表す。

全光強度の保存に対応する$\sum |A_j^2| = I_{const}$が保存量となる。これは、振幅$A_j$を二乗が光強度に比例するように定義しているためである。

#### **2. 正準形式**
正準変換により方程式の $z$ 依存性を除去する。
$$\boldsymbol{B}(z) = e^{i\boldsymbol{L}z} \boldsymbol{A}(z) \quad \text{where} \quad \boldsymbol{L} = \begin{pmatrix} 0 & 0 & 0 \\ 0 & \Delta k_1 & 0 \\ 0 & 0 & \Delta k_1 + \Delta k_2 \end{pmatrix}$$

$\boldsymbol{B}$が従う運動方程式はあるハミルトニアン $K$ を用いて以下の正準形式で記述される。

$$\frac{d\boldsymbol{B}}{dz} = i \boldsymbol{J} \nabla_{\boldsymbol{B}^*} K(\boldsymbol{B}, \boldsymbol{B}^*) \quad \text{where} \quad \boldsymbol{J} = \text{diag}(1, 2, 3)$$

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
$$\frac{d\boldsymbol{B}}{dz} = i \left( \boldsymbol{L}\boldsymbol{B} + \boldsymbol{N}(\boldsymbol{B}, \boldsymbol{B}^*) \right)$$

この方程式の厳密解は以下のようになる。

$$\boldsymbol{B}(z_n+h) = e^{i\boldsymbol{L}h} \boldsymbol{B}(z_n) + i \int_0^h e^{i\boldsymbol{L}(h-\tau')} \boldsymbol{N}(\boldsymbol{B}(z_n+\tau'), \boldsymbol{B}^*(z_n+\tau')) d\tau'$$

## 数値シミュレーション

### **EI Predictor Scheme**

#### **1. 概要**
状態ベクトル $\boldsymbol{B}_n$ をドメイン長 $h_n$ だけ時間発展させ、予測状態 $\boldsymbol{B}_{pred}$ を算出する。

微小である$\kappa$が乗算されている$K_{nl}$を無視して運動方程式を解くことで得られる、$\boldsymbol{B}(z_n + \tau')=e^{i\boldsymbol{L}\tau'}\boldsymbol{B}(z_n)$を用いると、非線形項を解析的に積分できる。

#### **2. 入力**
* ステップ開始時の状態ベクトル: $\boldsymbol{B}_n = (B_{1n}, B_{2n}, B_{3n})^T$
* 現在のドメインパラメータ: $P_n = \{h_n, \kappa_n\}$

#### **3. 出力**
* 予測された状態ベクトル: $\boldsymbol{B}_{pred}$

#### **4. 処理手順**

1.  **積分関数の定義:**
    ドメインパラメータ $P_n$ を用いて、積分関数 $\Phi$ を以下のように定義する。
    $$
    \phi(\omega, h) = \begin{cases}
    \frac{e^{i\omega h} - 1}{i\omega} & (\omega \neq 0) \\
    h & (\omega = 0)
    \end{cases}
    $$

2.  **非線形発展項の計算:**
    ステップ内の非線形効果による状態変化ベクトル $\delta \boldsymbol{b}_{nl}$ を、以下の解析解を用いて算出する。

$$\delta b_{nl, 1} = i\kappa_n e^{il_1 h_n} \left[ b_{1n}^* b_{2n} \phi(l_2-l_1-l_1, h_n) + b_{2n}^* b_{3n} \phi(l_3-l_2-l_1, h_n) \right] \\ \delta b_{nl, 2} = i\kappa_n e^{il_2 h_n} \left[ b_{1n}^2 \phi(2l_1-l_2, h_n) + 2 b_{1n}^* b_{3n} \phi(l_3-l_1-l_2, h_n) \right] \\ \delta b_{nl, 3} = i \, 3\kappa_n e^{il_3 h_n} \left[ b_{1n} b_{2n} \phi(l_1+l_2-l_3, h_n) \right]$$

3.  **状態の予測:**
    線形発展と非線形発展を組み合わせ、ステップ終了時の状態ベクトル $\boldsymbol{b}_{pred}$ を算出する。

$$\boldsymbol{b}_{pred} = e^{i\boldsymbol{l}h_n} \boldsymbol{b}_n + \delta \boldsymbol{b}_{nl}$$

---

### **i-k projection corrector scheme**

#### **1. 概要**
予測された状態ベクトル $\boldsymbol{b}_{pred}$ を、不変量である全光強度 $i_{const}$ とドメイン内ハミルトニアン $k_n$ の拘束条件を満たす多様体上へ射影し、補正された状態ベクトル $\boldsymbol{b}_{corr}$ を算出する。

#### **2. 入力**
* 予測された状態ベクトル: $\boldsymbol{b}_{pred}$
* 拘束条件の目標値:
    * 全光強度: $i_{const}$
    * ドメイン内ハミルトニアン: $k_n$
* 現在のドメインパラメータ: $p_n = \{h_n, \kappa_n\}$

#### **3. 出力**
* 補正された状態ベクトル: $\boldsymbol{b}_{corr}$

#### **4. 処理手順**

1.  **複素勾配ベクトルの算出:**
    不変量 $i, k$ の $\boldsymbol{b}^*$ に対する勾配を、$\boldsymbol{b}_{pred}$ とドメインパラメータ $p_n$ を用いて算出する。
    * $\nabla_{\boldsymbol{b}^*} i = \boldsymbol{b}_{pred}$
    * $\nabla_{\boldsymbol{b}^*} k = -i \boldsymbol{j}^{-1} \frac{d\boldsymbol{b}}{dz} \bigg|_{\boldsymbol{b}=\boldsymbol{b}_{pred}}$

    ここで、$\frac{d\boldsymbol{b}}{dz}$ はドメインパラメータ $p_n$ を用いた運動方程式の右辺を表す。

2.  **誤差の算出:**
    予測値と目標値との差分を算出する。
    * $e_i = \boldsymbol{b}_{pred}^\dagger \boldsymbol{b}_{pred} - i_{const}$
    * $e_k = k(\boldsymbol{b}_{pred}, \kappa_n) - k_n$

3.  **線形システムの構築:**
    ラグランジュ未定乗数 $\boldsymbol{\lambda} = (\lambda_i, \lambda_k)^t$ を求めるための 2x2 行列 $\boldsymbol{m}$ を構築する。
$$\boldsymbol{m} = \begin{pmatrix} (\nabla_{\boldsymbol{b}^*} i)^\dagger (\nabla_{\boldsymbol{b}^*} i) & \text{re}[(\nabla_{\boldsymbol{b}^*} i)^\dagger (\nabla_{\boldsymbol{b}^*} k)] \\ \text{re}[(\nabla_{\boldsymbol{b}^*} k)^\dagger (\nabla_{\boldsymbol{b}^*} i)] & (\nabla_{\boldsymbol{b}^*} k)^\dagger (\nabla_{\boldsymbol{b}^*} k) \end{pmatrix}$$

4.  **未定乗数の求解:**
    連立一次方程式 $\boldsymbol{m} \boldsymbol{\lambda} = \frac{1}{2}(e_i, e_k)^t$ を解き、$\boldsymbol{\lambda}$ を求める。

5.  **状態の補正:**
    求めた $\boldsymbol{\lambda}$ を用いて $\boldsymbol{b}_{pred}$ を補正する。
    $$\boldsymbol{b}_{corr} = \boldsymbol{b}_{pred} - \lambda_i (\nabla_{\boldsymbol{b}^*} i) - \lambda_k (\nabla_{\boldsymbol{b}^*} k)$$

# 導出

### **1. 変換の定義**
$$\boldsymbol{a}(z) = e^{-i\boldsymbol{l}z} \boldsymbol{b}(z) \quad \text{where} \quad \boldsymbol{l} = \begin{pmatrix} 0 & 0 & 0 \\ 0 & \delta k_1 & 0 \\ 0 & 0 & \delta k_1 + \delta k_2 \end{pmatrix}$$

### **2. 運動方程式の変換**
$$\frac{d\boldsymbol{a}}{dz} = -i\boldsymbol{l} e^{-i\boldsymbol{l}z} \boldsymbol{b}(z) + e^{-i\boldsymbol{l}z} \frac{d\boldsymbol{b}}{dz}$$
$$e^{-i\boldsymbol{l}z} \frac{d\boldsymbol{b}}{dz} = \frac{d\boldsymbol{a}}{dz} + i\boldsymbol{l} e^{-i\boldsymbol{l}z} \boldsymbol{b}(z)$$
$$\frac{d\boldsymbol{b}}{dz} = e^{i\boldsymbol{l}z} \frac{d\boldsymbol{a}}{dz} + i\boldsymbol{l} \boldsymbol{b}(z)$$

### **3. 各成分の導出**
#### **第一式 ($b_1$)**
$$\frac{db_1}{dz} = \frac{da_1}{dz} \quad (\because l_1=0) \\ = i \kappa \left[ (e^{-il_2z}b_2)(e^{il_1z}b_1^*)e^{i\delta k_1 z} + (e^{-il_3z}b_3)(e^{il_2z}b_2^*)e^{i\delta k_2 z} \right] \\ = i \kappa \left[ e^{-i\delta k_1 z} b_2 b_1^* e^{i\delta k_1 z} + e^{-i(\delta k_1+\delta k_2)z} b_3 e^{i\delta k_1 z} b_2^* e^{i\delta k_2 z} \right] \\ = i \kappa \left[ b_1^* b_2 + b_2^* b_3 \right]$$

#### **第二式 ($b_2$)**
$$\frac{db_2}{dz} = e^{il_2z} \frac{da_2}{dz} + il_2 b_2 \\ = e^{i\delta k_1 z} \left( i \cdot 2\kappa \left[ \frac{1}{2} a_1^2 e^{-i\delta k_1 z} + a_3 a_1^* e^{i\delta k_2 z} \right] \right) + i\delta k_1 b_2 \\ = i \kappa \left[ a_1^2 e^{i(\delta k_1 - \delta k_1)z} + 2a_3 a_1^* e^{i(\delta k_1 + \delta k_2)z} \right] + i\delta k_1 b_2 \\ = i \kappa \left[ b_1^2 + 2(e^{-il_3z}b_3)(e^{il_1z}b_1^*) e^{i(\delta k_1 + \delta k_2)z} \right] + i\delta k_1 b_2 \\ = i \kappa \left[ b_1^2 + 2 e^{-i(\delta k_1+\delta k_2)z} b_3 b_1^* e^{i(\delta k_1 + \delta k_2)z} \right] + i\delta k_1 b_2 \\ = i \kappa \left[ b_1^2 + 2 b_1^* b_3 \right] + i\delta k_1 b_2$$

#### **第三式 ($b_3$)**
$$\frac{db_3}{dz} = e^{il_3z} \frac{da_3}{dz} + il_3 b_3 = e^{i(\delta k_1+\delta k_2)z} \left( i \cdot 3\kappa \left[ a_1 a_2 e^{-i\delta k_2 z} \right] \right) + i(\delta k_1+\delta k_2)b_3 = i \cdot 3\kappa \left[ a_1 a_2 e^{i(\delta k_1+\delta k_2-\delta k_2)z} \right] + i(\delta k_1+\delta k_2)b_3 = i \cdot 3\kappa \left[ (e^{-il_1z}b_1)(e^{-il_2z}b_2) e^{i\delta k_1 z} \right] + i(\delta k_1+\delta k_2)b_3 = i \cdot 3\kappa \left[ b_1 e^{-i\delta k_1 z} b_2 e^{i\delta k_1 z} \right] + i(\delta k_1+\delta k_2)b_3 = i \cdot 3\kappa \left[ b_1 b_2 \right] + i(\delta k_1+\delta k_2)b_3$$

### **4. 最終形式**
以上の結果をベクトル形式でまとめ、線形項と非線形項に分離する。
$$\frac{d}{dz} \begin{pmatrix} b_1 \\ b_2 \\ b_3 \end{pmatrix} = i \begin{pmatrix} 0 \\ \delta k_1 b_2 \\ (\delta k_1+\delta k_2)b_3 \end{pmatrix} + i \kappa \begin{pmatrix} b_1^* b_2 + b_2^* b_3 \\ b_1^2 + 2 b_1^* b_3 \\ 3 b_1 b_2 \end{pmatrix}$$
$$\frac{d\boldsymbol{b}}{dz} = i \left( \boldsymbol{l}\boldsymbol{b} + \boldsymbol{n}(\boldsymbol{b}, \boldsymbol{b}^*) \right)$$
ここで、
$$\boldsymbol{l}\boldsymbol{b} = \begin{pmatrix} 0 & 0 & 0 \\ 0 & \delta k_1 & 0 \\ 0 & 0 & \delta k_1 + \delta k_2 \end{pmatrix} \begin{pmatrix} b_1 \\ b_2 \\ b_3 \end{pmatrix}$$
$$\boldsymbol{n}(\boldsymbol{b}, \boldsymbol{b}^*) = \kappa(z) \begin{pmatrix} b_1^* b_2 + b_2^* b_3 \\ b_1^2 + 2 b_1^* b_3 \\ 3 b_1 b_2 \end{pmatrix}$$

承知いたしました。ハミルトニアンの導出について、数式のみで示します。

### **1. 正準形式（ハミルトン形式）の定義**
運動方程式が、あるハミルトニアン $k$ を用いて以下の形式で記述されると仮定する。
$$\frac{d b_j}{dz} = i \, j \frac{\partial k}{\partial b_j^*} \quad \implies \quad \frac{\partial k}{\partial b_j^*} = \frac{1}{i j} \frac{d b_j}{dz}$$

### **2. ハミルトニアン $k$ の各偏導関数の導出**
$\boldsymbol{b}$ の運動方程式から、$k$ の各偏導関数を求める。
$$\frac{\partial k}{\partial b_1^*} = \frac{1}{i} \frac{db_1}{dz} = \frac{1}{i} \left( i \kappa \left[ b_1^* b_2 + b_2^* b_3 \right] \right) = \kappa \left( b_1^* b_2 + b_2^* b_3 \right)$$
$$\frac{\partial k}{\partial b_2^*} = \frac{1}{2i} \frac{db_2}{dz} = \frac{1}{2i} \left( i \delta k_1 b_2 + i \kappa \left[ b_1^2 + 2 b_1^* b_3 \right] \right) = \frac{\delta k_1}{2} b_2 + \frac{\kappa}{2} b_1^2 + \kappa b_1^* b_3$$
$$\frac{\partial k}{\partial b_3^*} = \frac{1}{3i} \frac{db_3}{dz} = \frac{1}{3i} \left( i (\delta k_1 + \delta k_2) b_3 + i \cdot 3\kappa \left[ b_1 b_2 \right] \right) = \frac{\delta k_1 + \delta k_2}{3} b_3 + \kappa b_1 b_2$$

### **3. ハミルトニアン $k$ の構築**
上記の偏導関数を（$b_1^*, b_2^*, b_3^*$ について）積分し、すべての条件を矛盾なく満たすように項を組み合わせることで、ハミルトニアン $k$ が以下のように構築される。
$$k(\boldsymbol{b}, \boldsymbol{b}^*) = \frac{\kappa}{2} \left( b_1^2 b_2^* + (b_1^*)^2 b_2 \right) + \kappa \left( b_1 b_2 b_3^* + b_1^* b_2^* b_3 \right) + \frac{\delta k_1}{2} |b_2|^2 + \frac{\delta k_1 + \delta k_2}{3} |b_3|^2$$
これは、非線形項 $k_{nl}$ と線形項 $k_{lin}$ に分割できる。
$$k_{nl} = \frac{\kappa}{2} \left( b_1^2 b_2^* + (b_1^*)^2 b_2 \right) + \kappa \left( b_1 b_2 b_3^* + b_1^* b_2^* b_3 \right)$$
$$k_{lin} = \frac{\delta k_1}{2} |b_2|^2 + \frac{\delta k_1 + \delta k_2}{3} |b_3|^2$$
$$k = k_{nl} + k_{lin}$$

### **4. 検証**
構築した $k$ から運動方程式を再生成し、元の方程式と一致することを確認する。
$$i \cdot 1 \frac{\partial k}{\partial b_1^*} = i \kappa \left( b_1^* b_2 + b_2^* b_3 \right) = \frac{db_1}{dz}$$
$$i \cdot 2 \frac{\partial k}{\partial b_2^*} = i \cdot 2 \left( \frac{\kappa}{2} b_1^2 + \kappa b_1^* b_3 + \frac{\delta k_1}{2} b_2 \right) = i\kappa(b_1^2 + 2b_1^* b_3) + i\delta k_1 b_2 = \frac{db_2}{dz}$$
$$i \cdot 3 \frac{\partial k}{\partial b_3^*} = i \cdot 3 \left( \kappa b_1 b_2 + \frac{\delta k_1 + \delta k_2}{3} b_3 \right) = i \cdot 3\kappa b_1 b_2 + i(\delta k_1 + \delta k_2) b_3 = \frac{db_3}{dz}$$

$$\frac{d\boldsymbol{b}}{dz} = i \boldsymbol{l}\boldsymbol{b} + i\boldsymbol{n}(\boldsymbol{b}, \boldsymbol{b}^*)$$
$$\frac{d\boldsymbol{b}}{dz} - i\boldsymbol{l}\boldsymbol{b} = i\boldsymbol{n}(\boldsymbol{b}, \boldsymbol{b}^*)$$
$$e^{-i\boldsymbol{l}z} \left( \frac{d\boldsymbol{b}}{dz} - i\boldsymbol{l}\boldsymbol{b} \right) = i e^{-i\boldsymbol{l}z} \boldsymbol{n}(\boldsymbol{b}, \boldsymbol{b}^*)$$
$$\frac{d}{dz} \left( e^{-i\boldsymbol{l}z} \boldsymbol{b}(z) \right) = i e^{-i\boldsymbol{l}z} \boldsymbol{n}(\boldsymbol{b}(z), \boldsymbol{b}^*(z))$$
$$\int_{z_n}^{z_n+h} \frac{d}{d\tau} \left( e^{-i\boldsymbol{l}\tau} \boldsymbol{b}(\tau) \right) d\tau = i \int_{z_n}^{z_n+h} e^{-i\boldsymbol{l}\tau} \boldsymbol{n}(\boldsymbol{b}(\tau), \boldsymbol{b}^*(\tau)) d\tau$$
$$\left[ e^{-i\boldsymbol{l}\tau} \boldsymbol{b}(\tau) \right]_{z_n}^{z_n+h} = i \int_{z_n}^{z_n+h} e^{-i\boldsymbol{l}\tau} \boldsymbol{n}(\boldsymbol{b}(\tau), \boldsymbol{b}^*(\tau)) d\tau$$
$$e^{-i\boldsymbol{l}(z_n+h)} \boldsymbol{b}(z_n+h) - e^{-i\boldsymbol{l}z_n} \boldsymbol{b}(z_n) = i \int_{z_n}^{z_n+h} e^{-i\boldsymbol{l}\tau} \boldsymbol{n}(\boldsymbol{b}(\tau), \boldsymbol{b}^*(\tau)) d\tau$$
$$e^{-i\boldsymbol{l}(z_n+h)} \boldsymbol{b}(z_n+h) = e^{-i\boldsymbol{l}z_n} \boldsymbol{b}(z_n) + i \int_{z_n}^{z_n+h} e^{-i\boldsymbol{l}\tau} \boldsymbol{n}(\boldsymbol{b}(\tau), \boldsymbol{b}^*(\tau)) d\tau$$
$$\boldsymbol{b}(z_n+h) = e^{i\boldsymbol{l}(z_n+h)}e^{-i\boldsymbol{l}z_n} \boldsymbol{b}(z_n) + i e^{i\boldsymbol{l}(z_n+h)} \int_{z_n}^{z_n+h} e^{-i\boldsymbol{l}\tau} \boldsymbol{n}(\boldsymbol{b}(\tau), \boldsymbol{b}^*(\tau)) d\tau$$
$$\boldsymbol{b}(z_n+h) = e^{i\boldsymbol{l}h} \boldsymbol{b}(z_n) + i \int_{z_n}^{z_n+h} e^{i\boldsymbol{l}(z_n+h-\tau)} \boldsymbol{n}(\boldsymbol{b}(\tau), \boldsymbol{b}^*(\tau)) d\tau$$
変数変換 $\tau' = \tau - z_n$ を行うと ($d\tau' = d\tau$, 積分範囲は $0$ から $h$)
$$\boldsymbol{b}(z_n+h) = e^{i\boldsymbol{l}h} \boldsymbol{b}(z_n) + i \int_0^h e^{i\boldsymbol{l}(h-\tau')} \boldsymbol{n}(\boldsymbol{b}(z_n+\tau'), \boldsymbol{b}^*(z_n+\tau')) d\tau'$$


# 非線形項の解析解の導出

$$\delta \boldsymbol{b}_{nl} = i \int_0^{h_n} e^{i\boldsymbol{l}(h_n-\tau')} \boldsymbol{n}(\boldsymbol{b}(z_n+\tau'), \boldsymbol{b}^*(z_n+\tau')) d\tau'$$

$$\boldsymbol{b}(z_n+\tau') \approx e^{i\boldsymbol{l}\tau'} \boldsymbol{b}(z_n)$$

$$\boldsymbol{n}(\boldsymbol{b}(z_n+\tau'), \boldsymbol{b}^*(z_n+\tau')) \approx \kappa_n \begin{pmatrix} (e^{-il_1\tau'}b_{1n}^*)(e^{il_2\tau'}b_{2n}) + (e^{-il_2\tau'}b_{2n}^*)(e^{il_3\tau'}b_{3n}) \\ (e^{il_1\tau'}b_{1n})^2 + 2(e^{-il_1\tau'}b_{1n}^*)(e^{il_3\tau'}b_{3n}) \\ 3(e^{il_1\tau'}b_{1n})(e^{il_2\tau'}b_{2n}) \end{pmatrix} = \kappa_n \begin{pmatrix} e^{i(l_2-l_1)\tau'}b_{1n}^*b_{2n} + e^{i(l_3-l_2)\tau'}b_{2n}^*b_{3n} \\ e^{i2l_1\tau'}b_{1n}^2 + 2e^{i(l_3-l_1)\tau'}b_{1n}^*b_{3n} \\ 3e^{i(l_1+l_2)\tau'}b_{1n}b_{2n} \end{pmatrix}$$

$$\delta \boldsymbol{b}_{nl} = i \kappa_n \int_0^{h_n} \begin{pmatrix} e^{il_1(h_n-\tau')} & 0 & 0 \\ 0 & e^{il_2(h_n-\tau')} & 0 \\ 0 & 0 & e^{il_3(h_n-\tau')} \end{pmatrix} \begin{pmatrix} e^{i(l_2-l_1)\tau'}b_{1n}^*b_{2n} + e^{i(l_3-l_2)\tau'}b_{2n}^*b_{3n} \\ e^{i2l_1\tau'}b_{1n}^2 + 2e^{i(l_3-l_1)\tau'}b_{1n}^*b_{3n} \\ 3e^{i(l_1+l_2)\tau'}b_{1n}b_{2n} \end{pmatrix} d\tau'$$

$$= i \kappa_n \begin{pmatrix} e^{il_1h_n} \int_0^{h_n} [e^{i(l_2-2l_1)\tau'}b_{1n}^*b_{2n} + e^{i(l_3-l_2-l_1)\tau'}b_{2n}^*b_{3n}] d\tau' \\ e^{il_2h_n} \int_0^{h_n} [e^{i(2l_1-l_2)\tau'}b_{1n}^2 + 2e^{i(l_3-l_1-l_2)\tau'}b_{1n}^*b_{3n}] d\tau' \\ 3e^{il_3h_n} \int_0^{h_n} [e^{i(l_1+l_2-l_3)\tau'}b_{1n}b_{2n}] d\tau' \end{pmatrix}$$

$$= i \kappa_n \begin{pmatrix} e^{il_1h_n} [b_{1n}^*b_{2n}\phi(l_2-2l_1, h_n) + b_{2n}^*b_{3n}\phi(l_3-l_2-l_1, h_n)] \\ e^{il_2h_n} [b_{1n}^2\phi(2l_1-l_2, h_n) + 2b_{1n}^*b_{3n}\phi(l_3-l_1-l_2, h_n)] \\ 3e^{il_3h_n} [b_{1n}b_{2n}\phi(l_1+l_2-l_3, h_n)] \end{pmatrix}$$

$$\phi(\omega, h) = \begin{cases} \frac{e^{i\omega h} - 1}{i\omega} & (\omega \neq 0) h & (\omega = 0) \end{cases}$$