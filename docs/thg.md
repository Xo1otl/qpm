### **研究目的**
SHGとSFGによるカスケード二次非線形過程を用いたTHGのスペクトル(位相不整合量)分布のフラットトップ高効率化。

### **カスケードTHGを記述する結合波方程式**

本研究のシミュレーションは、cascaded second-order nonlinear processにおける三つの波(FW、SHW、THW)の相互作用を記述する結合波方程式に基づきます。以下に、理論的な出発点から数値計算に用いる形式までを段階的に示します。

#### **1. SVEAに基づく基本方程式 ($\chi^{(2)}$)**

非線形感受率 $\chi^{(2)}(z)$ で直接記述した基本方程式は以下の通りです。

$$\frac{\partial E_1}{\partial z} = i \frac{\omega_1^2}{2k_1 c^2} \chi^{(2)}(z) [E_2 E_1^* \exp(i\Delta k_1 z) + E_3 E_2^* \exp(i\Delta k_2 z)]$$

$$\frac{\partial E_2}{\partial z} = i \frac{\omega_2^2}{2k_2 c^2} \chi^{(2)}(z) [\frac{1}{2} E_1^2 \exp(-i\Delta k_1 z) + E_3 E_1^* \exp(i\Delta k_2 z)]$$

$$\frac{\partial E_3}{\partial z} = i \frac{\omega_3^2}{2k_3 c^2} \chi^{(2)}(z) [E_1 E_2 \exp(-i\Delta k_2 z)]$$

#### **2. 実用的な一般形 ($d_{eff}$)**

実効非線形係数 $d_{eff}$（ただし $\chi^{(2)} = 2d_{eff}$）を用いて上記の式を書き換えると、以下の物理的に直感的な形式が得られます。

$$\frac{\partial E_1}{\partial z} = i \frac{\omega_1 d_{eff}(z)}{n_1 c} \left[ E_2 E_1^* \exp(i\Delta k_1 z) + E_3 E_2^* \exp(i\Delta k_2 z) \right]$$

$$\frac{\partial E_2}{\partial z} = i \frac{\omega_2 d_{eff}(z)}{n_2 c} \left[ \frac{1}{2} E_1^2 \exp(-i\Delta k_1 z) + E_3 E_1^* \exp(i\Delta k_2 z) \right]$$

$$\frac{\partial E_3}{\partial z} = i \frac{\omega_3 d_{eff}(z)}{n_3 c} \left[ E_1 E_2 \exp(-i\Delta k_2 z) \right]$$

#### **3. 数値計算のための規格化形式**

数値シミュレーションのため、光強度に比例する複素振幅 $A_j$ と結合係数 $\kappa(z)$ を導入し、方程式を規格化します。

$$\frac{\partial A_1}{\partial z} = i \kappa(z) \left[ A_2 A_1^* e^{i\Delta k_1 z} + A_3 A_2^* e^{i\Delta k_2 z} \right]$$

$$\frac{\partial A_2}{\partial z} = i \, 2\kappa(z) \left[ \frac{1}{2} A_1^2 e^{-i\Delta k_1 z} + A_3 A_1^* e^{i\Delta k_2 z} \right]$$

$$\frac{\partial A_3}{\partial z} = i \, 3\kappa(z) \left[ A_1 A_2 e^{-i\Delta k_2 z} \right]$$

---

### **主要な変数とパラメータの定義**

* **複素振幅 $A_j(z)$**: $j=1, 2, 3$ はそれぞれFW、SHW、THWに対応。光強度 $I_j(z) = |A_j(z)|^2$ です。
* **結合係数 $\kappa(z)$**: $\kappa(z) \equiv d_{eff}(z) \omega_1 \sqrt{\frac{2}{n^3 c^3 \epsilon_0}}$。$d_{eff}(z)$ は実効非線形光学係数です。
* **位相不整合 $\Delta k_j$**:
    * $\Delta k_1 = k_2 - 2k_1$: SHG過程の位相不整合。
    * $\Delta k_2 = k_3 - k_2 - k_1$: SFG過程の位相不整合。
    * 波数 $k_j = n_j(\omega_j)\omega_j/c$ の計算では、材料の屈折率の波長分散 $n_j(\omega_j)$ を厳密に考慮する必要があります。

---

### **方程式の導出における主要な近似**

1.  **SVEA (Slowly Varying Envelope Approximation)**
2.  **平面波近似**
3.  **CW (連続波) または準CW近似**
4.  **無損失媒質**
5.  **同一直線伝播**
6.  **結合係数における分散無視**: 結合係数 $\kappa$ の導出において $n_j \approx n$ としますが、位相不整合 $\Delta k$ の計算では分散を厳密に扱います。
7.  **Kleinman対称性の仮定**: SHG過程とSFG過程で同じ実効非線形係数を用います。これは使用する波長域が媒質の吸収端から十分に離れている場合に成立する近似です。
8.  **三次非線形効果の無視**: このモデルは二次非線形効果のみを考慮しています。

---

### **区分一定媒質における解法**

本研究で扱う分極反転素子は、結合係数 $\kappa(z)$ が区分的に一定値をとる構造です。媒質を複数の区間に分割し、それぞれを**domain** と呼びます。$k$番目のドメインは $z_k \le z < z_{k+1}$ の範囲を占め、終端の$z_{k+1}$を**domain wall**と呼びます

本研究では単一種類の結晶を用いるため、結合係数の絶対値 $|\kappa(z)|$ はデバイス全長にわたって一定値 $\kappa_0$ であると仮定します。したがって、分極反転は係数の符号のみをドメインごとに切り替える操作 ($\kappa_k = \pm \kappa_0$) に対応します。

#### **1. 一般形式と伝播演算子**

まず、結合波方程式を複素振幅ベクトル $\boldsymbol{A}(z) = (A_1, A_2, A_3)^T$ を用いて、以下のように一般的に記述します。

$$\frac{d\boldsymbol{A}}{dz} = \boldsymbol{f}(\boldsymbol{A}, z; \kappa(z))$$

ここで、関数 $\boldsymbol{f}$ は、採用するモデルに応じた方程式に対応します。

$k$番目の区間 $[z_k, z_{k+1}]$ において結合係数 $\kappa_k$ は一定であり、この区間における伝播は、始点での振幅ベクトル $\boldsymbol{A}(z_k)$ を終点での振幅ベクトル $\boldsymbol{A}(z_{k+1})$ に写す非線形な「伝播演算子」$\mathcal{P}_k$ を用いて、以下のように定義できます。

$$\boldsymbol{A}(z_{k+1}) = \mathcal{P}_k \left[ \boldsymbol{A}(z_k) \right]$$

演算子 $\mathcal{P}_k$ の具体的な形は、区間長 $\Delta z_k = z_{k+1} - z_k$ と係数 $\kappa_k$ 、そして関数 $\boldsymbol{f}$ に依存します。

#### **2. 形式解**

媒質全体（$z=0$ から $z=L$）の伝播は、初期状態 $\boldsymbol{A}(0)$ に対して、各区間の伝播演算子 $\mathcal{P}_0, \mathcal{P}_1, \dots, \mathcal{P}_{N-1}$ を順次適用する操作、すなわち演算子の**Composition of Operators**によって与えられます。

$$\boldsymbol{A}(L) = \mathcal{P}_{N-1} \left[ \mathcal{P}_{N-2} \left[ \dots \left[ \mathcal{P}_0 \left[ \boldsymbol{A}(0) \right] \right] \dots \right] \right]$$

これは、積の記号 $\prod$ を用いてより簡潔に表現できます（演算子は右から順に作用）。

$$\boldsymbol{A}(L) = \left( \prod_{k=N-1}^{0} \mathcal{P}_k \right) \boldsymbol{A}(0)$$

この式が、区分一定媒質における結合波方程式の形式的な解（Formal Solution）です。

#### **3. THG変換効率**

第三高調波（THG）への変換効率 $\eta_3(L)$ は、初期の基本波光強度 $I_1(0)$ に対する、最終的な第三高調波光強度 $I_3(L)$ の比で定義されます。上記の形式解を用いると、変換効率は以下のように定式化されます。

$$\eta_3(L) = \frac{|A_3(L)|^2}{|A_1(0)|^2} = \frac{\left| \left[ \left( \prod_{k=N-1}^{0} \mathcal{P}_k \right) \boldsymbol{A}(0) \right]_3 \right|^2}{|A_1(0)|^2}$$

ただし、$[\cdot]_3$ はベクトルの第3成分を取り出す操作を示します。この定式化における各ドメインでの作用は、変化量が前の状態に依存し非線形であるため、$\sum$ではなく$\prod$の形で表されます。

### 数値解法

[Interaction Picture Method](./ipm_pc.md)でシミュレーションする

### **最適化による逆設計**

素子の性能(所望周波数および位相不整合量範囲でフラットトップかつ高効率)を定量的に評価する目的関数$J$を定義し、ansatzと最適化手法として以下の二つを考えています

#### **1. 勾配ベース最適化**

domain wallの連続的な位置 $\{z_k\}$ をansatzとします。目的関数 $J$ の勾配 $\nabla J$ を随伴法により算出し、勾配に沿ってansatzを更新します。勾配計算の自動化には`JAX`のようなフレームワークの利用を想定します。

#### **2. 組み合わせ最適化**

domain wallの位置を固定し、各ドメインの分極の向き $s_k \in \{+1, -1\}$ をansatzとします。Simulated AnnealingやMonte Carloを考えています

#### **3. 初期条件**

最適化計算の成功は、探索を開始する初期条件に大きく依存します。広大なパラメータ空間の中から効率的に最適な解を見出すため、物理的な直感に基づいた初期構造を設定します。

本研究で扱うカスケードTHGは、以下の2つの連続したプロセスとして理解できます。
1.  **SHG過程**: 基本波 (FW) → 第二高調波 (SHW)
2.  **SFG過程**: 基本波 (FW) + 第二高調波 (SHW) → 第三高調波 (THW)

この物理的描像に基づき、素子を機能的に2つのセクションに分割した構造を初期条件として採用します。具体的には、素子の前半部分をSHG効率の最大化に、後半部分をSFG効率の最大化にそれぞれ特化させます。

* **前半セクション (SHG部)**: デバイスの前半領域には、SHG過程の位相不整合 $\Delta k_1$ を補償するための擬似位相整合(QPM)条件を満たすようにドメインウォールを配置します。反転周期はコヒーレンス長 $L_c^{SHG} = \pi/\Delta k_1$ に対応します。
* **後半セクション (SFG部)**: デバイスの後半領域には、前半で生成されたSHWを効率よくTHWに変換するため、SFG過程の位相不整合 $\Delta k_2$ を補償するQPM条件に従ってドメインウォールを配置します。反転周期はコヒーレンス長 $L_c^{SFG} = \pi/\Delta k_2$ に対応します。

この「タンデム構造」を初期ansatzとすることで、完全にランダムな状態から始めるのではなく、物理的に解に近いと期待される有望な領域から最適化を開始できます。これにより、計算コストを削減し、より高性能な解への収束を促進することが期待されます。
