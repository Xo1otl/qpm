## **結合波方程式の積分形**
カスケード二次非線形過程(SHG+SFG)によるTHGの結合波方程式系。
変数定義: $\boldsymbol{A}(z)$は各波の複素振幅ベクトル, $\kappa(z)$は非線形結合係数, $\Delta k_j$は位相不整合量。

初期の結合波方程式系:
$$
\begin{align}
\frac{d A_1}{dz} &= i \kappa(z) \left[ A_2 A_1^* e^{i\Delta k_1 z} + A_3 A_2^* e^{i\Delta k_2 z} \right] \\
\frac{d A_2}{dz} &= i \, 2\kappa(z) \left[ \frac{1}{2} A_1^2 e^{-i\Delta k_1 z} + A_3 A_1^* e^{i\Delta k_2 z} \right] \\
\frac{d A_3}{dz} &= i \, 3\kappa(z) \left[ A_1 A_2 e^{-i\Delta k_2 z} \right]
\end{align}
$$

変数変換: 新しい変数ベクトル $\boldsymbol{B}(z)$ を $A_1(z) = B_1(z)$, $A_2(z) = B_2(z) e^{-i\Delta k_1 z}$, $A_3(z) = B_3(z) e^{-i(\Delta k_1 + \Delta k_2) z}$ で定義する。

これにより、方程式系は以下の行列形式 $\frac{d\boldsymbol{B}}{dz} = \boldsymbol{L}\boldsymbol{B} + \boldsymbol{N}(\boldsymbol{B})$ に変換される。
$$
\frac{d\boldsymbol{B}}{dz} = \underbrace{i \begin{pmatrix} 0 & 0 & 0 \\ 0 & \Delta k_1 & 0 \\ 0 & 0 & \Delta k_1 + \Delta k_2 \end{pmatrix}}_{\boldsymbol{L}} \boldsymbol{B} + \underbrace{i \kappa(z) \begin{pmatrix} B_1^* B_2 + B_2^* B_3 \\ B_1^2 + 2 B_1^* B_3 \\ 3 B_1 B_2 \end{pmatrix}}_{\boldsymbol{N}(\boldsymbol{B})}
$$

この方程式は以下の積分形式に恒等変形できる。
$$\boldsymbol{B}(z_n+h) = e^{\boldsymbol{L}h} \boldsymbol{B}(z_n) + \int_0^h e^{\boldsymbol{L}(h-\tau')} \boldsymbol{N}(\boldsymbol{B}(z_n+\tau')) d\tau'$$

## **ETDRK4法による高次精度計算**

ETDRK4法は、stiffな常微分方程式系のための4次精度数値解法である。

### **K&Tのスキーム**

$\boldsymbol{B}$ の $z_n$ から $z_{n+1} = z_n + h$ への1ステップ更新は、以下の中間ベクトルと最終更新式によって実行される。

**1. 中間ベクトル**
$$
\begin{align}
a_n &= e^{\boldsymbol{L}h/2} \boldsymbol{B}_n + \boldsymbol{Q} \boldsymbol{N}(\boldsymbol{B}_n, z_n) \\
b_n &= e^{\boldsymbol{L}h/2} \boldsymbol{B}_n + \boldsymbol{Q} \boldsymbol{N}(a_n, z_n+h/2) \\
c_n &= e^{\boldsymbol{L}h/2} a_n + \boldsymbol{Q} \left[ 2\boldsymbol{N}(b_n, z_n+h/2) - \boldsymbol{N}(\boldsymbol{B}_n, z_n) \right]
\end{align}
$$

**2. 最終更新式**
$$\boldsymbol{B}_{n+1} = e^{\boldsymbol{L}h} \boldsymbol{B}_n + \boldsymbol{f}_1 \boldsymbol{N}(\boldsymbol{B}_n, z_n) + \boldsymbol{f}_2 \left[ \boldsymbol{N}(a_n, z_n+h/2) + \boldsymbol{N}(b_n, z_n+h/2) \right] + \boldsymbol{f}_3 \boldsymbol{N}(c_n, z_n+h)$$

### **係数の定義**

上記スキームは、係数行列 $\boldsymbol{Q}, \boldsymbol{f}_1, \boldsymbol{f}_2, \boldsymbol{f}_3$ を必要とする。これらは、線形演算子$\boldsymbol{L}$、ステップサイズ$h$、そして以下で定義される$\phi$関数を用いて構築される。

$Z = h\boldsymbol{L}$ と定義すると、各係数は次式で与えられる。
$$
\begin{align}
\boldsymbol{Q} &= \frac{h}{2} \phi_1\left(\frac{Z}{2}\right) \\
\boldsymbol{f}_1 &= h \left[ \phi_1(Z) - 3\phi_2(Z) + 4\phi_3(Z) \right] \\
\boldsymbol{f}_2 &= h \left[ 2\phi_2(Z) - 4\phi_3(Z) \right] \\
\boldsymbol{f}_3 &= h \left[ 4\phi_3(Z) - \phi_2(Z) \right]
\end{align}
$$

### **対角行列への適用**

線形演算子$\boldsymbol{L}$が対角行列であるため、行列関数$\phi_k(h\boldsymbol{L})$の計算は、各対角成分$L_{jj}$を$\phi_k(x)$に代入して計算される。
$$\phi_k(h\boldsymbol{L}) = \begin{pmatrix} \phi_k(hL_{11}) & 0 & 0 \\ 0 & \phi_k(hL_{22}) & 0 \\ 0 & 0 & \phi_k(hL_{33}) \end{pmatrix}$$

### **$\phi$関数の定義**

$\phi_k(x)$関数は以下の再帰関係式で定義される。
$$\phi_{k+1}(x) = \frac{\phi_k(x) - 1/k!}{x}, \quad \text{ただし} \quad \phi_0(x) = e^x$$

この関係式から導かれる主要な解析的表現は次の通り。
$$
\begin{align}
\phi_1(x) &= \frac{e^x - 1}{x} \\
\phi_2(x) &= \frac{e^x - 1 - x}{x^2} \\
\phi_3(x) &= \frac{e^x - 1 - x - x^2/2}{x^3}
\end{align}
$$

### **$hL_{jj}$がゼロに近い場合の数値的安定性**

対角成分 $L_{jj}$ がゼロに非常に近い場合、すなわち $x=hL_{jj}$ の絶対値が小さい場合、$\phi_k(x)$ の解析的表現、例えば $\phi_1(x) = (e^x - 1)/x$ などは、分子の計算で桁落ちが発生する危険性がある。

この問題を回避するため、$|x|$ が特定の閾値よりも小さい場合には、$\phi_k(x)$ の計算にテイラー展開による近似式を用いる。

$$
\begin{align}
\phi_1(x) &= \sum_{n=0}^{\infty} \frac{x^n}{(n+1)!} \approx 1 + \frac{x}{2!} + \frac{x^2}{3!} + \dots \\
\phi_2(x) &= \sum_{n=0}^{\infty} \frac{x^n}{(n+2)!} \approx \frac{1}{2!} + \frac{x}{3!} + \frac{x^2}{4!} + \dots \\
\phi_3(x) &= \sum_{n=0}^{\infty} \frac{x^n}{(n+3)!} \approx \frac{1}{3!} + \frac{x}{4!} + \frac{x^2}{5!} + \dots
\end{align}
$$

特に、$L_{jj}=0$ の場合、$x=0$ となり、その値はテイラー展開の初項（$n=0$の項）に一致する。

$$
\begin{align}
\phi_1(0) = 1 \\
\phi_2(0) = 1/2 \\
\phi_3(0) = 1/6
\end{align}
$$

## 周期分極反転構造でのETDRK4法

周期的分極反転構造では、位相不整合 $\Delta k$ を補償するため、周期 $\Lambda \approx 2\pi/\Delta k$ で非線形係数 $\kappa(z)$ の符号を反転させる。$\kappa(z)$ は系全体としては$\boldsymbol{L}$に近いstiffnessだが、各ドメイン内部では $\kappa$ が定数である。そのため、ドメイン境界を数値計算のステップ境界とすることで、ETDRK4法を適用することが可能である。
