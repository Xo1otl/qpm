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
