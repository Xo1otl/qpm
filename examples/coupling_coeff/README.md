# 計算に必要なもの

TM00モードの電界分布で$\kappa$を計算

屈折率分布からTM00モードの電界分布を計算

拡散方程式から屈折率分布を計算

屈折率分布はアニールプロトン交換導波路の濃度分布から計算

$$
\Delta n = \Delta n_0 \frac {C}{C_0}
$$

拡散方程式は縦と横があり、それぞれ計算すると濃度分布がでる

1. 拡散方程式をといて濃度分布計算
2. 濃度分布から屈折率分布を計算
3. 屈折率分布からTM00モードの電界分布を$\omega, 2\omega, 3\omega$について計算
4. 積分の式をといて$\kappa_{SHG}, \kappa_{SFG}$を計算

$$
\kappa_{SHG} = \frac{2\omega\epsilon_0}{4} \iint E_{2\omega}(x,y)^*d(x,y)E_\omega(x,y)^2 dxdy
$$

$$
\kappa_{SFG} = \frac{\omega_3\epsilon_0}{2} \iint E_3(x,y)^*d(x,y)E_2(x,y)E_1(x,y) dxdy
$$
