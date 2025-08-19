### 積分形式

まず、運動方程式の厳密な積分形式から始めます。これは、ステップ開始時の状態 $\boldsymbol{B}(z_n)$ からステップ長 $h$ だけ進んだ状態 $\boldsymbol{B}(z_n+h)$ を記述するものです。

$$\boldsymbol{B}(z_n+h) = e^{i\boldsymbol{L}h} \boldsymbol{B}(z_n) + i \int_0^h e^{i\boldsymbol{L}(h-\tau')} \boldsymbol{N}(\boldsymbol{B}(z_n+\tau'), \boldsymbol{B}^*(z_n+\tau')) d\tau' \quad \cdots (1)$$

この式の右辺第一項 $e^{i\boldsymbol{L}h} \boldsymbol{B}(z_n)$ は線形な位相回転による発展を表し、第二項の積分はステップ中の非線形効果の寄与を表します。この積分を直接計算することは困難です。なぜなら、被積分関数が未来の状態 $\boldsymbol{B}(z_n+\tau')$ に依存しているためです。

### 一次近似

ここでInteraction Picture Methodの核心となる近似を導入します。ステップ長 $h$ が十分に短い場合、積分の中の状態ベクトル $\boldsymbol{B}(z_n+\tau')$ は、非線形項 $\boldsymbol{N}$ の影響がまだ小さいため、主に線形項によって発展すると考えられます。

そこで、積分内の $\boldsymbol{B}(z_n+\tau')$ を、ステップ開始時の状態 $\boldsymbol{B}(z_n)$ が**線形発展だけをした状態**で近似します。

$$\boldsymbol{B}(z_n+\tau') \approx e^{i\boldsymbol{L}\tau'} \boldsymbol{B}(z_n) \quad \cdots (2)$$

この近似を式(1)の積分項に代入することで、積分が計算可能になります。

### 近似式の計算

式(2)の近似を非線形項 $\boldsymbol{N}$ に適用します。$\boldsymbol{N}$ の各成分は $\boldsymbol{B}$ の要素の積で構成されているため、例えば以下のような置き換えが行われます。

* $B_1(\tau') \to e^{iL_1\tau'} B_{1n}$
* $B_2(\tau') \to e^{iL_2\tau'} B_{2n}$
* $B_1^*(\tau') B_2(\tau') \to (e^{iL_1\tau'} B_{1n})^* (e^{iL_2\tau'} B_{2n}) = e^{i(L_2-L_1)\tau'} B_{1n}^* B_{2n}$

この近似を適用した非線形項を $\boldsymbol{N}_{approx}(\tau')$ と書くことにします。
$$\boldsymbol{N}_{approx}(\tau') = \boldsymbol{N}(e^{i\boldsymbol{L}\tau'} \boldsymbol{B}_n, e^{-i\boldsymbol{L}\tau'} \boldsymbol{B}_n^*)$$

これを式(1)の積分項に代入すると、予測される非線形発展項 $\Delta \boldsymbol{B}_{NL}$ は次のようになります。

$$\Delta \boldsymbol{B}_{NL} \approx i \int_0^h e^{i\boldsymbol{L}(h-\tau')} \boldsymbol{N}_{approx}(\tau') d\tau'$$

### 成分ごとの導出（例：$\Delta B_{NL, 2}$）

具体的に $\Delta B_{NL, 2}$ の項を計算してみましょう。
まず、非線形項 $N_2$ は以下の通りです。
$$N_2(\boldsymbol{B}, \boldsymbol{B}^*) = \kappa (B_1^2 + 2 B_1^* B_3)$$
これに近似式(2)を適用すると、
$$
N_{2, approx}(\tau') = \kappa \left[ (e^{iL_1\tau'} B_{1n})^2 + 2(e^{iL_1\tau'} B_{1n})^* (e^{iL_3\tau'} B_{3n}) \right] \\
= \kappa \left[ e^{i(2L_1)\tau'} B_{1n}^2 + 2e^{i(L_3-L_1)\tau'} B_{1n}^* B_{3n} \right]
$$

これを $\Delta B_{NL, 2}$ の積分式に代入します。

$$
\begin{aligned}
\Delta B_{NL, 2} &\approx i \int_0^h e^{iL_2(h-\tau')} N_{2, approx}(\tau') d\tau' \\
&= i \int_0^h e^{iL_2h} e^{-iL_2\tau'} \kappa \left[ e^{i(2L_1)\tau'} B_{1n}^2 + 2e^{i(L_3-L_1)\tau'} B_{1n}^* B_{3n} \right] d\tau' \\
\end{aligned}
$$
定数項（$\tau'$ に依存しない項）を積分の外に出します。
$$
\Delta B_{NL, 2} = i \kappa e^{iL_2h} \left[ B_{1n}^2 \int_0^h e^{i(2L_1-L_2)\tau'} d\tau' + 2B_{1n}^* B_{3n} \int_0^h e^{i(L_3-L_1-L_2)\tau'} d\tau' \right]
$$
ここで、積分関数 $\Phi(\Omega, h)$ の定義を思い出します。
$$
\int_0^h e^{i\Omega\tau'} d\tau' = \frac{e^{i\Omega h}-1}{i\Omega} \equiv \Phi(\Omega, h)
$$
この定義を用いると、積分は $\Phi$ 関数で置き換えられます。
$$
\Delta B_{NL, 2} = i \kappa e^{iL_2h} \left[ B_{1n}^2 \Phi(2L_1-L_2, h) + 2B_{1n}^* B_{3n} \Phi(L_3-L_1-L_2, h) \right]
$$
これは、ご提示の資料にある $\Delta B_{NL, 2}$ の式と完全に一致します。
他の成分である $\Delta B_{NL, 1}$ と $\Delta B_{NL, 3}$ も同様の手順で導出することができます。

### 結論

IPM1 Predictor Schemeは、以下の手順で導出されます。

1.  運動方程式の厳密な積分形式から出発する。
2.  積分内の未知の状態ベクトル $\boldsymbol{B}(z_n+\tau')$ を、**線形演算子のみで時間発展させた状態 $e^{i\boldsymbol{L}\tau'} \boldsymbol{B}(z_n)$ で近似**する（一次近似）。
3.  この近似により、非線形項 $\boldsymbol{N}$ は $\tau'$ に関する指数関数の和となり、全体の積分が解析的に実行可能になる。
4.  積分を実行した結果が、提示された $\Phi$ 関数を用いた非線形発展項 $\Delta \boldsymbol{B}_{NL}$ の各式を与える。

この方法は、高速な線形発展（位相回転）の効果を近似の段階で織り込んでいるため、単純に $\boldsymbol{B}(z_n+\tau') \approx \boldsymbol{B}(z_n)$ と近似するよりも高い精度を達成できます。
