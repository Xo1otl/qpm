### [結合波方程式の出典](https://www.nature.com/articles/lsa201470)

geminiとdeepthinkが縮退を考慮すると係数が間違っているといっており全体を半分にしている

### [一定周期でNPDA使えるならプロパゲータが線形でイジングモデル](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2022.1038240/full)

$\kappa$を符号反転した後の効率はそのドメインの寄与だけ再計算すればよく高速に探索できる

多分波長ごとのドメインの寄与を行列にして一番バランス取れる符号の組み合わせをアルゴリズムで探す感じになりそう

あきらかに焼きなましよりましなアルゴリズムあるだろ、競プロだしたらだれか解けそう

### [ETDRK4のK&Tスキーム](https://people.maths.ox.ac.uk/trefethen/publication/PDF/2005_111.pdf)

このスキームで非線形項を更新してる

h大きめL固定対角行列という条件から解析解を使用している

対角成分0の部分は別で処理している

#### [エネルギーを保存するExponential Integrator](https://www.alphaxiv.org/pdf/2506.07072)
