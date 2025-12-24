# THG最適化パラメータ

# 概要
THG計算再現パラメータ。

# 動作環境
- **入力**: 基本波 1 W @ 1031 nm のみ
- **温度**: 70°C

# デバイス仕様
- **構造**: [ドメイン幅のリスト (一次元配列)](https://github.com/Xo1otl/qpm/blob/main/examples/datasets/optimized_thg_2000_1489_e4.npy)
- **全長**: 23 mm

# 光学特性
| 波 | $\lambda$ (nm) | $n_{eff}$ |
| :--- | :--- | :--- |
| **Fund** | 1031.00 | 2.132502317428589 |
| **SHW** | 515.50 | 2.20408296585083 |
| **SFW/THW** | 343.67 | 2.35564160346985 |

- **結合係数**:
    - $\kappa_{SHG} = \kappa_{SFG} = \kappa$ とする
    - $\kappa$: $2.0577 \times 10^{-5}$ ($= \kappa_{eff} \times \pi / 2$)
    - $\kappa_{eff}$: $1.31 \times 10^{-5}$
    - 単位: $W^{-1/2} \mu m^{-1}$

# プロット設定
- **範囲**: 1025 - 1035 nm (1000 pts)
- **バルクと%の補正**: $100 \times 1.07 / 2.84$ (%/W)
- **ソース**: https://github.com/Xo1otl/qpm/blob/main/examples/presentation/optimized_thg.ipynb
