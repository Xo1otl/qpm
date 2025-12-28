# 1.1 波長変換デバイスの用途

非線形光学効果を用いた波長変換デバイスは、既存のレーザー光源では発振が困難な波長域へのアクセスを可能にする重要な技術である。特に、擬似位相整合 (Quasi-Phase Matching: QPM) デバイスは、高い波長変換効率と設計の自由度を有しており、幅広い分野で応用されている。本節では、波長変換デバイスの主要な用途について概説し、特に本研究の主題である「変換効率スペクトルの制御」が求められる背景について述べる。

## 1.1.1 レーザー光源の波長域拡大
波長変換の最も基本的な用途は、コヒーレント光源の波長域拡大である。固体レーザーや半導体レーザーの発振波長は、利得媒質固有のエネルギー順位によって決まる特定の波長に限られている。波長変換技術を用いることで、紫外から中赤外に至る広範な波長域でコヒーレント光を得ることが可能となる。

- **可視・紫外光発生**: ディスプレイ技術やレーザー加工、医療応用において、青色や緑色、紫外光の需要が高い。QPMデバイスを用いた第二高調波発生 (SHG) は、コンパクトで高効率な短波長光源を実現する手段として広く利用されている。
- **中赤外光発生**: 分子分光や環境計測の分野では、分子の指紋領域である中赤外光が必要とされる。差周波発生 (DFG) や光パラメトリック発振 (OPO) を用いることで、近赤外レーザーから中赤外域への変換が可能となる。

## 1.1.2 光通信システムにおける波長変換
光ファイバ通信の大容量化に伴い、波長分割多重 (Wavelength Division Multiplexing: WDM) 技術が不可欠となっている。WDMネットワークの柔軟性と効率を向上させるために、光信号を電気信号に変換することなく波長のみを変換する「全光波長変換」技術が重要となる。

- **波長ミキシングによる変換**: QPMデバイスを用いた差周波発生 (DFG) やカスケード法による波長変換は、強度変調信号や位相変調信号のフォーマットを維持したまま変換できるため、トランスペアレントな波長変換が可能である。
- **広帯域かつ平坦な変換特性の必要性**: WDMシステムでは多数の波長チャネルを扱うため、波長変換デバイスにはCバンドやLバンドを含む広い帯域幅が求められる。また、チャネル間の信号品質を均一に保つためには、変換効率スペクトルが平坦であることが望ましい。これは本研究におけるスペクトル形状制御の重要な応用例の一つである。

## 参考文献

1. **基本理論**: M. M. Fejer, G. A. Magel, D. H. Jundt, and R. L. Byer, "Quasi-phase-matched second harmonic generation: tuning and tolerances," *IEEE Journal of Quantum Electronics*, vol. 28, no. 11, pp. 2631–2654, Nov. 1992.
2. **中赤外OPO**: L. E. Myers et al., "Quasi-phase-matched optical parametric oscillators in bulk periodically poled LiNbO3," *Journal of the Optical Society of America B*, vol. 12, no. 11, pp. 2102–2116, Nov. 1995.
3. **広帯域波長変換 (WDM)**: M. H. Chou, I. Brener, M. M. Fejer, E. E. Chaban, and S. B. Christman, "1.5-μm-band wavelength conversion based on cascaded second-order nonlinearity in LiNbO3 waveguides," *IEEE Photonics Technology Letters*, vol. 11, no. 6, pp. 653–655, June 1999.
