# 1.1 序論執筆用 参考文献リスト

`ask.md` および `ask_analysis.md` で検討された「本研究（スペクトル制御）の適用範囲」を裏付けるための参考文献リストです。

## 1. 量子もつれ光子対のスペクトル制御 (SPDC Spectrum Engineering)
**文脈**:
本研究の「ドメイン構造エンジニアリングによるスペクトル形状制御」は、量子もつれ光子対（SPDC）生成において極めて重要です。自然な矩形領域では位相整合関数がSinc関数となり、サイドローブが生じますが、これをガウス関数型に整形することで、純粋状態（Factorable state）の光子対を生成し、量子忠実度を向上させる研究があります。

*   **理論的提案 (Seminal Paper)**
    *   **Brańczyk, A. M.**, Fedrizzi, A., Stace, T. M., Ralph, T. C., & White, A. G. (2011). "Engineered optical nonlinearity for quantum light generation." *Optics Express*, 19(1), 55-65.
    *   **内容**: QPM格子のデューティ比変調や極性反転パターンを最適化（ガウスアポダイゼーション）することで、ジョイントスペクトル振幅（JSA）のサイドローブを除去できることを理論的に示した論文。

*   **実験的実証**
    *   **Dosseva, M.**, Cincio, L., & Brańczyk, A. M. (2016). "Shaping the joint spectrum of entangled photons via domain engineering." *Physical Review A*, 93(1), 013801.
    *   **内容**: PPLN導波路においてドメインエンジニアリングを行い、実際に整形されたスペクトルを持つ光子対生成を実証した研究。

*   **関連レビュー**
    *   **Graffitti, F.**, et al. (2018). "Independent high-purity photons created in domain-engineered crystals." *Optica*, 5(5), 514-517.

## 2. フェムト秒パルス整形・広帯域変換 (Femtosecond Pulse Shaping)
**文脈**:
「広帯域化」は「超短パルス（フェムト秒）対応」と等価です。CW近似での設計（伝達関数設計）が、フェムト秒レーザーのパルス整形や圧縮に適用可能であることを示すための、M. M. Fejer教授らのグループによる基礎論文です。

*   **伝達関数理論とパルス整形**
    *   **Imeshev, G.**, Fejer, M. M., Galvanauskas, A., & Harter, D. (2001). "Pulse shaping by difference-frequency mixing with quasi-phase-matching gratings." *Journal of the Optical Society of America B*, 18(4), 534-547.
    *   **内容**: QPM格子を特定の周波数応答関数（Transfer Function）を持つフィルタとして設計し、波長変換と同時にパルス整形を行う理論と実証。本研究の「スペクトル制御」のアナロジーとして最適です。

*   **SHGにおけるパルス圧縮・整形**
    *   **Imeshev, G.**, et al. (1998). "Engineerable femtosecond pulse shaping by second-harmonic generation with Fourier synthetic quasi-phase-matching gratings." *Optics Letters*, 23(11), 864-866.
    *   **内容**: フーリエ合成QPM格子を用いて、SHG光の振幅と位相を制御してパルス整形を行う研究。

## 3. 温度許容度向上 (Broadband/Temperature Tolerance)
**文脈**:
単純な広帯域化（フラットトップ化）の産業的メリットとして、「温度変化に強い（TEC-free）」という点を挙げるための文献です。

*   **チャープQPMによる温度許容度拡大**
    *   **Ashihara, S.**, et al. (2014). "Large temperature acceptance in second-harmonic generation with a chirped quasi-phase-matching grating." *Journal of Applied Physics*, 115, 093102.
    *   **内容**: チャープ周期PPLNを用いてSHG帯域を広げることで、通常のPPLNに比べて格段に広い温度許容幅（Temperature Acceptance）を実現できることを示した論文。

*   **基礎理論**
    *   **Mizuuchi, K.**, Yamamoto, K., Kato, M., & Sato, H. (1994). "Broadening of second-harmonic generation bandwidth in quasi-phase-matched waveguides." *IEEE Journal of Quantum Electronics*, 30(7), 1596-1604.
    *   **内容**: 導波路QPMデバイスにおいて、チャープグレーティングを用いて位相整合帯域を広げる手法の初期の研究。

## 4. cTHG (第三高調波発生) の高効率化と応用 (UV Generation)
**文脈**:
本研究のSHG+SFGモデルの直接的な応用先である、1064nmからの355nm（UV）生成とその用途（バイオイメージング等）に関する文献です。

*   **PPLNによる高効率355nm発生 (Casaded SHG+SFG)**
    *   **Mizuuchi, K.**, et al. (1997). "Generation of 355-nm light by frequency doubling of a distributed Bragg reflector diode laser ... in a bulk periodically poled LiNbO3 crystal." *Optics Letters*, 22(16), 1217-1219.
        *   (または **J. Capmany** et al., "Continuous-wave generation of 355 nm light in a single periodically poled lithium niobate crystal," *Opt. Lett.* 23, 16 (1998).)
    *   **内容**: 単一のPPLN結晶を用いたカスケード過程（SHG+SFG）による355nm連続波発生の実証。

*   **cTHGの一般的応用（バイオイメージング）**
    *   特定の論文というよりは、一般に「355nmレーザー」はフローサイトメトリーやDNAシーケンサ、共焦点顕微鏡の光源として使用されます。イントロでは「355nm光源はバイオイメージング等で需要が高い」と述べ、上記デバイス論文を引くのが通例です。

## 5. WDM波長変換 (Cascaded SHG+DFG)
**文脈**:
WDM通信における波長変換デバイスとしてのQPMの応用。SHG+SFGと物理的に等価なSHG+DFGの代表論文。

*   **代表的論文**
    *   **Chou, M. H.**, Brener, I., Fejer, M. M., et al. (1999). "1.5-μm-band wavelength conversion based on cascaded second-order nonlinearity in LiNbO3 waveguides." *IEEE Photonics Technology Letters*, 11(6), 653-655.
    *   **内容**: PPLN導波路を用いたカスケードSHG+DFGによる、広帯域かつ高効率なWDM波長変換の実証。この分野の金字塔的論文です。
