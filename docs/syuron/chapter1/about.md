SHGのスペクトル制御:  
1.効率度外視で任意形状  
2.効率維持でフラット広帯域  

cTHGの高効率化:  
3.高速計算方法  
4.タンデムからの改善  

1.量子通信の光子対発生  
2.やってる先行研究あるから目的パクる  
3,4.光ファイバーかUV光源の設計で使えるかなぁ

## 懸念点
1.SHGとSPDCの違い
2.やってること同じだけど先行研究の研究目的が謎  
3,4.光ファイバーについて、SHG+SFGとSHG+DFGでやってること違う、1の場合と違って、SHG+DFGはそもそもSHG周期だけでもできてしまうレベルだし、cTHGの高効率化設計の出番なさそう。スピングラス問題なので、広帯域化などの設計は見込みが少ない.

微妙に違う部分却下される気もする。Geminiあればすぐに式修正してPoCまで書ける気もするけど、実際の数値の設定とか大変すぎるし相当急がなあかん。

うーむ...

## 進展
SPDCについてアバウトに理解した
1. **Biphoton State Representation & Pure State Condition**
   $$ |\Psi\rangle = \iint d\omega_s \, d\omega_i \, f(\omega_s, \omega_i) \, \hat{a}^\dagger_s(\omega_s) \, \hat{a}^\dagger_i(\omega_i) \, |0\rangle $$
   $$ \text{Tr}_i(|\Psi\rangle\langle\Psi|) = \text{Pure} \iff f(\omega_s, \omega_i) = \phi(\omega_s)\psi(\omega_i) \quad [\text{Factorizable}] $$

2. **Definition of JSA**
   $$ f(\omega_s, \omega_i) = \alpha(\omega_s + \omega_i) \times \Phi(\Delta k) $$
   $$ \qquad \qquad [\text{Pump: Gaussian}] \qquad [\text{PMF}] $$

3. **Definition of PMF**
   $$ \Phi(\Delta k) = \mathcal{F}[ d(z) ] $$

4. **Logical Consequence**  
   To achieve a pure state $\iff$ $[\text{Factorizable}]$
   $$ \downarrow \quad (\text{Since the Pump is Gaussian...}) $$
   The PMF must also be $[\text{Gaussian}]$
   $$ \downarrow \quad (\text{Due to binary material constraints...}) $$
   $d(z)$ is Duty-Cycle Modulated to simulate a Gaussian profile

## Task