# 3. Theoretical Analysis of Nonlinear Interactions

There exist a variety of nonlinear-optic (NLO) interactions that enable device implementation. This chapter presents a theoretical analysis of second-order NLO interactions for traveling waves in waveguide structures. The fundamental concepts, techniques for the theoretical analysis, and the characteristics of each interaction are described in detail. The results of the analysis are discussed from the viewpoint of device design. Some results are summarized in graphic data with normalized parameters useful for designing, and guidelines and criteria for optimum design are given. NLO interactions in optical cavities will be discussed in the next chapter.

## 3.1 Guided-Mode Nonlinear-Optic Interactions

This section presents a theoretical analysis of basic NLO interactions between guided-mode traveling waves. To derive a general mathematical formulation, we assume quasi-phase matching (QPM) configurations. The process and the result, however, apply for all cases using phase matching between guided modes, including birefringence phase matching and mode dispersion phase matching.

### 3.1.1 Second-Harmonic Generation

Second-harmonic generation (SHG) in optical waveguides has been theoretically analyzed by several authors. The earlier work employed simple perturbation analysis for planar waveguides [3.1], [3.2]. The results, however, are not sufficient for predicting the performances of recent high-efficiency devices using channel waveguides. Jaskorzynska et al. gave a more general analysis of waveguide SHG phase-matched with a grating consisting of periodic modulation in nonlinear and linear permittivity [3.3], although the analysis was limited to mathematical expressions of exactly phase-matched SHG in a planar waveguide. Suhara et al. gave an analysis of SHG phase-matched with uniform and chirped gratings in channel waveguides, including cases where efficiency is high and residual phase mismatch is involved, and gave numerical and graphic data useful for device design [3.4].

![Fig. 3.1. Fundamental structure of quasi-phase matched nonlinear-optic device](figure_3_1.png)
**Fig. 3.1.** Fundamental structure of quasi-phase matched nonlinear-optic device

![Fig. 3.2. Periodic modulation of optical constants for quasi-phase matching](figure_3_2.png)
**Fig. 3.2.** Periodic modulation of optical constants for quasi-phase matching

***Expression of Waveguide with Grating***

Consider a channel waveguide having a uniform cross section, as a canonical structure. The channel is assumed to be parallel to one of the optical principal axes of the waveguide material, and the axis is taken as the $z$ axis of the coordinate system. The guide is described by the relative permittivity profile $\epsilon(x, y)$. We assume that the guide supports at least one guided mode for both fundamental (pump) and second-harmonic (SH) waves.

Figure 3.1 illustrates a channel waveguide with a grating of period $\Lambda$ for quasi-phase matching (QPM). The device can be fabricated by giving a periodic modulation of optical constants (permittivity and/or SHG coefficient) to the waveguide: the structure can be described by the distributions of the permittivity and the SHG coefficient. Let $\Delta\epsilon(x,y,z)$ be the periodic modulation in relative permittivity, then $\Delta\epsilon$ can be expressed in the form of a Fourier series:
$$\Delta\epsilon(x,y,z) = \sum_{q} \Delta\epsilon_q(x,y) \exp(-jqKz) , \quad K=2\pi/\Lambda, \tag{3.1}$$
where $\Delta\epsilon_q$ is the amplitude of the $q$-th order Fourier component, and $K$ is the spatial frequency of the grating, i.e., the magnitude of the grating vector. The distribution of the SHG coefficient with periodic modulation can be written as
$$d(x,y,z) = \sum_{q} d_q(x,y) \exp(-jqKz) , \quad K=2\pi/\Lambda. \tag{3.2}$$
Note that a waveguide without a grating is described by setting all the $\Delta\epsilon_q$ and $d_q$, except for $d_0$, to 0. Therefore the following expressions include cases of phase

matching without a grating (birefringence phase matching, etc.) as a special case. If $\Delta\epsilon$ and $d$ are represented by scalars, and the grating is a binary modulation, as shown in Fig. 3.2, the Fourier coefficients can be written as

$$
\Delta\epsilon_q = (\epsilon_a - \epsilon_b) \left( \frac{\sin q a \pi}{q \pi} \right) = (n_a^2 - n_b^2) \left( \frac{\sin q a \pi}{q \pi} \right), \quad (q \neq 0) \tag{3.3}
$$

$$
d_q = \begin{cases} (d_a - d_b) \frac{\sin q a \pi}{q \pi}, & (q \neq 0) \\ a d_a + (1 - a) d_b, & (q = 0) \end{cases} \tag{3.4}
$$

where $\epsilon_a, \epsilon_b; n_a, n_b; d_a, d_b$ are the permittivity, refractive index, and SHG coefficient of the two regions within a grating period, respectively, and $a$ denotes the duty ratio of the width of one region to the period. The subscripts to specify the tensor elements are omitted for simplicity; the appropriate component(s) should be taken according to the polarizations under consideration. The above Fourier coefficients can be functions of $(x, y)$, although they are not described explicitly.

### Nonlinear Coupled-Mode Equations

We derive nonlinear coupled-mode equations to describe SHG in waveguides with a grating. We use $\omega$ and $\lambda$ to denote the frequency and vacuum wavelength of the fundamental (pump) wave, respectively, and $2\omega$ and $\lambda/2$ for the second-harmonic (SH) wave.

The fundamental formula for deriving the coupled-mode equations is given by (2.58). Although (2.57) and (2.58) are written for the pump waves of frequency $\omega$, we have another set of equations for the SH waves, and they are given by replacing $\omega$ by $2\omega$ in (2.57) and (2.58). The polarization $\mathbf{P}(x, y, z)$ in the present SHG device can be written as

$$
\mathbf{P}^\omega = \mathbf{P}_L^\omega + \mathbf{P}_{NL}^\omega = \epsilon_0 \Delta\epsilon^\omega \mathbf{E}^\omega + 2 \epsilon_0 d \mathbf{E}^{\omega*} \mathbf{E}^{2\omega} , \tag{3.5a}
$$

$$
\mathbf{P}^{2\omega} = \mathbf{P}_L^{2\omega} + \mathbf{P}_{NL}^{2\omega} = \epsilon_0 \Delta\epsilon^{2\omega} \mathbf{E}^{2\omega} + \epsilon_0 d \mathbf{E}^\omega \mathbf{E}^\omega , \tag{3.5b}
$$

where $\mathbf{P}_L$ is the linear polarization due to the grating permittivity, $\mathbf{P}_{NL}$ the nonlinear polarization due to the SHG coefficient, and use has been made of (2.43).

Here we assume that there is no substantial coupling between modes of same frequency, but a mode of pump frequency couples with a mode of SH frequency through the quasi-phase matching (QPM) of the fundamental diffraction order ($q=1$). The condition for QPM (including a slight phase mismatch) is given by

$$
2\beta^\omega + K \cong \beta^{2\omega} \quad \text{or} \quad \Lambda \cong (\lambda/2) / (N^{2\omega} - N^\omega) , \tag{3.6}
$$

where $N^\omega$ and $N^{2\omega}$ are the mode indexes of the pump and SH waves, respectively. We also neglect the guided-mode to radiation-mode coupling. Then we have only to consider one mode each for fundamental and SH waves, and from (2.57), (2.58) and (3.5) we obtain the coupled-mode equations:

$$
\frac{d}{dz} A^\omega(z) + j(2\kappa_L^\omega \cos Kz) A^\omega(z) = -j \sum_q [\kappa_{NL}^{(q)} \exp(j2\Delta_q z)]^* [A^\omega(z)]^* A^{2\omega}(z), \tag{3.7a}
$$

$$
\frac{d}{dz} A^{2\omega}(z) + j(2\kappa_L^{2\omega} \cos Kz) A^{2\omega}(z) = -j \sum_q [\kappa_{NL}^{(q)} \exp(j2\Delta_q z)][A^\omega(z)]^2, \tag{3.7b}
$$

where $A^\omega(z)$ and $A^{2\omega}(z)$ are the amplitudes of the pump and SH waves, in which the subscript $m$ to denote the mode is omitted, and

$$
2\Delta_q = \beta^{2\omega} - (2\beta^\omega + qK). \tag{3.8}
$$

The linear and nonlinear coupling coefficients are defined as

$$
\kappa_L^\omega = \frac{\omega \epsilon_0}{4} \iint [E^\omega(x, y)]^* \Delta \epsilon_1^\omega(x, y) E^\omega(x, y) \, dxdy, \tag{3.9a}
$$

$$
\kappa_L^{2\omega} = \frac{2\omega \epsilon_0}{4} \iint [E^{2\omega}(x, y)]^* \Delta \epsilon_1^{2\omega}(x, y) E^{2\omega}(x, y) \, dxdy, \tag{3.9b}
$$

$$
\kappa_{NL}^{(q)} = \frac{2\omega \epsilon_0}{4} \iint [E^{2\omega}(x, y)]^* d_q(x, y) [E^\omega(x, y)]^2 \, dxdy. \tag{3.10}
$$

In the process to derive (3.7), only the fundamental ($q = \pm1$) Fourier components were retained, and the other unimportant terms in (3.1) were omitted. We also assumed, without loss of generality, that $\Delta \epsilon_q$ and $d_q$ are real.

Here we put

$$
A^\omega(z) = A(z) \exp[-j(2\kappa_L^\omega / K) \sin Kz], \tag{3.11a}
$$

$$
A^{2\omega}(z) = B(z) \exp[-j(2\kappa_L^{2\omega} / K) \sin Kz], \tag{3.11b}
$$

and substitute them into (3.7), and then in the resultant equations we use the mathematical formula:

$$
\exp(j \phi \sin Kz) = \sum_p J_p(\phi) \exp(j p K z), \tag{3.12}
$$

and take major terms with slow spatial variation. In (3.12), $J_p$ denotes the Bessel function of the $p$-th order. As a result, we obtain the simplified coupled-mode equations:

$$
\frac{d}{dz} A(z) = -j \kappa^* A^*(z) B(z) \exp[-j(2\Delta) z], \tag{3.13a}
$$

$$
\frac{d}{dz} B(z) = -j \kappa [A(z)]^2 \exp[+j(2\Delta) z], \tag{3.13b}
$$

where $A(z)$ and $B(z)$ are the amplitudes of the pump and SH waves.

$$
2\Delta = 2\Delta_1 = \beta^{2\omega} - (2\beta^\omega + K) \tag{3.14}
$$

is the phase mismatch parameter, and

$$
\kappa = \kappa_{NL}^{(1)} [J_0(\phi_L) + J_2(\phi_L)] - \kappa_{NL}^{(0)} J_1(\phi_L), \quad \phi_L = 2(\kappa_L^{2\omega} - 2\kappa_L^\omega) / K \tag{3.15}
$$

is the coupling coefficient for the SHG. Equations (3.9), (3.10), and (3.15) show that the SH wave is generated by direct SHG interaction due to the nonlinear grating represented by the $\kappa_{NL}^{(1)} [J_0 + J_2]$ term in (3.15) and indirect interaction through spatial harmonics generation due to the grating permittivity ($-\kappa_{NL}^{(0)} J_1$ term), and that both effects interfere with each other to give the total effect. Although in general the resultant $\kappa$ is complex, we hereafter assume that $\kappa$ is real and positive, since a constant phase factor is unimportant. In cases where $\kappa$ is complex, $|\kappa|$ should be used for $\kappa$. From (3.11) and the coupled mode equations (3.13) we obtain

$$\frac{d}{dz} \left( |A^\omega(z)|^2 + |A^{2\omega}(z)|^2 \right) = \frac{d}{dz} \left( |A(z)|^2 + |B(z)|^2 \right) = 0 . \tag{3.16}$$

Since $|A^\omega(z)|^2$ and $|A^{2\omega}(z)|^2$ give the power of the pump and harmonic waves, (3.16) shows the conservation of total power.

For SHG devices phase matched without a grating, we have $\kappa_L^\omega = \kappa_L^{2\omega} = 0$, and $\phi_L=0$. For such devices, the following discussion holds if $\Delta_0$ is used for $\Delta$ and the right hand side of (3.10) with $d_q$ replaced by $d$ is used for $\kappa$.

### Solutions of Coupled-Mode Equations

Consider SHG in the waveguide grating structure of Fig. 3.1 with interaction (grating) region in $0 \le z \le L$. The coupled-mode equations (3.13) are solved with boundary conditions:

$$A(0) = A_0, \quad B(0) = 0, \tag{3.17}$$

where $A_0$ is the amplitude of the incident pump wave.

**1) Approximate Solution for No Pump Depletion:** If the SHG efficiency is low and therefore there is a very small depletion of the pump wave power, we can approximate $A(z)$ as $A(z) = A_0$. In the no pump depletion approximation (NPDA), (3.13b) can readily be integrated using (3.17) to give

$$B(z) = -j\kappa A_0^2 z \exp(j\Delta z) \left( \frac{\sin \Delta z}{\Delta z} \right) , \tag{3.18}$$

and the well-known expression of the SHG efficiency [3.2]

$$\eta = |B(L)|^2 / |A(0)|^2 = \kappa^2 P_0 L^2 \left( \frac{\sin \Delta L}{\Delta L} \right)^2 , \tag{3.19}$$

where $P_0 = A_0^2$ is the incident pump power. For $\Delta=0, \eta = \kappa^2 P_0 L^2$, and $\eta/P_0 = \kappa^2 L^2$ is called the normalized efficiency and is often used as a convenient numerical figure to represent the device performance.

**2) Solution with Pump Depletion:** If the efficiency is not low enough, the pump amplitude $A(z)$ cannot be approximated by a constant, and the nonlinear differential equations (3.13a) and (3.13b) must be solved with the boundary condition (3.17). The equations were solved by Armstrong, et al. [3.5]. Using the total power conservation relation (3.16) and the boundary condition (3.17), $A(z)$ in

(3.13b) can be eliminated, and (3.13b) can be integrated to obtain $B(z)$. Without loss of generality $\kappa$ can be assumed real. The result for the SHG efficiency can be written as
$$
\eta = |B(L)|^2 / |A(0)|^2 = \gamma \text{sn}^2 [\kappa \sqrt{P_0} L / \sqrt{\gamma}; \gamma] , \tag{3.20a}
$$
$$
\gamma = \left[ \sqrt{1 + (\Delta/2\kappa \sqrt{P_0})^2} + |\Delta|/2\kappa \sqrt{P_0} \right]^{-2} , \tag{3.20b}
$$
where $\text{sn}[\zeta; \gamma]$ is the Jacobian elliptic function defined by
$$
\zeta = \int_0^\xi \frac{d\xi'}{\sqrt{(1-\xi'^2)(1-\gamma^2\xi'^2)}} , \quad \xi = \text{sn}[\zeta; \gamma] . \tag{3.21}
$$
For exact phase matching, $2\Delta = 0$, the solution of (3.13) is given by
$$
A(z) = A_0 \text{sech}(\kappa \sqrt{P_0} z), \quad B(z) = -j(A_0^2 / \sqrt{P_0}) \text{tanh}(\kappa \sqrt{P_0} z) . \tag{3.22}
$$
For $2\Delta = 0$, we have $\gamma = 1$, and the expression for the efficiency (3.20) reduces to
$$
\eta_{pm} = \text{sn}^2 [\kappa \sqrt{P_0} L; 1] = \text{tanh}^2 (\kappa \sqrt{P_0} L) , \tag{3.23}
$$
and $\eta_{pm}$ approaches $\eta = 1$ asymptotically with increasing $\kappa \sqrt{P_0} L$.

If there is a phase mismatch ($2\Delta \neq 0$), $\gamma < 1$, and the efficiency is a periodic function of $\kappa \sqrt{P_0} L$. The period is given by

$$
\Pi = 2\sqrt{\gamma} \int_0^1 \frac{d\xi}{\sqrt{(1-\xi^2)(1-\gamma^2\xi^2)}} = 2\sqrt{\gamma} F(\gamma, 1) \tag{3.24a}
$$
$$
\cong 2\sqrt{\gamma} F(0, 1) = \pi\sqrt{\gamma} \quad (\gamma \ll 1) \tag{3.24b}
$$
$$
\cong \pi \kappa \sqrt{P_0} / |\Delta| \quad (\gamma \ll 1, \ |\Delta|/2\kappa \sqrt{P_0} \gg 1) , \tag{3.24c}
$$

where $F(\gamma, 1)$ is the complete elliptic integral. Since $\text{sn}[\zeta; \gamma] \le 1$, the maximum efficiency is $\eta_{max} = \gamma (<1)$.

Although in the above discussion we assumed that the pump wave is of a single frequency, the SHG efficiency is affected by the pump wave spectrum. It has been shown by Helmfrid, et al. that, when pumped by a multimode laser, NPDA efficiency is enhanced by a factor of two as compared to a single mode case [3.6]. A theoretical analysis of QPM-SHG by backward propagating interaction, where the SH wave is generated in reflection, was presented by Matsumoto, et al., and Ding, et al. [3.7]. It was shown that bistability appears in the output SH power and the transmitted pump power within a range of phase mismatch.

### Coupling Coefficients and Effective SHG Coefficients

We have derived expressions for the coupling coefficients relevant to SHG. We modify them here to obtain formulas more convenient for device designing. Al-

though guided waves in a channel waveguide are hybrid modes with vector mode profiles, we can use approximate scalar expressions for the major component. The normalized mode profiles can be rewritten as
$$E^\omega(x,y) = C^\omega \mathcal{E}^\omega(x,y), \quad E^{2\omega}(x,y) = C^{2\omega} \mathcal{E}^{2\omega}(x,y), \tag{3.25}$$
using unnormalized profiles $\mathcal{E}^\omega(x, y)$ and $\mathcal{E}^{2\omega}(x, y)$. The normalization factors $C^\omega$, $C^{2\omega}$ are obtained from (2.12) as
$$C^\omega = \left[ \frac{\beta^\omega}{2\omega \mu_0} \iint |\mathcal{E}^\omega|^2 dxdy \right]^{-1/2}, \quad C^{2\omega} = \left[ \frac{\beta^{2\omega}}{2(2\omega) \mu_0} \iint |\mathcal{E}^{2\omega}|^2 dxdy \right]^{-1/2}, \tag{3.26}$$
We can then rewrite (3.9) and (3.10), and combining the result with (3.15) yields
$$\kappa = \varepsilon_0 \sqrt{\frac{(2\omega)^2}{2(N^\omega)^2 N^{2\omega}} \left( \frac{\mu_0}{\varepsilon_0} \right)^{3/2} \frac{d_{\text{eff}}^2}{S_{\text{eff}}}}, \tag{3.27}$$
where
$$d_{\text{eff}} = \frac{\sqrt{S_{\text{eff}}} \iint [\mathcal{E}^{2\omega}]^* [ \{J_0(\phi_L) + J_2(\phi_L)\} d_1 - J_1(\phi_L) d_0 ] [\mathcal{E}^\omega]^2 dxdy}{\sqrt{\iint |\mathcal{E}^{2\omega}|^2 dxdy} \iint |\mathcal{E}^\omega|^2 dxdy}, \tag{3.28}$$
$$\phi_L = \frac{1}{N^{2\omega} - N^\omega} \left[ \frac{\iint \Delta\varepsilon_1^{2\omega} |E^{2\omega}|^2 dxdy}{\iint |E^{2\omega}|^2 dxdy} - \frac{\iint \Delta\varepsilon_1^\omega |E^\omega|^2 dxdy}{\iint |E^\omega|^2 dxdy} \right]. \tag{3.29}$$
To obtain (3.29), use has been made of (3.6). An effective cross section, denoted by $S_{\text{eff}}$, can be defined rather arbitrarily since the combination of $d_{\text{eff}}$ and $S_{\text{eff}}$ to represent a device is not unique. In many cases, it is convenient to use $d_{\text{eff}}$ and $S_{\text{eff}}$ close to the area of guided mode profiles; examples will be given in (3.32) and (3.34).

Here we consider SHG with optical beams having uniform cross sections in homogeneous bulk medium (without grating) of SHG coefficient $d$. Putting $\Delta\varepsilon = 0$ and $d(x, y, z) = d = \text{constant}$, considering the field profiles constant over area $S_{\text{eff}}$, and making a similar analysis as described above, we easily see that the coupled-mode expression for this bulk SHG is the same as (3.13) and the coupling coefficient is given by (3.27) with $d_{\text{eff}}$ replaced by an appropriate element of $d$. This implies that the waveguide SHG, in turn, is equivalent to SHG in a bulk medium having a SHG coefficient $d_{\text{eff}}$, and therefore, $d_{\text{eff}}$ is the effective SHG coefficient for the waveguide SHG.

We can simplify the expression of $d_{\text{eff}}$ for special cases. If the grating modulations are uniform over the waveguide channel, i.e., $\Delta\varepsilon_q(x, y) = \Delta\varepsilon_q = \text{const}$, $d_q(x, y) = d_q = \text{const}$, we can move them to the outside of the integrals in (3.28) and (3.29), to obtain
$$d_{\text{eff}} = [J_0(\phi_L) + J_2(\phi_L)] d_1 - J_1(\phi_L) d_0, \tag{3.30}$$
$$\phi_L = [(\Delta\varepsilon_1^{2\omega} / N^{2\omega}) - (\Delta\varepsilon_1^\omega / N^\omega)] / (N^{2\omega} - N^\omega), \tag{3.31}$$

where we put
$$S_{eff} = \frac{\iint |E^{2\omega}|^2 dxdy \left[ \iint |E^\omega|^2 dxdy \right]^2}{\left[ \iint [E^{2\omega}]^* [E^\omega]^2 dxdy \right]^2} . \tag{3.32}$$

For guided waves of the fundamental lateral mode, the mode profile can be approximated by a Gaussian function:
$$E(x,y) = \exp[-(2x / W_x)^2] \exp[-(2y / W_y)^2] , \tag{3.33}$$
where $W_x$ and $W_y$ denote the $1/e^2$ mode widths in the $x$ and $y$ directions, respectively. The distance in the $x$ direction between the peaks of pump and harmonic profiles, $d_x$, may be incorporated. Substituting (3.33) into (3.32) yields
$$S_{eff} = \frac{\pi}{32} \left[ \frac{(W_x^\omega)^2 + 2(W_x^{2\omega})^2}{W_x^{2\omega}} \right] \left[ \frac{(W_y^\omega)^2 + 2(W_y^{2\omega})^2}{W_y^{2\omega}} \right] \exp \left[ \frac{16d_x^2}{(W_x^\omega)^2 + 2(W_x^{2\omega})^2} \right] \tag{3.34}$$
as an approximate expression for an effective waveguide cross section. The $1/e^2$ width $W$ is $(2/\ln 2)^{1/2} = 1.70$ times FWHM.

We next consider simpler cases of laterally uniform gratings. If the modulation is only in the permittivity (linear grating), we have $d_0=d, d_1=0$, and therefore $d_{eff} = -J_1(\phi_L)d$. If the modulation is only in the SHG coefficient (nonlinear grating), on the other hand, we have $\Delta \varepsilon_1 = 0, \phi_L = 0$ and therefore $d_{eff} = d_1$. A special case is periodic inversion of the SHG coefficient, for which we can substitute $d_a = -d_b = d$ into (3.4), and for $a=1/2$ we have $d_{eff} = (2/\pi)d$.

The periodic modulation is not laterally uniform in gratings fabricated by etching, cladding, or shallow diffusion, etc. For such gratings, $d_{eff}$ must be calculated by (3.28), (3.29), although the definition of $S_{eff}$ by (3.32) or (3.34) can be used.

The effective SHG coefficient $d_{eff}$ defined above can be used as a convenient measure to compare various waveguide grating structures in device design; the larger $d_{eff}$ is, the higher efficiency the device will have. For QPM SHG using $d_{33}$ element ($d_{eff} = (2/\pi)d_{33}$) in a $\text{LiNbO}_3$ waveguide of an effective cross section of $S_{eff} = 5 \mu\text{m}^2$, for examples, the coupling coefficient $\kappa$ is around $0.7 \text{ W}^{-1/2}\text{mm}^{-1}$.

### SHG Characteristics and Design Guidelines

**1) Dependence of Efficiency on Interaction Length:** Equation (3.20) shows that the SHG efficiency $\eta$ is a function of $\kappa \sqrt{P_0} L$ and $|\Delta|/\kappa \sqrt{P_0}$, which are dimensionless variables. For constant input power $P_0, \kappa \sqrt{P_0} L$ and $|\Delta|/\kappa \sqrt{P_0}$ are considered a normalized interaction length and a normalized phase mismatch, respectively. Figure 3.3 shows the calculated dependence of the efficiency on the interaction length. While the efficiency with exact phase matching increases monotonically with the interaction length as (3.23), the efficiency with phase mismatch oscillates periodically.

![Fig. 3.3. Dependence of SHG efficiency on interaction length [3.4]](https://example.com/figure3_3.png)

**Fig. 3.3.** Dependence of SHG efficiency on interaction length [3.4]

The NPDA result (3.19) shows that the period in $L$ is two times $L_c = \pi/2|\Delta|$, which is the coherence length for QPM with residual mismatch given by (3.14). The corresponding NPDA period in $\kappa\sqrt{P_0}L$ is $\pi/(|\Delta|/\kappa\sqrt{P_0})$, which is the same as (3.24c). We confirm from Fig. 3.3 that the NPDA result (3.19) is a good approximation of the exact result for large $|\Delta|/\kappa\sqrt{P_0}$ (poor phase matching or small pump power); the exact period $\Pi$ (3.24a) is close to the NPDA period. For small $|\Delta|/\kappa\sqrt{P_0}$, however, $\Pi$ is considerably smaller than the NPDA period, and (3.24b) with (3.20b) is a better approximation for $|\Delta|/\kappa\sqrt{P_0} > 0.5$.

The efficiency $\eta$ for $\Delta \neq 0$ takes the maximum value of $\eta_{\text{max}} = \gamma$ at $L$ corresponding to half period $\Pi/2$, and then decreases. This means that interaction length $L$ should not be too large; a design guideline is to determine the value of $L$ as large as possible within this upper limit.

**2) Dependence of Efficiency on Pump Power:** Equation (3.20) gives the SHG efficiency as a function of $\kappa\sqrt{P_0}L$ and $\Delta/\kappa\sqrt{P_0} = \Delta L/(\kappa\sqrt{P_0}L)$. For constant $L$, $\kappa\sqrt{P_0}L$ and $\Delta L$ are considered a normalized pump amplitude and a normalized phase mismatch, respectively. Figure 3.4 shows the calculated dependence of the efficiency on the pump amplitude. The efficiency for SHG with exact phase matching increases monotonically with the pump power, as described by (3.23), and also by (3.19) for small pump power. It should be noted that for SHG with phase mismatch the exact efficiency increases first and then decreases, although the NPDA result (3.19) gives only the monotonic increase in the low pump power region. The first peak of the efficiency is in $1 < \kappa\sqrt{P_0}L < 2$ for $0.5 < |\Delta L| < 2$. This oscillatory behavior is related to the dependence of the period in $L$ on the pump power discussed in 1).

**3) Phase-Matching Bandwidth:** Figure 3.5 shows the dependence of the efficiency on the normalized phase mismatch $\Delta L$, calculated from (3.20). The efficiency is normalized by the phase-matched efficiency $\eta_{pm}$ given by (3.23). The

![Fig. 3.4. Dependence of SHG efficiency on pump amplitude [3.4]](figure_3_4.png)
**Fig. 3.4.** Dependence of SHG efficiency on pump amplitude [3.4]

![Fig. 3.5. Dependence of SHG efficiency on phase mismatch [3.4]](figure_3_5.png)
**Fig. 3.5.** Dependence of SHG efficiency on phase mismatch [3.4]

result for small pump power is in good agreement with the squared sinc NPDA response given by (3.19); the 3-dB full bandwidth, in terms of the mismatch parameter, is $|\Delta| < 1.39/L$. While a wide bandwidth can be obtained with a short $L$, the efficiency is lower for small $L$. From the NPDA result we have a relation $|\Delta|\sqrt{\eta_{pm}} < 1.39 \kappa \sqrt{P_0}$, which shows that efficiency and bandwidth must be traded off.

We also see from Fig. 3.5 that the bandwidth decreases with increasing pump power. In the presence of NLO interaction with phase mismatch, the pump phase shifts from the free-propagating values as a reaction of SH generation (this is cascaded $\chi^{(2)}$ effect discussed in Section 3.5), and the shift results in additional mismatch to narrow the bandwidth. This implies that the phase matching requirement for large pump power is more stringent than the NPDA estimation, and therefore bandwidth narrowing must be taken into account in the design of high-efficiency SHG devices.

### 3.1.2 Sum-Frequency Generation

When optical waves of two different frequencies are simultaneously incident in a second-order NLO medium, the induced nonlinear polarization includes components of frequencies corresponding to the sum and the difference of the incident

frequencies. Efficient generation of the sum-frequency wave takes place, if phase matching or quasi-phase matching is satisfied among the relevant phase constants. This is called sum-frequency generation (SFG) or parametric up-conversion.

### Coupled-Mode Equations

Let $\omega_1$ and $\omega_2$ ($\omega_1 \neq \omega_2$) be the angular frequencies of the incident waves, and $\omega_3 = \omega_1 + \omega_2$ be the sum frequency. Assuming the existence of optical waves of the three frequencies, we write

$$E(t) = \text{Re} \{ E_1 \exp(j\omega_1 t) + E_2 \exp(j\omega_2 t) + E_3 \exp(j\omega_3 t) \}, \tag{3.35a}$$

$$P(t) = \text{Re} \{ P_1 \exp(j\omega_1 t) + P_2 \exp(j\omega_2 t) + P_3 \exp(j\omega_3 t) \}, \tag{3.35b}$$

where $E_1, E_2, E_3, P_1, P_2, P_3$ are complex expressions of the field and polarization of frequencies $\omega_1, \omega_2, \omega_3$. Then, using (2.38), we obtain expressions of NLO polarization components

$$P_1 = 2\varepsilon_0 d E_3 E_2^*, \quad P_2 = 2\varepsilon_0 d E_3 E_1^*, \quad P_3 = 2\varepsilon_0 d E_1 E_2, \tag{3.36}$$

where we put $\chi^{(2)} (-\omega_3, \omega_1, \omega_2) / 2 = d$. We next write the electric fields by using the normalized modes and the propagation constants as

$$E_1(x, y, z) = A_1(z) E_1(x, y) \exp(-j\beta_1 z), \tag{3.37a}$$

$$E_2(x, y, z) = A_2(z) E_2(x, y) \exp(-j\beta_2 z), \tag{3.37b}$$

$$E_3(x, y, z) = A_3(z) E_3(x, y) \exp(-j\beta_3 z), \tag{3.37c}$$

where $A_1(z), A_2(z), A_3(z)$ are mode amplitudes.

Here we consider a quasi-phase matching (QPM) configuration where the periodic modulation is only in the nonlinear permittivity. Then the QPM grating is described by (3.2). From (2.58), (3.2), (3.36) (3.37), we find that the $q$-th order QPM condition (including slight phase mismatch) for SFG is given by

$$\beta_1 + \beta_2 + qK \cong \beta_3 \quad \text{or} \quad \Lambda \cong q[N_3/\lambda_3 - (N_1/\lambda_1 + N_2/\lambda_2)]^{-1}, \tag{3.38}$$

where $\lambda_1, \lambda_2, \lambda_3$ are the wavelengths for each frequency, and $N_1, N_2, N_3$ are mode indexes of the guided wave. Using (5.28), (3.2), (3.36)–(3.38), we obtain coupled-mode equations:

$$\frac{d}{dz} A_1(z) = -j \kappa_1 A_3(z) A_2(z)^* \exp(-j 2\Delta z), \tag{3.39a}$$

$$\frac{d}{dz} A_2(z) = -j \kappa_2 A_3(z) A_1(z)^* \exp(-j 2\Delta z), \tag{3.39b}$$

$$\frac{d}{dz} A_3(z) = -j \kappa_3 A_1(z) A_2(z) \exp(+j 2\Delta z), \tag{3.39c}$$

where

$$2\Delta = \beta_3 - (\beta_1 + \beta_2 + qK) \tag{3.40}$$

is the deviation from the exact phase matching, and the coupling coefficients are defined by

$$\kappa_1 = \frac{\omega_1 \epsilon_0}{2} \iint [E_1(x, y)]^* d_q(x, y) E_3(x, y) [E_2(x, y)]^* dx dy , \tag{3.41a}$$

$$\kappa_2 = \frac{\omega_2 \epsilon_0}{2} \iint [E_2(x, y)]^* d_q(x, y) E_3(x, y) [E_1(x, y)]^* dx dy , \tag{3.41b}$$

$$\kappa_3 = \frac{\omega_3 \epsilon_0}{2} \iint [E_3(x, y)]^* d_q(x, y) E_1(x, y) E_2(x, y) dx dy . \tag{3.41c}$$

Using the permutation symmetry of $d_q$, we see that the coupling coefficients are correlated by

$$\kappa_1 / \omega_1 = \kappa_2 / \omega_2 = \kappa_3^* / \omega_3 . \tag{3.42}$$

Although the above expressions are given for $q$-th order PQM, the expressions can be used for other guided-mode phase matching schemes by simply putting $q=0$.

From the coupled-mode equations (3.39), together with $\omega_3 = \omega_1 + \omega_2$ and (3.42), we obtain

$$\frac{d}{dz} \left( |A_1(z)|^2 + |A_2(z)|^2 + |A_3(z)|^2 \right) = 0 . \tag{3.43}$$

This relation shows the conservation of total power. We also obtain from (3.39) and (3.42)

$$\frac{d}{dz} (|A_1(z)|^2 / \omega_1 + |A_3(z)|^2 / \omega_3) = 0 , \tag{3.44a}$$

$$\frac{d}{dz} (|A_2(z)|^2 / \omega_2 + |A_3(z)|^2 / \omega_3) = 0 . \tag{3.44b}$$

Since $|A_i(z)|^2 / \omega_i$ gives photon flow (the number of photons per unit time) for the wave of frequency $\omega_i$, (3.44) implies that creation of a number of $\omega_3$ photon requires annihilation of the same number of $\omega_1$ photon and $\omega_2$ photon. In other words, one $\omega_1$ photon and one $\omega_2$ photon are merged to generated one $\omega_3$ photon. This is an important fundamental relation, which dominates the general feature of NLO interactions, and is called the Manley-Rowe relations. It is interesting to note that the quantum characteristic of optical waves is derived through the classic discussion described above.

### Interaction Characteristics

Consider waveguide SFG with interaction region in $0 \le z \le L$. The device structure is the same as that in Fig. 3.1, except that the grating period is chosen so as to satisfy the QPM condition (3.38) for SFG. The coupled-mode equations (3.37) should be solved with boundary conditions:

$$A_1(0) = A_{10} , \quad A_2(0) = A_{20} , \quad A_3(0) = 0 , \tag{3.45}$$

where $A_{10}$ and $A_{20}$ are the amplitudes of the incident waves.

Consider first a case where the incident power of the $\omega_2$ wave is much larger than that of the $\omega_1$ wave, i.e., $|A_{10}| \ll |A_{20}|$. Then from the Manley-Rowe relations we have $|A_2(z)| \gg |A_1(z)|, |A_3(z)|$, and it follows through the power conservation that $A_2(z)$ does not vary significantly with $z$ and therefore we can approximate $A_2(z)$ as $A_2(z) \equiv A_{20}$. This treatment may be called no pump depletion approximation (NPDA), if the strong $\omega_2$ wave is considered a pump wave. Then the coupled-mode equations (3.39) are reduced to

$$ \frac{d}{dz} A_1(z) = -j\kappa_1 A_{20}^* A_3(z) \exp(-j2\Delta z) , \tag{3.46a} $$
$$ \frac{d}{dz} A_3(z) = -j\kappa_3 A_{20} A_1(z) \exp(+j2\Delta z) . \tag{3.46b} $$

These equations are linear differential equations and can readily be solved with the boundary conditions (3.45) to yield

$$ A_1(z) = A_{10} \exp(-j\Delta z) \times \left[ \cos \sqrt{(\omega_1 / \omega_3)\kappa^2 P_{20} + \Delta^2} z + \frac{j\Delta}{\sqrt{(\omega_1 / \omega_3)\kappa^2 P_{20} + \Delta^2}} \sin \sqrt{(\omega_1 / \omega_3)\kappa^2 P_{20} + \Delta^2} z \right] , \tag{3.47a} $$
$$ A_3(z) = A_{10} \exp(+j\Delta z) \times \frac{-j\kappa A_{20}}{\sqrt{(\omega_1 / \omega_3)\kappa^2 P_{20} + \Delta^2}} \sin \sqrt{(\omega_1 / \omega_3)\kappa^2 P_{20} + \Delta^2} z , \tag{3.47b} $$

where $P_{20} = |A_{20}|^2$ is the input pump power. To obtain (3.47), use has been made of (3.42), $\kappa_3$ has been rewritten as $\kappa$, and, without loss of generality, $\kappa$ was assumed to be real. It should be noted that $\kappa$ for SFG is different from $\kappa$ for SHG; when $\omega_1 = \omega_2 \equiv \omega, \omega_3 = 2\omega$ and therefore from (3.10), (3.15) and (3.41c), $\kappa_{SFG} = \kappa_3 = 2\kappa_{SHG}$. From (3.47), we obtain NPDA expressions for the output powers of the three waves:

$$ P_1(L) = |A_1(L)|^2 = P_{10} - (\omega_1 / \omega_3) P_3(L) , \tag{3.48a} $$
$$ P_2(L) = |A_2(L)|^2 = P_{20} - (\omega_2 / \omega_3) P_3(L) , \tag{3.48b} $$
$$ P_3(L) = |A_3(L)|^2 = P_{10} P_{20} \kappa^2 L^2 \left[ \frac{\sin \sqrt{(\omega_1 / \omega_3)\kappa^2 P_{20} + \Delta^2} L}{\sqrt{(\omega_1 / \omega_3)\kappa^2 P_{20} + \Delta^2} L} \right]^2 , \tag{3.48c} $$

where $P_{10} = |A_{10}|^2$ is the input $\omega_1$ power (signal power). To obtain (3.48b), use has been made of Manley-Rowe relations (3.44).

The dependence of $P_1(L), P_2(L), P_3(L)$ upon the normalized interaction length $\kappa \sqrt{(\omega_1 / \omega_3) P_{20}} L$ described by (3.48) is shown in Fig. 3.6 with the normalized phase mismatch $|\Delta| / \kappa \sqrt{(\omega_1 / \omega_3) P_{20}}$ as a parameter. Under exact (quasi-)phase matching ($\Delta = 0$), (3.48c) reduces to

![Fig. 3.6. Dependence of signal, pump and sum-frequency powers on interaction length](figure_3_6.png)
**Fig. 3.6.** Dependence of signal, pump and sum-frequency powers on interaction length

$$P_3(L) = (\omega_3 / \omega_1) P_{10} \sin^2 \kappa \sqrt{(\omega_1 / \omega_3) P_{20}} L . \tag{3.49}$$

This result shows that the sum-frequency (SF) power $P_3(L)$ is a periodic function of interaction length $L$. It increases first with $L$, and after reaching at $(\omega_3 / \omega_1) P_{10}$ it starts to decrease. This is because SFG from the $\omega_1$ and $\omega_2$ waves gives rise to depletion of $\omega_1$ and $\omega_2$ powers. When the $\omega_1$ power is completely depleted, there is no $\omega_1$ photon available for SFG, and in turn difference frequency ($\omega_1 = \omega_3 - \omega_2$) generation from $\omega_2$ and $\omega_3$ photons takes place. This process is repeated.

For small $\kappa \sqrt{(\omega_1 / \omega_3) P_{20}} L$, (3.49) reduces to
$$P_3(L) = P_{10} P_{20} \kappa^2 L^2 , \tag{3.50}$$
and the output SF power is proportional to the product of the input powers. The normalized wavelength conversion efficiency is $P_3(L) / P_{10} P_{20} = \kappa^2 L^2$. For small $\kappa \sqrt{P_{20}} L$, the $\{ \}^2$ factor of (3.48c) can be approximated by $\text{sinc}^2 (\Delta L)$. Therefore the 3-dB full bandwidth of phase matching is given by $|\Delta| < 1.39 / L$. It is similar to the SHG case that for large $\kappa \sqrt{P_{20}} L$ the bandwidth depends upon $\kappa \sqrt{P_{20}} L$.

The nonlinear coupled-mode equations (3.39) were solved for arbitrary boundary conditions by Armstrong, et al. [3.5]. Using the Manley-Rowe relations (3.44) to eliminate $A_1(z)$ and $A_2(z)$, (3.39c) can be integrated to obtain $A_3(z)$. The output SF power for (quasi-) phase matched cases ($\Delta=0$) of boundary conditions (3.45) with $|A_{10}| \le |A_{20}|$ is given by
$$P_3(L) = (\omega_3 / \omega_1) P_{10} \text{sn}^2 [\kappa \sqrt{(\omega_1 / \omega_3) P_{20}} L ; \gamma] , \quad \gamma = \sqrt{(P_{10} / \omega_1) / (P_{20} / \omega_2)} , \tag{3.51}$$
where $\text{sn}[\zeta ; \gamma]$ is the Jacobian elliptic function defined by (3.21). The other output powers, $P_1(L)$ and $P_2(L)$, are given by (3.48a) and (3.48b) with (3.51). For $P_{10} \ll P_{20}$ ($\gamma \ll 1$), (3.51) reduces to (3.49).

A special case of interest is $P_{10} / \omega_1 = P_{20} / \omega_2$ so that $\gamma = 1$. Then (3.48a), (3.48b) and (3.51) reduce to

$$
P_1(L) = P_{10} \text{sech}^2(\kappa \sqrt{(\omega_1 / \omega_3) P_{20}} L), \tag{3.52a}
$$
$$
P_2(L) = P_{20} \text{sech}^2(\kappa \sqrt{(\omega_1 / \omega_3) P_{20}} L), \tag{3.52b}
$$
$$
P_3(L) = (P_{10} + P_{20}) \tanh^2(\kappa \sqrt{(\omega_1 / \omega_3) P_{20}} L) . \tag{3.52c}
$$

The above equations imply that asymptotically complete power conversion to the SF wave is possible for large input powers or large interaction length, provided that the input waves are of equal photon flow.

### 3.1.3 Difference-Frequency Generation

When optical waves of two frequencies $\omega_1$ and $\omega_3$ ($\omega_3 > \omega_1$) are incident in the same device considered in the previous subsection, an optical wave of frequency $\omega_2 = \omega_3 - \omega_1$ may be generated. This is called difference-frequency generation (DFG) or parametric down-conversion. DFG is associated with amplification of the power of the $\omega_1$ wave. This is called optical parametric amplification and is presented in the next subsection.

#### Coupled-Mode Equations

DFG, being an interaction of three optical waves of different frequencies, can be analyzed based on the coupled-mode formalism given by (3.35)–(3.42). Let $\omega_1$ and $\omega_3$ ($\omega_3 > \omega_1$) be the angular frequencies of the incident waves, and $\omega_2 = \omega_3 - \omega_1$ be the difference frequency, and assume $\omega_2 \neq \omega_1$. Then the coupled-mode equations and the (quasi-)phase matching conditions are given by exactly the same equations as those for SFG. The $\omega_1$ and $\omega_3$ waves are called signal and pump waves, respectively. The total power conservation (3.43) and Manley-Rowe relations (3.44) also hold for DFG. Manley-Rowe relations indicate that creation of a number of $\omega_2$ photons requires annihilation of the same number of $\omega_3$ photons and is associated with creation of the same number of $\omega_1$ photons. In other words, one $\omega_3$ photon is cracked to be one $\omega_2$ photon and one $\omega_1$ photon.

#### Interaction Characteristics

Consider waveguide DFG with an interaction region in $0 \le z \le L$. The coupled-mode equations (3.39) should be solved with boundary conditions:

$$
A_1(0) = A_{10}, \quad A_2(0) = 0, \quad A_3(0) = A_{30}, \tag{3.53}
$$

where $A_{10}$ and $A_{30}$ are the amplitudes of the incident signal and pump waves.

Consider first a case where the incident pump power wave is much larger than the signal power, i.e., $|A_{10}| \ll |A_{30}|$, and the DFG conversion efficiency is low. Then we can assume $|A_3(z)| \gg |A_1(z)|, |A_2(z)|$ and approximate $A_3(z)$ as $A_3(z) \cong A_{30}$. This treatment may be called the no pump depletion approximation (NPDA). Then, the coupled mode equations (3.39) are reduced to

$$
\frac{d}{dz} A_1(z) = -j \kappa_1 A_{30} A_2(z)^* \exp(-j 2 \Delta z) , \tag{3.54a}
$$
$$
\frac{d}{dz} A_2(z) = -j \kappa_2 A_{30} A_1(z)^* \exp(-j 2 \Delta z) . \tag{3.54b}
$$

These equations are linear differential equations and can readily be solved with the boundary conditions (3.53) to yield

$$
A_1(z) = A_{10} \exp(-j \Delta z) \times \left[ \cosh \sqrt{(\omega_1 / \omega_2) \kappa^2 P_{30} - \Delta^2} z + \frac{j \Delta}{\sqrt{(\omega_1 / \omega_2) \kappa^2 P_{30} - \Delta^2}} \sinh \sqrt{(\omega_1 / \omega_2) \kappa^2 P_{30} - \Delta^2} z \right] , \tag{3.55a}
$$
$$
A_2(z) = A_{10}^* \exp(-j \Delta z) \times \frac{-j \kappa A_{30}}{\sqrt{(\omega_1 / \omega_2) \kappa^2 P_{30} - \Delta^2}} \sinh \sqrt{(\omega_1 / \omega_2) \kappa^2 P_{30} - \Delta^2} z , \tag{3.55b}
$$

where $P_{30} = |A_{30}|^2$ is the input pump power. To obtain (3.55), use has been made of (3.42), $\kappa_2$ has been rewritten as $\kappa$ and, without loss of generality, $\kappa$ was assumed to be real. It should be noted that $\kappa$ for DFG is different from $\kappa$ for SFG. When $\omega_1 \cong \omega_2 \cong \omega$, $\omega_3 \cong 2\omega$ and therefore, from (3.10), (3.15), and (3.41b), $\kappa_{\text{DFG}} = \kappa_2 \cong \kappa_{\text{SHG}} \cong \kappa_{\text{SFG}} / 2$. It is important to note that the DF wave amplitude $A_2(z)$ is proportional to the complex conjugate of the incident signal wave amplitude. From (3.55), we obtain NPDA expressions for the output powers of the three waves:

$$
P_1(L) = |A_1(L)|^2 = P_{10} + (\omega_1 / \omega_2) P_2(L) , \tag{3.56a}
$$
$$
P_2(L) = |A_2(L)|^2 = P_{10} P_{30} \kappa^2 L^2 \left[ \frac{\sinh \sqrt{(\omega_1 / \omega_2) \kappa^2 P_{30} - \Delta^2} L}{\sqrt{(\omega_1 / \omega_2) \kappa^2 P_{30} - \Delta^2} L} \right]^2 , \tag{3.56b}
$$
$$
P_3(L) = |A_3(L)|^2 = P_{30} - (\omega_3 / \omega_2) P_2(L) , \tag{3.56c}
$$

where $P_{10} = |A_{10}|^2$ is the input $\omega_1$ power (signal power). To obtain (3.56c), use has been made of Manley-Rowe relations (3.44).

Under exact (quasi-) phase matching ($\Delta=0$), (3.56b) reduces to

$$
P_2(L) = (\omega_2 / \omega_1) P_{10} \sinh^2 \Gamma L , \tag{3.57}
$$

where

$$
\Gamma = \kappa \sqrt{(\omega_1 / \omega_2) P_{30}} . \tag{3.58}
$$

This result shows that the difference-frequency (DF) power $P_2(L)$ increases monotonically with increasing $\Gamma L$. The DFG conversion efficiency $\eta = P_2(L) / P_{10}$ can be larger than 100%; even a substantial gain can be obtained with large $\Gamma L$. This is in contrast to SFG where the efficiency $\eta = P_3(L) / P_{10}$ is limited up to the maximum $\omega_3 / \omega_1$. For small $\Gamma L$, (3.57) reduces to

![Fig. 3.7. Dependence of signal and idler powers upon interaction length](https://vlm.ai/placeholder.png)

**Fig. 3.7.** Dependence of signal and idler powers upon interaction length

$$P_2(L) = (\omega_2 / \omega_1) P_{10} \Gamma^2 L^2 = P_{10} P_{30} \kappa^2 L^2, \tag{3.59}$$

and the output DF power is proportional to the product of the input powers. The normalized wavelength conversion efficiency is $P_2(L)/P_{10}P_{30} = \kappa^2 L^2$. The dependence of $P_1(L)$, $P_2(L)$, described by (3.56), upon normalized interaction length $\Gamma L$ is shown in Fig. 3.7 with normalized phase mismatch $|\Delta| / \Gamma$ as a parameter.

The phase matching bandwidth can be evaluated by using (3.56b). For small $\Gamma L$ (small efficiency limit), the $|\text{sinc}|^2$ factor of (3.56b) can be approximated by $\text{sinc}^2(\Delta L)$, and the full bandwidth is given by

$$|\Delta| < 1.39 / L. \tag{3.60}$$

For large $\Gamma L$, (3.60) may be expanded as

$$|\Delta| < \sqrt{(1.39 / L)^2 + (\omega_1 / \omega_2) \kappa^2 P_{30}}. \tag{3.61}$$

This means that the bandwidth is substantially broadened for the high gain case.

The nonlinear coupled-mode equations (3.39) can be solved for the DFG boundary conditions (3.53) similarly to SFG. Using the Manley-Rowe relations (3.44) to eliminate $A_1(z)$ and $A_3(z)$, (3.39b) can be integrated to obtain $A_2(z)$. The output DF power for phase matched cases ($\Delta = 0$) is given by

$$P_2(L) = -(\omega_2 / \omega_1) P_{10} \text{sn}^2 [j \kappa \sqrt{(\omega_1 / \omega_3) P_{30}} L; \gamma], \quad \gamma = j \sqrt{(P_{10} / \omega_1) / (P_{30} / \omega_3)}, \tag{3.62}$$

where $\text{sn}[\zeta; \gamma]$ is the Jacobian elliptic function defined by (3.21). The other output powers, $P_1(L)$ and $P_3(L)$, are given by (3.56a) and (3.56c) with (3.62). For $P_{10} \ll P_{30}$ ($|\gamma| \ll 1$), (3.62) reduces to (3.57).

### 3.1.4 Optical Parametric Amplification

The discussion in the previous subsection shows that, when optical waves of two frequencies $\omega_1$ and $\omega_3$ ($\omega_3 > \omega_1$) are incident, the power of the $\omega_1$ wave is ampli-

fied. This is called optical parametric amplification (OPA). OPA takes place simultaneously with DFG. When OPA is discussed, the $\omega_1$ and $\omega_3$ waves are called signal and pump waves, respectively, and the $\omega_2$ wave is called an idler wave.

### Nondegenerate Optical Parametric Amplification

Mathematical expressions to describe OPA with $\omega_2 \neq \omega_1$ have been derived in the previous subsections. The phase matching condition for OPA is exactly the same as that for SFG and DFG. The dependence of the relevant optical powers upon normalized interaction length $\Gamma L$ is shown in Fig. 3.7 with normalized phase mismatch $|\Delta| / \Gamma$ as a parameter. The NPDA power gain is obtained from (3.56a) and (3.56b) and is given by

$$G = \frac{P_1(L)}{P_{10}} = 1 + (\omega_1 / \omega_2) \kappa^2 P_{30} L^2 \left[ \frac{\sinh \sqrt{(\omega_1 / \omega_2) \kappa^2 P_{30} - \Delta^2} L}{\sqrt{(\omega_1 / \omega_2) \kappa^2 P_{30} - \Delta^2} L} \right]^2 \tag{3.63}$$

Under exact (quasi-)phase matching ($\Delta=0$), (3.63) reduces to

$$G = \cosh^2 \Gamma L , \tag{3.64}$$

where $\Gamma = \kappa \sqrt{(\omega_1 / \omega_2) P_{30}} = \kappa_2 \sqrt{(\omega_1 / \omega_2) P_{30}} = \kappa_1 \sqrt{(\omega_2 / \omega_1) P_{30}} = \sqrt{\kappa_1 \kappa_2 P_{30}}$ is a gain factor. For large $\Gamma L$ (high gain limit), (3.64) may be approximated as

$$G = (1/4) \exp 2\Gamma L . \tag{3.65}$$

The phase matching bandwidth can be evaluated by using (3.63) and is given by (3.60) or (3.61). It should be noted that the bandwidth is broadened for the high gain case.

The OPA gain with the pump depletion considered for (quasi-)phase matched cases ($\Delta=0$) is obtained from (3.56a) and (3.62) and is given by

$$G = 1 - \text{sn}^2 [i \Gamma L; \gamma], \quad \gamma = i \sqrt{(P_{10} / \omega_1) / (P_{30} / \omega_3)} , \tag{3.66}$$

which reduces to (3.64) for $P_{10} \ll P_{30}$ ($|\gamma| \ll 1$).

OPA offers several advantages compared to widely used laser amplifiers. The most outstanding is that OPA with a given NLO material can cover a very wide wavelength range by appropriate design and/or adjustment of phase matching, whereas lasers cover only a narrow range around a specific wavelength depending upon the material. A unique feature of OPA is that amplification is unidirectional; the backward propagating signal wave is not amplified. This is in contrast to laser amplification, which is bidirectional. While laser amplification involves amplified spontaneous emission (ASE) noises, OPA is free from it and therefore is of low noise. The noise analysis will be presented in Chapter 5.

### Degenerate Optical Parametric Amplification

A special case of OPA, where $\omega_2 = \omega_1$, is called degenerate optical parametric amplification (DOPA). In this case, the signal wave of frequency $\omega = \omega_1 (= \omega_2)$ is

amplified by excitation with a pump wave of frequency $2 \omega\left(=\omega_{3}\right)$. This is an inverse process to SHG, and accordingly, DOPA can be analyzed based on the same coupled-mode formalism as SHG. The coupled-mode equations and the (quasi-) phase matching condition are given by (3.13) and (3.14), respectively. The coupled-mode equations (3.13) should be solved with boundary conditions:
$$A(0)=A_{0}, \quad B(0)=B_{0}, \tag{3.67}$$
where $A_{0}$ and $B_{0}$ are the amplitudes of the incident signal and pump waves, respectively.

Putting $B(z) \cong B_{0}$ in (3.13a) yields a NPDA coupled-mode equation for the signal wave amplitude:
$$\frac{d}{d z} A(z)=-j \kappa^{*} A^{*}(z) B_{0} \exp (-j 2 \Delta z) . \tag{3.68}$$
Then substituting
$$A(z)=\exp (-j \Delta z)[X(z)-j Y(z)], \quad A(z)^{*}=\exp (+j \Delta z)[X(z)+j Y(z)] \tag{3.69}$$
into (3.68) and its complex conjugate, we obtain
$$\frac{d}{d z} X(z)=\Gamma X(z)+\Delta Y(z), \quad \frac{d}{d z} Y(z)=-\Gamma Y(z)-\Delta X(z), \tag{3.70}$$
where $\Gamma=j \kappa B_{0}^{*}$ and, without loss of generality, $\Gamma$ is assumed to be positive. Equations (3.70) can readily be solved with (3.67), and the solution can be substituted into (3.69) to obtain $A(z)$. The result for $\Delta=0$ can be written as
$$A(z)=X_{0} \exp (+\Gamma z)-j Y_{0} \exp (-\Gamma z), \tag{3.71}$$
where $A_{0}=X_{0}-j Y_{0}$ with real $X_{0}$ and $Y_{0}$. The result shows that one of the quadratures of input signal, $X_{0}$, is amplified, while another, $Y_{0}$, is deamplified. In other words, the input signal is amplified (when $A_{0}$ is real) or deamplified (when $A_{0}$ is imaginary) depending upon the phase of input signal relative to the pump wave. Deamplification results from constructive interference of the second harmonic of the input signal wave and the input pump wave, which allow SHG from and depletion of the signal power. Thus DOPA is phase-sensitive amplification. This characteristic deserves special attention, since it allows unique applications such as noiseless amplification and squeezing, which will be discussed in Chapter 5.

The solutions of the coupled-mode equations (3.13) for phase-matched $(\Delta=0)$ DOPA with pump depletion considered can be obtained by extension of (3.22) [3.8], and for real $A_{0}$ and $B_{0}$ they are given by
$$\begin{gathered}
A(z)=A_{0} \cosh \zeta_{0} \operatorname{sech}\left(\zeta+\zeta_{0}\right), \quad B(z)=A_{0} \cosh \zeta_{0} \tanh \left(\zeta+\zeta_{0}\right), \\
\zeta=\left(|\kappa| \sqrt{A_{0}^{2}+B_{0}^{2}}\right) z, \quad \zeta_{0}=\sinh ^{-1}\left(B_{0} / A_{0}\right) .
\end{gathered} \tag{3.72}$$

### Counterpropagating Optical Parametric Amplification

OPA also takes place with counterpropagating $\omega_{1}$ and $\omega_{2}$ waves. Although the derivation of coupled-mode equations in Section 2.3 assumed forward (propagat-

ing in the $+z$ direction) waves, it can easily be seen from (2.12) and (2.52)–(2.58) that those for backward ($-z$ direction) waves are given by (2.58) with the sign of the left-hand side and the sign of $\beta_m$ inverted. Assume that the pump ($\omega_3$) wave and the $\omega_1$ wave are forward waves and the $\omega_2$ wave is backward wave. Then, the NPDA coupled-mode equations can be written as

$$
\begin{aligned}
+\frac{d}{dz} A_1(z) &= -j\kappa_1 A_{30} A_2(z)^* \exp(-j2\Delta z), & (3.73\text{a}) \\
-\frac{d}{dz} A_2(z) &= -j\kappa_2 A_{30} A_1(z)^* \exp(-j2\Delta z), & (3.73\text{b})
\end{aligned}
$$

$$
2\Delta = \beta_3 - (\beta_1 - \beta_2) - K. \quad (3.73\text{c})
$$

The QPM condition is given by $\Delta = 0$ with (3.73c). The coupled-mode equations (3.73) can readily be solved for boundary conditions $A_1(0) = A_{10}$ and $A_2(L) = A_{20}$, where $L$ is the interaction length. The output amplitudes for $\Delta = 0$ are given by

$$
\begin{aligned}
A_1(L) &= A_{10} / \cos \Gamma L - j \sqrt{\omega_1/\omega_2} (A_{30}/|A_{30}|) A_{20}^* \tan \Gamma L, & (3.74\text{a}) \\
A_2(0) &= A_{20} / \cos \Gamma L - j \sqrt{\omega_2/\omega_1} (A_{30}/|A_{30}|) A_{10}^* \tan \Gamma L, & (3.74\text{b})
\end{aligned}
$$

$$
\Gamma = \sqrt{\kappa_1 \kappa_2 P_{30}} = \kappa_1 \sqrt{(\omega_2/\omega_1) P_{30}} = \kappa_2 \sqrt{(\omega_1/\omega_2) P_{30}}. \quad (3.74\text{c})
$$

The result shows that both forward and backward waves are amplified. The gain can be larger than that for copropagating OPA of the same $\Gamma$ and $L$. It is more important to note that when $\Gamma L = \pi/2$ both $A_1(L)$ and $A_2(0)$ can be nonzero for zero input ($A_{10} \to 0, A_{20} \to 0$). This is because the two waves are amplified in opposite directions and give positive feedback to each other through the NLO interaction. This implies that, with counterpropagating OPA, mirrorless oscillators are feasible [3.9]. The oscillation threshold is given by $\Gamma L = \pi/2$. Thus counterpropagating OPA offers attractive features, although the device implementation may be difficult because of the very fine period required for QPM.

## 3.2 Quasi-Phase Matching by Chirped Grating

In the QPM NLO interactions, exact phase matching can be accomplished, in principle, with a grating having a period to compensate for the difference between the propagation constants. In practice, however, many factors, e.g., uncertainty of the period due to limited accuracy of material constants, fabrication errors, and change of the propagation constants due to grating fabrication, give rise to residual mismatch. Working conditions also affect phase matching; change of ambient temperature, deviation and fluctuation of laser-diode wavelength, and photorefractive damage cause deviation from matching. It is therefore very important to design devices such that the residual mismatch and the deviation are tolerable.
