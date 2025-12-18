# Overview
Study of **Buried APE Waveguides** in PP MgO:SLT fabricated via **Surface-Activated Bonding (SAB)** for **SHG Devices** (1064 nm \rightarrow 532 nm). Includes **Proton Diffusion Analysis** and **Waveguide Simulation**.
# Proton Diffusion Analysis
**Model & Equations**
* Diffusion equation:
* PE depth (d_{PE}):
* Annealing profile (C_a(z)):
* Annealing depth (d_a):
* 2D Diffusion (Channel):
* Refractive index increase:
**Parameters (SIMS Measurement)**
* Material: +Z-MgO:SLT.
* PE Condition: Pure pyrophosphoric acid, 230^{\circ}C.
* PE Diffusion Coeff (D_{PE}): 0.045~\mu m^{2}h^{-1} (Measured) vs CLT 0.08~\mu m^{2}h^{-1}.
* Annealing Condition: 400^{\circ}C, dry O_2.
* Annealing Diffusion Coeff (D_a): 1.3~\mu m^{2}h^{-1} (Z-axis).
* Anisotropy assumption: D_x, D_y = D_z / 1.5.
# Waveguide Simulation
**Conditions**
* Wavelength: 1030 nm.
* $\Delta n_0$: 0.012.
* Target mode diameter: >30~\mu m (for Watt-class power).
* Lattice constant increase: 0.57% (assumed same as CLT).
**Optimal Configuration**
* PE width: 50~\mu m.
* PE time: 8 h.
* Annealing time: 100 h.
* Resulting mode (1/e^2): 36~\mu m (width) \times 30~\mu m (depth).
* Projected 532 nm mode: 23~\mu m \times 19~\mu m, Area 3.4\times10^{2}\mu m^{2}.
* Power capacity: 3.4 W (below 2 MW cm^{-2} threshold).
# Buried APE Waveguide Fabrication
**Process**
1. 
**Mask:** Ta (100 nm), opening widths 20-60~\mu m (actual 18-58~\mu m).
2. 
**PE:** Pyrophosphoric acid, 230^{\circ}C (4, 8, 16 h).
* Protrusion: 7-10 nm.
3. **SAB (Bonding):**
* Activation: Ar fast atomic beam (FAB), 1.0 kV, 200 mA, 100 s.
* Bonding: Load 10 MPa.
* Heat: 110^{\circ}C (1.5 h hold), total press 6.0 h.
4. 
**Annealing:** 400^{\circ}C (25-100 h).
**Measurements (1032 nm)**
* PE 48~\mu m, 8 h, Anneal 100 h:
* Mode: 32~\mu m (width) \times 30~\mu m (depth).
* Width discrepancy attributed to D_y variation or residual air layer.
**Measurements (532 nm)**
* PE 48~\mu m, 8 h, Anneal 100 h:
* Mode: 20~\mu m (width) \times 15~\mu m (depth).
* Symmetric profile obtained.
# SHG Device
**Fabrication**
* 
**Substrate:** 1.0 mol% MgO:SLT, 0.5 mm thick, 17.5\times9.0~mm^{2} chips.
* **PP Structure:**
* Period \Lambda: 8.0~\mu m.
* Electrodes: +Z corrugated Au (250 nm); -Z SiO_2 (1.0~\mu m) + flat Au.
* Poling: 120^{\circ}C, 1428 V/mm, pulse 0.05 ms, period 0.15 ms.
* Charge Q: 32~\mu C (P_s=55~\mu C~cm^{-2}).
* 
**Alignment Strategy:** One crystal uniform PP; other crystal 8 sets shifted by 1.0~\mu m to mitigate longitudinal misalignment.
* Effective d_{eff} \ge (2/\pi)\times cos(\pi/16)\times d_{33}.
* 
**Final Assembly:** PE (48~\mu m, 8 h) \rightarrow SAB (<0.04^{\circ} rot. misalignment) \rightarrow Anneal (400^{\circ}C, 90 h).
* 
**Device Length:** 14.2 mm.
**Performance (1064 nm Pump)**
* Peak Temp: 25.5^{\circ}C (Theory 31.5^{\circ}C), FWHM 2.5^{\circ}C.
* Mode Size (1/e^2): Pump 31 \times 31~\mu m; SH 19 \times 16~\mu m.
* Normalized Efficiency \eta_{norm}: 4.8% W^{-1}.
* Nonlinear Coupling Coefficient: 0.15~W^{-1/2}cm^{-1}.
* Max SH Power: 6.7 mW (at ~400 mW pump).
* Projection: 2 W SH output at 10 W pump.