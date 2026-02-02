# Coupled-Wave Equations for Cascaded THG
$$
\begin{aligned}
% --- A1 (Fundamental) ---
\frac{\partial A_1}{\partial z} + \frac{1}{v_{g1}} \frac{\partial A_1}{\partial t} + \frac{i \beta_{2,1}}{2} \frac{\partial^2 A_1}{\partial t^2} &= i \left[ \kappa_{SHG}(z) A_2 A_1^* e^{i\Delta k_{SHG} z} + \kappa_{SFG}(z) A_3 A_2^* e^{i\Delta k_{SFG} z} \right] \\
\\
% --- A2 (SHG/SFG Pump) ---
\frac{\partial A_2}{\partial z} + \frac{1}{v_{g2}} \frac{\partial A_2}{\partial t} + \frac{i \beta_{2,2}}{2} \frac{\partial^2 A_2}{\partial t^2} &= i \left[ \kappa_{SHG}(z) A_1^2 e^{-i\Delta k_{SHG} z} + 2 \kappa_{SFG}(z) A_3 A_1^* e^{i\Delta k_{SFG} z} \right] \\
\\
% --- A3 (SFG Output) ---
\frac{\partial A_3}{\partial z} + \frac{1}{v_{g3}} \frac{\partial A_3}{\partial t} + \frac{i \beta_{2,3}}{2} \frac{\partial^2 A_3}{\partial t^2} &= i \left[ 3 \kappa_{SFG}(z) A_1 A_2 e^{-i\Delta k_{SFG} z} \right]
\end{aligned}
$$

# Coupled-Wave Equations for SHG in CMT
$$\begin{aligned}

\frac{\partial A_1}{\partial z} + \frac{1}{v_{g1}} \frac{\partial A_1}{\partial t} + \frac{i \beta_{2,1}}{2} \frac{\partial^2 A_1}{\partial t^2} &= i \kappa(z) A_2 A_1^* e^{-i\Delta k z} \\

\frac{\partial A_2}{\partial z} + \frac{1}{v_{g2}} \frac{\partial A_2}{\partial t} + \frac{i \beta_{2,2}}{2} \frac{\partial^2 A_2}{\partial t^2} &= i \kappa(z) A_1^2 e^{+i\Delta k z}

\end{aligned}$$