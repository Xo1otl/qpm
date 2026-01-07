# SATO's Program

入力: 基本波波長の誘電率分布 (実数屈折率の二乗)  
制約: 境界条件と波長・誘電率の関係式
出力: $\kappa$  
手順: SH波波長の誘電率分布 -> 電界分布 -> $\kappa$  

# 結果の見方
* `Eps_fw` : 基本波波長に対する誘電率分布
* `Eps_shw`: SH波波長に対する誘電率分布
* `Ey_fw`  : 基本波波長に対するTM00の電界分布
* `Ey_shw` : SH波波長に対するTM03の電界分布
* `kappa`  : 重なり積分値 ($\kappa_{s14}$)

# Code
```matlab
addpath('refractiveindex\');

% --- Configuration ---
dx = 0.01; dy = 0.01;   % Grid step
wl_fw = 0.8;            % Fundamental wavelength (µm)
Temp  = 20;             % Temperature
h_core = 1.4;           % Core height
h_slab = 0;             % Slab height
w = 1.2;                % Rib width

% --- Initialization ---
% Get grid dimensions (dummy call)
geom_args = {h_slab, h_core, w, w, dx, dy}; 
[~, nx, ny] = waveguidemeshfull_trapezoid_rib(1,1,1,1,1, geom_args{:});

% Data Containers
Eps_fw = []; Ey_fw  = []; % Fundamental (TM00)
Eps_shw= []; Ey_shw = []; % SH (TM03)
eps_core_shw = 0;         % Core permittivity at SH

% --- Simulation Loop ---
for k = 1:2
    wl = wl_fw / k;     % Current wavelength (fw or shw)
    n_modes = [5, 30];  % Search depth
    
    % 1. Get Permittivity Tensor (Diagonal elements for substrate(1), core(2), clad(5))
    [e1x,e1y,~,~,e1z, e2x,e2y,~,~,e2z, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, e5x,e5y,~,~,e5z, g] = define_eps_SL(wl, 0, 0, Temp);
    
    % 2. Build 2D Permittivity Mesh
    [Ex,~] = waveguidemeshfull_trapezoid_rib(sqrt(e1x),sqrt(e1x),sqrt(e2x),sqrt(e5x),sqrt(e5x), geom_args{:});
    [Ey,~] = waveguidemeshfull_trapezoid_rib(sqrt(e1y),sqrt(e1y),sqrt(e2y),sqrt(e5y),sqrt(e5y), geom_args{:});
    [Ez,~] = waveguidemeshfull_trapezoid_rib(sqrt(e1z),sqrt(e1z),sqrt(e2z),sqrt(e5z),sqrt(e5z), geom_args{:});

    % Save Maps
    if k == 1
        Eps_fw = Ey; 
    else
        Eps_shw = Ey; 
        eps_core_shw = e2y; 
    end

    % 3. Solve Modes
    [Hx, Hy, neff] = wgmodes(wl, g, n_modes(k), dx, dy, Ex, Ey, Ez, '0000');

    % 4. Extract Target Modes
    for m = 1:n_modes(k)
        if polarization(dx, dy, Hx(:,:,m), Hy(:,:,m)) == 2 % Check for TM
            [mx, my] = modeindex(Hx(:,:,m), Ez, e2z);
            
            % Target: Fund=TM00 (1,1), SH=TM03 (1,4)
            if (k == 1 && mx == 1 && my == 1) || (k == 2 && mx == 1 && my == 4)
                [hz, ex, ey, ez] = postprocess(wl, real(neff(m)), Hx(:,:,m), Hy(:,:,m), dx, dy, Ex, Ey, Ez, '0000');
                [~, ey, ~] = normalize(dx, dy, ex, ey, ez, Hx(:,:,m), Hy(:,:,m), hz);
                
                if k == 1, Ey_fw = ey; else, Ey_shw = ey; end
            end
        end
    end
end

% --- Overlap Integral ---
kappa = 0;
if ~isempty(Ey_fw) && ~isempty(Ey_shw)
    kappa = fullreverseintegral(dx, dy, Ey_fw, Ey_shw, Eps_shw, eps_core_shw, nx, ny, wl_fw);
end
```
