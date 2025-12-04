function [adjusted_thickness, boundary_conditions] = processSubgradeWinkler(thickness, config, load_params)
% PROCESSSUBGRADEWINKLER - Winkler elastic foundation model for subgrade processing
% This function implements the Winkler spring foundation model for subgrade
% analysis with dynamic range correction and enhanced physical interpretation
%
% Inputs:
%   thickness    - Layer thickness vector [surface; base; subbase; subgrade] (cm)
%   config       - Configuration parameters (optional)
%   load_params  - Load parameters structure containing soil_modulus field
%
% Outputs:
%   adjusted_thickness    - Adjusted thickness vector for pavement layers only
%   boundary_conditions   - Boundary conditions structure for PDE modeling
%
% Theory:
%   Based on Winkler elastic foundation theory with comprehensive correction
%   k = 0.65 × Es/(B^0.5×(1-γ²)) × CF
%   where CF is the comprehensive correction factor

fprintf('=== Winkler Elastic Foundation Model Processing ===\n');

try
    % === 1. Extract soil parameters ===
    if isstruct(load_params) && isfield(load_params, 'soil_modulus')
        Es = load_params.soil_modulus; % MPa
        fprintf('Soil modulus Es: %.0f MPa\n', Es);
    else
        Es = 50; % Default value
        fprintf('Warning: Using default soil modulus: %.0f MPa\n', Es);
    end
    
    % === 2. Standard theoretical parameters ===
    B = 0.4;        % Pavement width (m), theoretical value
    gamma = 0.40;   % Soil Poisson's ratio, theoretical value
    
    fprintf('Theoretical parameters: B=%.1f m, γ=%.2f\n', B, gamma);
    
    % === 3. Calculate base spring coefficient ===
    Es_Pa = Es * 1e6; % MPa -> Pa
    k_base = 0.65 * Es_Pa / (sqrt(B) * (1 - gamma^2));
    
    fprintf('Base spring coefficient: k_base = %.2e N/m³\n', k_base);
    
    % === 4. Calculate comprehensive correction factor CF ===
    CF = calculateComprehensiveCorrectionFixed(Es, thickness, load_params);
    fprintf('Comprehensive correction factor: CF = %.3f\n', CF);
    
    % === 5. Final spring coefficient with soil dependency ===
    k_winkler = k_base * CF;
    
    % Dynamic range check based on Es
    k_min_Es = Es * 1e6 * 0.1;   % Minimum = 10% of Es (Pa)
    k_max_Es = Es * 1e6 * 50;    % Maximum = 50x Es
    k_min = max(k_min_Es, 1e6);  % Absolute lower limit 1 MN/m³
    k_max = min(k_max_Es, 1000e6); % Absolute upper limit 1000 MN/m³
    
    % Clip to dynamic range
    k_original = k_winkler;
    k_winkler = max(k_winkler, k_min);
    k_winkler = min(k_winkler, k_max);
    
    was_clipped = (k_original < k_min) || (k_original > k_max);
    if was_clipped
        fprintf('Warning: Spring coefficient clipped from %.2e to %.2e N/m³\n', ...
            k_original, k_winkler);
    end
    
    fprintf('Final spring coefficient: k = %.2e N/m³\n', k_winkler);
    
    % === 6. Thickness adjustment (Winkler model uses pavement layers only) ===
    if length(thickness) >= 3
        adjusted_thickness = thickness(1:3); % [surface; base; subbase]
    else
        adjusted_thickness = [12; 30; 20]; % Default structure
        fprintf('Warning: Using default pavement structure\n');
    end
    
    % === 7. Boundary conditions setup ===
    boundary_conditions = struct();
    boundary_conditions.method = 'winkler_springs';
    boundary_conditions.modeling_type = 'winkler_springs';
    boundary_conditions.spring_coefficient = k_winkler;
    boundary_conditions.k_winkler = k_winkler;
    boundary_conditions.soil_modulus = Es;
    boundary_conditions.subgrade_modulus = Es;
    boundary_conditions.subgrade_poisson = gamma;
    
    % Theoretical parameters record
    boundary_conditions.theory_params = struct(...
        'Es_MPa', Es, ...
        'B_m', B, ...
        'gamma', gamma, ...
        'CF', CF, ...
        'k_base', k_base, ...
        'k_original', k_original, ...
        'k_final', k_winkler, ...
        'k_min', k_min, ...
        'k_max', k_max, ...
        'was_clipped', was_clipped);
    
    boundary_conditions.theory_basis = 'Standard Winkler Theory (Enhanced)';
    boundary_conditions.formula = 'k = 0.65 × Es/(B^0.5×(1-γ²)) × CF';
    boundary_conditions.version = 'v2.0_dynamic_range';
    
    fprintf('Winkler modeling completed successfully\n');
    fprintf('====================================\n');
    
catch ME
    fprintf('Error: Winkler modeling failed: %s\n', ME.message);
    
    % Emergency default result
    adjusted_thickness = [12; 30; 20];
    boundary_conditions = struct();
    boundary_conditions.method = 'winkler_springs';
    boundary_conditions.spring_coefficient = 50e6;
    boundary_conditions.soil_modulus = 50;
    boundary_conditions.theory_basis = 'Default (calculation failed)';
    boundary_conditions.version = 'v2.0_failed';
end
end

function CF = calculateComprehensiveCorrectionFixed(Es, thickness, load_params)
% Calculate comprehensive correction factor CF
% CF = Cbase × Cload × Cthickness
% Enhanced version with expanded range and improved physics

fprintf('Calculating comprehensive correction factor CF:\n');

% === 1. Soil condition correction factor Cbase ===
if Es < 30
    Cbase = 0.8; % Soft soil
    soil_type = 'Soft soil';
elseif Es > 80
    Cbase = 1.2; % Hard soil
    soil_type = 'Hard soil';
else
    Cbase = 1.0; % Medium soil
    soil_type = 'Medium soil';
end
fprintf('  Soil condition correction Cbase = %.1f (%s, Es=%dMPa)\n', Cbase, soil_type, Es);

% === 2. Load level correction factor Cload ===
if isfield(load_params, 'load_pressure')
    P = load_params.load_pressure;
    P_standard = 0.7; % Standard axle load pressure (MPa)
    
    % Continuous correction based on pressure ratio
    pressure_ratio = P / P_standard;
    if pressure_ratio > 1.15
        Cload = 1.0 + 0.15 * (pressure_ratio - 1.0); % Heavy load, increase stiffness
        Cload = min(Cload, 1.3); % Upper limit 30%
        load_type = 'Heavy load';
    elseif pressure_ratio < 0.85
        Cload = 1.0 - 0.15 * (1.0 - pressure_ratio); % Light load, reduce stiffness
        Cload = max(Cload, 0.7); % Lower limit -30%
        load_type = 'Light load';
    else
        Cload = 1.0;
        load_type = 'Standard load';
    end
    fprintf('  Load level correction Cload = %.3f (%s, P=%.2fMPa)\n', ...
        Cload, load_type, P);
else
    Cload = 1.0;
    fprintf('  Load level correction Cload = 1.0 (default)\n');
end

% === 3. Structure thickness correction factor Cthickness ===
thickness_data = thickness(:);
num_layers = length(thickness_data);
pavement_layers = min(3, num_layers); % First 3 layers are pavement structure
pavement_thickness = thickness_data(1:pavement_layers);

% Intelligent unit detection (pavement layers only)
if all(pavement_thickness > 1) && all(pavement_thickness < 100)
    % Likely cm unit: typical range [4-5, 15-25, 10-20]
    total_thickness_cm = sum(pavement_thickness); 
    fprintf('  Detected pavement thickness unit: cm, total %.0f cm\n', total_thickness_cm);
elseif all(pavement_thickness < 1)
    % Likely m unit
    total_thickness_cm = sum(pavement_thickness) * 100; % Convert to cm
    fprintf('  Detected pavement thickness unit: m, total %.0f cm\n', total_thickness_cm);
else
    % Mixed or ambiguous, use heuristic
    if mean(pavement_thickness) > 1
        total_thickness_cm = sum(pavement_thickness);
        fprintf('  Warning: Ambiguous thickness unit, assuming cm: %.0f cm\n', total_thickness_cm);
    else
        total_thickness_cm = sum(pavement_thickness) * 100;
        fprintf('  Warning: Ambiguous thickness unit, assuming m: %.0f cm\n', total_thickness_cm);
    end
end

% Thickness correction calculation (expanded range)
reference_thickness = 60; % Reference thickness (cm)
if total_thickness_cm > reference_thickness
    % Thick structure: enhancement correction
    excess_cm = total_thickness_cm - reference_thickness;
    Cthickness = 1.0 + 0.08 * (excess_cm / 10);  % 8% increase per 10cm
    Cthickness = min(Cthickness, 1.25); % Upper limit 25%
    thickness_type = 'Thick structure';
elseif total_thickness_cm < reference_thickness
    % Thin structure: reduction correction
    deficit_cm = reference_thickness - total_thickness_cm;
    Cthickness = 1.0 - 0.08 * (deficit_cm / 10);  % 8% decrease per 10cm
    Cthickness = max(Cthickness, 0.75); % Lower limit -25%
    thickness_type = 'Thin structure';
else
    Cthickness = 1.0;
    thickness_type = 'Standard structure';
end

fprintf('  Structure thickness correction Cthickness = %.3f (%s, %.0fcm)\n', ...
    Cthickness, thickness_type, total_thickness_cm);

% === 4. Calculate comprehensive correction factor ===
CF = Cbase * Cload * Cthickness;

% Limit to reasonable range (expanded)
CF_original = CF;
CF = max(0.4, min(CF, 2.5)); % Expanded from [0.5,2.0] to [0.4,2.5]

if CF ~= CF_original
    fprintf('  Warning: CF clipped from %.3f to %.3f\n', CF_original, CF);
end

fprintf('  Comprehensive correction factor CF = %.2f × %.3f × %.3f = %.3f\n', ...
    Cbase, Cload, Cthickness, CF);
end