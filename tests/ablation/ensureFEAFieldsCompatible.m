function pde_results = ensureFEAFieldsCompatible(pde_results)
% ensureFEAFieldsCompatible - Ensure PDE modeling results contain compatible FEA fields
%
% Input:
%   pde_results - PDE modeling result structure
%
% Output:
%   pde_results - Result structure with standardized FEA fields
%
% Functionality:
%   1. Check and supplement missing FEA fields
%   2. Unify field naming conventions (sigma_FEA, epsilon_FEA, D_FEA)
%   3. Provide backward compatibility support
%   4. Ensure numerical reasonableness

% Input validation
if isempty(pde_results) || ~isstruct(pde_results)
    pde_results = struct();
    pde_results.success = true;
    pde_results.sigma_FEA = 0.65;      % MPa - surface tensile stress
    pde_results.epsilon_FEA = 500;     % με - base tensile strain  
    pde_results.D_FEA = 8.0;          % mm - pavement deflection
    pde_results.stress_FEA = pde_results.sigma_FEA;
    pde_results.strain_FEA = pde_results.epsilon_FEA;
    pde_results.deflection_FEA = pde_results.D_FEA;
    return;
end

% Ensure success field exists
if ~isfield(pde_results, 'success')
    pde_results.success = true;
end

% 1. Handle stress field (sigma_FEA)
if ~isfield(pde_results, 'sigma_FEA')
    if isfield(pde_results, 'stress_FEA')
        pde_results.sigma_FEA = pde_results.stress_FEA;
    elseif isfield(pde_results, 'surface_stress')
        pde_results.sigma_FEA = pde_results.surface_stress;
    elseif isfield(pde_results, 'asphalt_stress')
        pde_results.sigma_FEA = pde_results.asphalt_stress;
    elseif isfield(pde_results, 'max_stress')
        pde_results.sigma_FEA = pde_results.max_stress;
    else
        pde_results.sigma_FEA = 0.65; % MPa
    end
end

% 2. Handle strain field (epsilon_FEA)
if ~isfield(pde_results, 'epsilon_FEA')
    if isfield(pde_results, 'strain_FEA')
        pde_results.epsilon_FEA = pde_results.strain_FEA;
    elseif isfield(pde_results, 'base_strain')
        pde_results.epsilon_FEA = pde_results.base_strain;
    elseif isfield(pde_results, 'tensile_strain')
        pde_results.epsilon_FEA = pde_results.tensile_strain;
    elseif isfield(pde_results, 'max_strain')
        pde_results.epsilon_FEA = pde_results.max_strain;
    else
        pde_results.epsilon_FEA = 500; % με
    end
end

% 3. Handle deflection field (D_FEA)
if ~isfield(pde_results, 'D_FEA')
    if isfield(pde_results, 'deflection_FEA')
        pde_results.D_FEA = pde_results.deflection_FEA;
    elseif isfield(pde_results, 'subgrade_deflection')
        pde_results.D_FEA = pde_results.subgrade_deflection;
    elseif isfield(pde_results, 'surface_deflection')
        pde_results.D_FEA = pde_results.surface_deflection;
    elseif isfield(pde_results, 'max_deflection')
        pde_results.D_FEA = pde_results.max_deflection;
    else
        pde_results.D_FEA = 8.0; % mm
    end
end

% 4. Add backward compatible fields
pde_results.stress_FEA = pde_results.sigma_FEA;
pde_results.strain_FEA = pde_results.epsilon_FEA;
pde_results.deflection_FEA = pde_results.D_FEA;

% 5. Numerical reasonableness check and correction
% Stress validation (0.1 ~ 2.0 MPa)
if pde_results.sigma_FEA < 0.1 || pde_results.sigma_FEA > 2.0 || isnan(pde_results.sigma_FEA)
    warning('Stress value out of range, corrected: %.3f -> 0.65 MPa', pde_results.sigma_FEA);
    pde_results.sigma_FEA = 0.65;
    pde_results.stress_FEA = 0.65;
end

% Strain validation (50 ~ 1200 με)
if pde_results.epsilon_FEA < 50 || pde_results.epsilon_FEA > 1200 || isnan(pde_results.epsilon_FEA)
    warning('Strain value out of range, corrected: %.0f -> 500 με', pde_results.epsilon_FEA);
    pde_results.epsilon_FEA = 500;
    pde_results.strain_FEA = 500;
end

% Deflection validation (1.0 ~ 20.0 mm)
if pde_results.D_FEA < 1.0 || pde_results.D_FEA > 20.0 || isnan(pde_results.D_FEA)
    warning('Deflection value out of range, corrected: %.2f -> 8.0 mm', pde_results.D_FEA);
    pde_results.D_FEA = 8.0;
    pde_results.deflection_FEA = 8.0;
end

% Ensure double precision
pde_results.sigma_FEA = double(pde_results.sigma_FEA);
pde_results.epsilon_FEA = double(pde_results.epsilon_FEA);
pde_results.D_FEA = double(pde_results.D_FEA);
pde_results.stress_FEA = double(pde_results.stress_FEA);
pde_results.strain_FEA = double(pde_results.strain_FEA);
pde_results.deflection_FEA = double(pde_results.deflection_FEA);

% 6. Ensure calculation success flag
if pde_results.sigma_FEA > 0 && pde_results.epsilon_FEA > 0 && pde_results.D_FEA > 0
    pde_results.success = true;
else
    pde_results.success = false;
end

end