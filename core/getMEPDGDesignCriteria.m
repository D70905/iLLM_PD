function design_criteria = getMEPDGDesignCriteria(user_input, parsed_params)
% ME-PDG (NCHRP 1-37A) Mechanistic-Empirical pavement design allowable values
% Based on ME-PDG performance prediction model for allowable values calculation
%
% Core features:
%   1. Fatigue transfer functions
%   2. Local calibration coefficients (adjustable)
%   3. Mechanical response model (elastic layered system)
%   4. Climate and material effects consideration
%
% Inputs:  user_input - Natural language input from user
%          parsed_params - Parsed design parameters
% Outputs: design_criteria - Complete design criteria with 3D allowable values
%
% References: 
%   NCHRP Report 1-37A (2004)
%   ME-PDG Design Guide (2008, 2015)
%   AASHTO Pavement ME Design (2020)

fprintf(' ME-PDG mechanistic-empirical design standard...\n');

try
    % Initialize design criteria structure
    design_criteria = struct();
    
    % Basic pavement information
    if isfield(parsed_params, 'pavement_type')
        design_criteria.pavement_type = parsed_params.pavement_type;
    else
        design_criteria.pavement_type = analyzePavementType(parsed_params);
    end
    
    if isfield(parsed_params, 'road_type')
        design_criteria.road_class = parsed_params.road_type;
    else
        design_criteria.road_class = 'highway';
    end
    
    if isfield(parsed_params, 'traffic_level')
        design_criteria.traffic_level = parsed_params.traffic_level;
    else
        design_criteria.traffic_level = inferTrafficLevel(user_input);
    end
    
    fprintf('  Pavement type: %s\n', design_criteria.pavement_type);
    fprintf('  Road class: %s\n', design_criteria.road_class);
    fprintf('  Traffic level: %s\n', design_criteria.traffic_level);
    
    % ME-PDG core design parameters
    design_criteria.ESAL = calculateESAL(design_criteria.traffic_level, design_criteria.road_class);
    design_criteria.reliability = getReliability(design_criteria.road_class);
    design_criteria.design_life = getDesignLife(design_criteria.road_class);
    
    fprintf('  Design ESAL: %.2e\n', design_criteria.ESAL);
    fprintf('  Reliability R: %.1f%%\n', design_criteria.reliability * 100);
    fprintf('  Design life: %d years\n', design_criteria.design_life);
    
    % Material properties extraction
    material_properties = extractMaterialProperties(parsed_params);
    design_criteria.material_properties = material_properties;
    
    % Climate and environmental conditions
    climate_data = getClimateData(parsed_params);
    design_criteria.climate_data = climate_data;
    
    % Determine control indices
    [primary_indices, critical_index] = determineControlIndices(design_criteria.pavement_type);
    
    design_criteria.control_indices = struct();
    design_criteria.control_indices.primary = primary_indices;
    design_criteria.control_indices.critical = critical_index;
    
    fprintf('  Primary indices: %s\n', strjoin(primary_indices, ', '));
    fprintf('  Critical index: %s\n', critical_index);
    
    % Calculate ME-PDG allowable values based on transfer functions
    design_criteria.allowable_values = calculateMEPDGAllowableValues(...
        design_criteria.ESAL, ...
        material_properties, ...
        design_criteria.reliability, ...
        climate_data, ...
        design_criteria.pavement_type);
    
    fprintf('  ✓ ME-PDG allowable values calculation complete:\n');
    fprintf('    σ_std (surface_tensile_stress): %.3f MPa\n', ...
        design_criteria.allowable_values.surface_tensile_stress);
    fprintf('    ε_std (base_tensile_strain): %.0f με\n', ...
        design_criteria.allowable_values.base_tensile_strain);
    fprintf('    D_std (subgrade_deflection): %.2f mm\n', ...
        design_criteria.allowable_values.subgrade_deflection);
    
    % ME-PDG specific parameters
    design_criteria.performance_criteria = getPerformanceCriteria(design_criteria.road_class);
    design_criteria.distress_models = getDistressModels();
    design_criteria.calibration_factors = getCalibrationFactors(parsed_params);
    
    fprintf('  Performance criteria: fatigue cracking<%.0f%%, rutting<%.1fmm\n', ...
        design_criteria.performance_criteria.fatigue_cracking_limit * 100, ...
        design_criteria.performance_criteria.rutting_limit);
    
    % Success flag
    design_criteria.success = true;
    design_criteria.message = 'ME-PDG design criteria determination successful';
    design_criteria.standard = 'ME-PDG';
    design_criteria.version = 'MEPDG_NCHRP_1_37A_v3.0';
    design_criteria.creation_time = datestr(now);
    
    fprintf('✓ ME-PDG design criteria determination complete\n');
    
catch ME
    fprintf('✗ ME-PDG design criteria determination failed: %s\n', ME.message);
    
    % Return default design criteria
    design_criteria = getDefaultMEPDGDesignCriteria(parsed_params);
    design_criteria.success = false;
    design_criteria.message = sprintf('Using default ME-PDG criteria, reason: %s', ME.message);
end

end

%% ME-PDG allowable values calculation

function allowable_values = calculateMEPDGAllowableValues(ESAL, material_props, ...
    reliability, climate_data, pavement_type)
% Calculate allowable values based on ME-PDG transfer functions

fprintf(' Calculating allowable values based on ME-PDG transfer functions...\n');
allowable_values = struct();

% Surface fatigue cracking allowable values
fprintf('  Calculating surface tensile stress allowable values...\n');

% Get calibration coefficients
calibration = getCalibrationFactors(struct());

% Extract material parameters
E_asphalt = material_props.E_asphalt;  % MPa
h_asphalt = material_props.thickness_asphalt;  % cm

fprintf('    [Debug] Material parameters: E=%.0f MPa, h=%.1f cm\n', E_asphalt, h_asphalt);

% ME-PDG calibration coefficients
beta_f1 = calibration.beta_f1;
beta_f2 = calibration.beta_f2;
beta_f3 = calibration.beta_f3;

fprintf('    [Debug] Calibration coefficients: βf1=%.6f, βf2=%.4f, βf3=%.4f\n', ...
    beta_f1, beta_f2, beta_f3);

% Thickness effect coefficient
k1_thickness = (h_asphalt / 10)^(-0.854);
k1_thickness = max(k1_thickness, 0.1);

% Climate correction coefficient
C_climate = calculateClimateCorrection(climate_data, E_asphalt);

% Target fatigue life
Nf_target = ESAL * 10;

fprintf('    [Debug] Intermediate values: k1=%.4f, C_climate=%.4f, Nf_target=%.2e\n', ...
    k1_thickness, C_climate, Nf_target);

% Calculate allowable tensile strain
denominator = beta_f1 * k1_thickness * C_climate * E_asphalt^(-beta_f3);

fprintf('    [Debug] Denominator value: %.6e\n', denominator);

if denominator <= 0 || isnan(denominator) || isinf(denominator)
    fprintf('    Warning: Abnormal denominator (%.6e), using default value\n', denominator);
    epsilon_t_allowable = 200e-6;
else
    exponent_term = Nf_target / denominator;
    fprintf('    [Debug] Exponent term: %.6e\n', exponent_term);
    
    epsilon_t_allowable = exponent_term^(-1/beta_f2);
    fprintf('    [Debug] Preliminary ε_t: %.6e (%.0f με)\n', ...
        epsilon_t_allowable, epsilon_t_allowable*1e6);
    
    % Limit to reasonable range [50με, 500με]
    epsilon_t_allowable = max(min(epsilon_t_allowable, 500e-6), 50e-6);
end

fprintf('    [Debug] Final ε_t_allowable: %.2e (%.0f με)\n', ...
    epsilon_t_allowable, epsilon_t_allowable*1e6);

% Convert to stress σ = E * ε
sigma_allowable = E_asphalt * epsilon_t_allowable;  % MPa

fprintf('    [Debug] Preliminary stress: %.3f MPa\n', sigma_allowable);

% Reliability adjustment
ZR = norminv(reliability);
reliability_factor_stress = exp(-0.35 * ZR^2 * 0.45);
sigma_allowable = sigma_allowable * reliability_factor_stress;

fprintf('    [Debug] After reliability adjustment: %.3f MPa (factor=%.3f)\n', ...
    sigma_allowable, reliability_factor_stress);

% Ensure reasonable range [0.2, 1.5] MPa
if sigma_allowable < 0.2 || sigma_allowable > 1.5
    fprintf('    Warning: Stress outside reasonable range, correcting\n');
    sigma_allowable = max(min(sigma_allowable, 1.5), 0.2);
end

fprintf('    ✓ σ_allowable = %.3f MPa (ME-PDG transfer function)\n', sigma_allowable);

allowable_values.surface_tensile_stress = sigma_allowable;

% Base tensile strain allowable values
fprintf('  Calculating base tensile strain allowable values...\n');

E_base = material_props.E_base;
beta_b1 = calibration.beta_b1;
beta_b2 = calibration.beta_b2;

fprintf('    [Debug] Base E=%.0f MPa, βb1=%.6f, βb2=%.4f\n', ...
    E_base, beta_b1, beta_b2);

% Calculate allowable tensile strain
epsilon_base_allowable = (Nf_target / beta_b1)^(-1/beta_b2) * 1e6;  % με

fprintf('    [Debug] Preliminary strain: %.0f με\n', epsilon_base_allowable);

% Reliability adjustment
reliability_factor_strain = exp(-0.25 * ZR^2 * 0.35);
epsilon_base_allowable = epsilon_base_allowable * reliability_factor_strain;

% Pavement type adjustment
switch pavement_type
    case 'semi_rigid'
        type_factor = 0.8;
    case 'flexible'
        type_factor = 1.2;
    otherwise
        type_factor = 1.0;
end
epsilon_base_allowable = epsilon_base_allowable * type_factor;

fprintf('    [Debug] After type adjustment: %.0f με (factor=%.2f)\n', ...
    epsilon_base_allowable, type_factor);

% Ensure reasonable range [200, 2500] με
epsilon_base_allowable = max(min(epsilon_base_allowable, 2500), 200);

fprintf('    ✓ ε_allowable = %.0f με (ME-PDG transfer function)\n', epsilon_base_allowable);

allowable_values.base_tensile_strain = epsilon_base_allowable;

% Subgrade deflection allowable values
fprintf('  Calculating subgrade deflection allowable values...\n');

MR_subgrade = material_props.MR_subgrade;
P_std = 0.707;  % MPa, standard load

% Boussinesq elastic solution (simplified)
r = 15;  % cm, load radius
z_influence = 150;  % cm, influence depth

% Deflection calculation
D_base = (1.5 * P_std * r) / MR_subgrade;  % mm

fprintf('    [Debug] Base deflection: %.3f mm\n', D_base);

% Traffic adjustment
if ESAL < 1e5
    traffic_factor = 1.5;
elseif ESAL < 1e6
    traffic_factor = 1.2;
elseif ESAL < 5e6
    traffic_factor = 1.0;
else
    traffic_factor = 0.8;
end

% Structure coefficient (considering upper layer protection)
K_structure = calculateStructureCoefficient(material_props);

D_allowable = D_base * traffic_factor * K_structure;

fprintf('    [Debug] Traffic factor=%.2f, Structure factor=%.3f\n', ...
    traffic_factor, K_structure);

% Reliability adjustment
D_allowable = D_allowable * exp(-0.10 * ZR);

% Ensure reasonable range [3.0, 25.0] mm
D_allowable = max(min(D_allowable, 25.0), 3.0);

fprintf('    ✓ D_allowable = %.2f mm (ME-PDG method)\n', D_allowable);

allowable_values.subgrade_deflection = D_allowable;

fprintf('  ✓ All ME-PDG allowable values calculated\n');
end

%% Calibration factors

function calibration = getCalibrationFactors(parsed_params)
% Get ME-PDG calibration factors

calibration = struct();

% Bottom-up fatigue cracking calibration factors
calibration.beta_f1 = 0.00432;
calibration.beta_f2 = 3.9492;
calibration.beta_f3 = 1.281;

% Base fatigue calibration factors
calibration.beta_b1 = 0.004325;
calibration.beta_b2 = 3.291;

% Rutting calibration factors
calibration.beta_r1 = 1.35;
calibration.beta_r2 = 0.328;

% Thermal cracking calibration factors
calibration.beta_t1 = 2.0;
calibration.beta_t2 = 1.0;
end

function C_climate = calculateClimateCorrection(climate_data, E_asphalt)
% Calculate climate correction coefficient

T_mean = climate_data.mean_annual_temp;

% Temperature effect on asphalt modulus
if T_mean < 10
    C_temp = 1.2;
elseif T_mean < 20
    C_temp = 1.0;
else
    C_temp = 0.85;
end

% Aging effect
design_life = climate_data.design_life;
C_aging = 1 - 0.02 * design_life;
C_aging = max(C_aging, 0.7);

C_climate = C_temp * C_aging;
end

function K_structure = calculateStructureCoefficient(material_props)
% Structure coefficient (based on modulus ratio)

E_asphalt = material_props.E_asphalt;
E_base = material_props.E_base;
MR_subgrade = material_props.MR_subgrade;

% Modulus ratios
ratio_1 = E_asphalt / E_base;
ratio_2 = E_base / MR_subgrade;

% Odemark equivalent thickness method (simplified)
K_structure = 1 / (1 + 0.5 * log10(ratio_1) + 0.3 * log10(ratio_2));
end

%% Material properties extraction

function material_props = extractMaterialProperties(parsed_params)
% Extract material properties from parsed_params

material_props = struct();

% Surface layer modulus
if isfield(parsed_params, 'modulus') && length(parsed_params.modulus) >= 1
    material_props.E_asphalt = parsed_params.modulus(1);
else
    material_props.E_asphalt = 3500;  % MPa (typical HMA @20°C)
end

% Base layer modulus
if isfield(parsed_params, 'modulus') && length(parsed_params.modulus) >= 2
    material_props.E_base = parsed_params.modulus(2);
else
    material_props.E_base = 400;  % MPa (granular base)
end

% Subgrade resilient modulus
if isfield(parsed_params, 'modulus') && length(parsed_params.modulus) >= 3
    material_props.MR_subgrade = parsed_params.modulus(end);
else
    material_props.MR_subgrade = 50;  % MPa
end

% Layer thickness
if isfield(parsed_params, 'thickness') && length(parsed_params.thickness) >= 1
    material_props.thickness_asphalt = parsed_params.thickness(1);
else
    material_props.thickness_asphalt = 10;  % cm
end

fprintf('  Material parameters: E_AC=%.0f MPa, E_base=%.0f MPa, MR=%.0f MPa\n', ...
    material_props.E_asphalt, material_props.E_base, material_props.MR_subgrade);
end

function climate_data = getClimateData(parsed_params)
% Get climate data

climate_data = struct();

if isfield(parsed_params, 'mean_annual_temp')
    climate_data.mean_annual_temp = parsed_params.mean_annual_temp;
else
    climate_data.mean_annual_temp = 15;  % °C (temperate)
end

if isfield(parsed_params, 'design_life')
    climate_data.design_life = parsed_params.design_life;
else
    climate_data.design_life = 15;  % years
end

climate_data.freeze_index = 0;  % Freezing index (default: no freeze-thaw)
climate_data.precipitation = 800;  % mm/year
end

%% Performance criteria

function performance = getPerformanceCriteria(road_class)
% ME-PDG performance criteria

performance = struct();

switch road_class
    case {'highway', 'expressway'}
        % Highway (strict standards)
        performance.fatigue_cracking_limit = 0.10;  % 10% area
        performance.rutting_limit = 10;  % mm
        performance.IRI_limit = 2.7;  % m/km (International Roughness Index)
        
    case {'urban', 'arterial'}
        % Arterial
        performance.fatigue_cracking_limit = 0.15;  % 15%
        performance.rutting_limit = 12.5;  % mm
        performance.IRI_limit = 3.5;  % m/km
        
    otherwise
        % Other roads
        performance.fatigue_cracking_limit = 0.25;  % 25%
        performance.rutting_limit = 19;  % mm
        performance.IRI_limit = 4.5;  % m/km
end
end

function distress_models = getDistressModels()
% ME-PDG distress model types

distress_models = struct();
distress_models.fatigue_cracking = 'Bottom_Up_Transfer_Function';
distress_models.rutting = 'Permanent_Deformation_Model';
distress_models.top_down_cracking = 'Fracture_Mechanics';
distress_models.thermal_cracking = 'Paris_Law';
end

%% General auxiliary functions

function [primary_indices, critical_index] = determineControlIndices(pavement_type)
% Determine control indices

primary_indices = {'surface_tensile_stress', 'base_tensile_strain', 'subgrade_deflection'};

switch pavement_type
    case 'semi_rigid'
        critical_index = 'surface_tensile_stress';
    case 'full_asphalt'
        critical_index = 'surface_tensile_stress';
    case 'flexible'
        critical_index = 'subgrade_deflection';
    otherwise
        critical_index = 'surface_tensile_stress';
end
end

function ESAL = calculateESAL(traffic_level, road_class)
% Calculate equivalent single axle load

base_values = containers.Map(...
    {'light', 'medium', 'heavy', 'extra_heavy'}, ...
    {5e4, 5e5, 5e6, 5e7});

if isKey(base_values, traffic_level)
    ESAL = base_values(traffic_level);
else
    ESAL = 5e5;
end

% Road class adjustment
if contains(road_class, {'highway', 'expressway'})
    ESAL = ESAL * 2.0;
elseif contains(road_class, {'urban'})
    ESAL = ESAL * 1.5;
end
end

function reliability = getReliability(road_class)
% Get design reliability

switch road_class
    case {'highway', 'expressway'}
        reliability = 0.95;
    case {'urban', 'arterial'}
        reliability = 0.90;
    case {'collector'}
        reliability = 0.85;
    otherwise
        reliability = 0.90;
end
end

function design_life = getDesignLife(road_class)
% Get design life

switch road_class
    case {'highway', 'expressway'}
        design_life = 20;  % years
    case {'urban', 'arterial'}
        design_life = 15;
    otherwise
        design_life = 10;
end
end

function pavement_type = analyzePavementType(parsed_params)
% Analyze pavement type

if isfield(parsed_params, 'modulus') && length(parsed_params.modulus) >= 2
    base_modulus = parsed_params.modulus(2);
    if base_modulus >= 800
        pavement_type = 'full_asphalt';
    elseif base_modulus >= 300
        pavement_type = 'semi_rigid';
    else
        pavement_type = 'flexible';
    end
else
    pavement_type = 'flexible';
end
end

function traffic_level = inferTrafficLevel(user_input)
% Infer traffic level

if contains(user_input, {'heavy'})
    traffic_level = 'heavy';
elseif contains(user_input, {'light'})
    traffic_level = 'light';
elseif contains(user_input, {'extra heavy'})
    traffic_level = 'extra_heavy';
else
    traffic_level = 'medium';
end
end

function default_criteria = getDefaultMEPDGDesignCriteria(parsed_params)
% Get default ME-PDG design criteria

default_criteria = struct();
default_criteria.pavement_type = 'flexible';
default_criteria.road_class = 'highway';
default_criteria.traffic_level = 'medium';
default_criteria.ESAL = 5e5;
default_criteria.reliability = 0.90;
default_criteria.design_life = 15;

% Three-dimensional allowable values (default)
default_criteria.control_indices = struct();
default_criteria.control_indices.primary = {'surface_tensile_stress', 'base_tensile_strain', 'subgrade_deflection'};
default_criteria.control_indices.critical = 'surface_tensile_stress';

default_criteria.allowable_values = struct();
default_criteria.allowable_values.surface_tensile_stress = 0.55;
default_criteria.allowable_values.base_tensile_strain = 600;
default_criteria.allowable_values.subgrade_deflection = 8.5;

default_criteria.material_properties = struct();
default_criteria.material_properties.E_asphalt = 3500;
default_criteria.material_properties.E_base = 400;
default_criteria.material_properties.MR_subgrade = 50;

default_criteria.success = true;
default_criteria.message = 'Default ME-PDG design criteria';
default_criteria.standard = 'ME-PDG';
default_criteria.version = 'Default_v3.0';
end