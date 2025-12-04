function design_criteria = getJTG50DesignCriteria(user_input, parsed_params)
% JTG D50-2017 three-dimensional allowable values design criteria
% Comprehensive support for σ_std, ε_std, D_std calculation
%
% Inputs:  user_input - Natural language input from user
%          parsed_params - Parsed design parameters
% Outputs: design_criteria - Complete design criteria with 3D allowable values

fprintf(' Determining JTG D50-2017 three-dimensional design criteria...\n');

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
    
    % Design traffic volume parameters
    design_criteria.Ne = calculateDesignTrafficVolume(design_criteria.traffic_level, design_criteria.road_class);
    design_criteria.reliability_factor = getReliabilityFactor(design_criteria.road_class);
    
    fprintf('  Design traffic volume Ne: %.1e\n', design_criteria.Ne);
    fprintf('  Reliability factor: %.2f\n', design_criteria.reliability_factor);
    
    % Determine three-dimensional control indices
    [primary_indices, critical_index] = determine3DControlIndices(design_criteria.pavement_type, design_criteria.traffic_level);
    
    design_criteria.control_indices = struct();
    design_criteria.control_indices.primary = primary_indices;
    design_criteria.control_indices.critical = critical_index;
    
    fprintf('  Primary indices: %s\n', strjoin(primary_indices, ', '));
    fprintf('  Critical index: %s\n', critical_index);
    
    % Calculate three-dimensional allowable values: σ_std, ε_std, D_std
    design_criteria.allowable_values = calculate3DAllowableValues(design_criteria);
    
    % Fix field names if needed
    if isfield(design_criteria.allowable_values, 'base_tensile_stress')
        design_criteria.allowable_values.base_tensile_strain = design_criteria.allowable_values.base_tensile_stress;
        design_criteria.allowable_values = rmfield(design_criteria.allowable_values, 'base_tensile_stress');
        fprintf('  [Fixed] Field name: base_tensile_stress → base_tensile_strain\n');
    end
    
    if isfield(design_criteria.allowable_values, 'subgrade_strain')
        design_criteria.allowable_values.subgrade_deflection = design_criteria.allowable_values.subgrade_strain;
        design_criteria.allowable_values = rmfield(design_criteria.allowable_values, 'subgrade_strain');
        fprintf('  [Fixed] Field name: subgrade_strain → subgrade_deflection\n');
    end
    
    % Ensure required fields exist
    if ~isfield(design_criteria.allowable_values, 'surface_tensile_stress')
        design_criteria.allowable_values.surface_tensile_stress = 0.6;
    end
    if ~isfield(design_criteria.allowable_values, 'base_tensile_strain')
        design_criteria.allowable_values.base_tensile_strain = 600;
    end
    if ~isfield(design_criteria.allowable_values, 'subgrade_deflection')
        design_criteria.allowable_values.subgrade_deflection = 8.0;
    end
    
    fprintf('  ✓ JTG three-dimensional allowable values calculation complete:\n');
    fprintf('    σ_std (surface_tensile_stress): %.3f MPa\n', design_criteria.allowable_values.surface_tensile_stress);
    fprintf('    ε_std (base_tensile_strain): %.0f με\n', design_criteria.allowable_values.base_tensile_strain);
    fprintf('    D_std (subgrade_deflection): %.2f mm\n', design_criteria.allowable_values.subgrade_deflection);
    
    % Environmental and material parameters
    design_criteria.environmental_conditions = getEnvironmentalConditions(parsed_params);
    design_criteria.material_properties = getMaterialProperties(parsed_params);
    
    % Success flag
    design_criteria.success = true;
    design_criteria.message = 'JTG D50-2017 three-dimensional design criteria determination successful';
    design_criteria.standard = 'JTG D50-2017';
    design_criteria.version = 'Three_Dimensional_Allowable_Values_v11_Fixed';
    design_criteria.creation_time = datestr(now);
    
    fprintf('✓ JTG D50-2017 three-dimensional design criteria determination complete\n');
    
catch ME
    fprintf('✗ Three-dimensional design criteria determination failed: %s\n', ME.message);
    
    % Return default three-dimensional design criteria
    design_criteria = getDefault3DDesignCriteria_JTG50(parsed_params);
    design_criteria.success = false;
    design_criteria.message = sprintf('Using default three-dimensional criteria, reason: %s', ME.message);
end
end

%% Three-dimensional allowable values calculation function

function allowable_values = calculate3DAllowableValues(design_criteria)
% Calculate JTG D50-2017 three-dimensional allowable values: σ_std, ε_std, D_std

fprintf('Calculating JTG three-dimensional allowable values...\n');
allowable_values = struct();

% Extract basic parameters
pavement_type = design_criteria.pavement_type;
traffic_level = design_criteria.traffic_level;
Ne = design_criteria.Ne;
road_class = design_criteria.road_class;

fprintf('  Input parameters: Pavement type=%s, Traffic level=%s, Ne=%.1e\n', pavement_type, traffic_level, Ne);

switch pavement_type
    case 'semi_rigid'
        % Semi-rigid base pavement three-dimensional allowable values
        fprintf('  Applying semi-rigid base pavement formulas...\n');
        
        allowable_values.surface_tensile_stress = calculateSurfaceFatigueStress(Ne, 'semi_rigid');
        allowable_values.base_tensile_strain = calculateBaseTensileStrain(Ne, 'semi_rigid');
        allowable_values.subgrade_deflection = calculateSubgradeDeflection(Ne, road_class);
        
    case 'full_asphalt'
        % Full-depth asphalt pavement three-dimensional allowable values
        fprintf('  Applying full-depth asphalt pavement formulas...\n');
        
        allowable_values.surface_tensile_stress = calculateSurfaceFatigueStress(Ne, 'full_asphalt');
        allowable_values.base_tensile_strain = calculateAsphaltTensileStrain(Ne);
        allowable_values.subgrade_deflection = calculateSubgradeDeflection(Ne, road_class);
        
    case 'flexible'
        % Flexible base pavement three-dimensional allowable values
        fprintf('  Applying flexible base pavement formulas...\n');
        
        allowable_values.surface_tensile_stress = calculateSurfaceFatigueStress(Ne, 'flexible');
        allowable_values.base_tensile_strain = calculateFlexibleBaseTensileStrain(Ne);
        allowable_values.subgrade_deflection = calculateSubgradeDeflection(Ne, road_class) * 1.2;
        
    otherwise
        % Default to semi-rigid processing
        fprintf('  Warning: Unrecognized pavement type, processing as semi-rigid\n');
        allowable_values.surface_tensile_stress = 0.6;
        allowable_values.base_tensile_strain = 600;
        allowable_values.subgrade_deflection = 8.0;
end

% Ensure standard field names
final_allowable = struct();
final_allowable.surface_tensile_stress = allowable_values.surface_tensile_stress;
final_allowable.base_tensile_strain = allowable_values.base_tensile_strain;
final_allowable.subgrade_deflection = allowable_values.subgrade_deflection;

% Check and fix possible incorrect field names
if isfield(allowable_values, 'base_tensile_stress')
    final_allowable.base_tensile_strain = allowable_values.base_tensile_stress;
    fprintf('  [Fixed] Field name: base_tensile_stress → base_tensile_strain\n');
end

if isfield(allowable_values, 'subgrade_strain')
    final_allowable.subgrade_deflection = allowable_values.subgrade_strain;
    fprintf('  [Fixed] Field name: subgrade_strain → subgrade_deflection\n');
end

allowable_values = final_allowable;

fprintf('  ✓ Field name correction complete:\n');
fprintf('    surface_tensile_stress: %.4f MPa\n', allowable_values.surface_tensile_stress);
fprintf('    base_tensile_strain: %.0f με\n', allowable_values.base_tensile_strain);
fprintf('    subgrade_deflection: %.2f mm\n', allowable_values.subgrade_deflection);

% Apply reliability factor adjustment
reliability_factor = design_criteria.reliability_factor;
fprintf('  Applying reliability factor adjustment: γ = %.3f\n', reliability_factor);

allowable_values.surface_tensile_stress = allowable_values.surface_tensile_stress / reliability_factor;
allowable_values.base_tensile_strain = allowable_values.base_tensile_strain / reliability_factor;

% Apply engineering constraints
allowable_values = apply3DEngineeringConstraints(allowable_values, traffic_level);

fprintf('  ✓ Final three-dimensional allowable values:\n');
fprintf('    σ_std (surface_tensile_stress) = %.4f MPa\n', allowable_values.surface_tensile_stress);
fprintf('    ε_std (base_tensile_strain) = %.0f με\n', allowable_values.base_tensile_strain);
fprintf('    D_std (subgrade_deflection) = %.2f mm\n', allowable_values.subgrade_deflection);
end

%% JTG D50-2017 allowable value calculation functions

function [primary_indices, critical_index] = determine3DControlIndices(pavement_type, traffic_level)
% Determine three-dimensional control indices

primary_indices = {'surface_tensile_stress', 'base_tensile_strain', 'subgrade_deflection'};

switch pavement_type
    case 'semi_rigid'
        if strcmp(traffic_level, 'heavy') || strcmp(traffic_level, 'extra_heavy')
            critical_index = 'surface_tensile_stress';
        end
        
    case 'full_asphalt'
        critical_index = 'surface_tensile_stress';
        
    case 'flexible'
        if strcmp(traffic_level, 'light')
            critical_index = 'subgrade_deflection';
        else
            critical_index = 'base_tensile_strain';
        end
        
    otherwise
        critical_index = 'surface_tensile_stress';
end
end

function sigma_std = calculateSurfaceFatigueStress(Ne, pavement_type)
% Calculate surface fatigue stress allowable value (JTG D50-2017)

if Ne <= 1e5
    k_Ne = 2.5;
elseif Ne <= 1e6
    k_Ne = 2.0;
elseif Ne <= 5e6
    k_Ne = 1.5;
else
    k_Ne = 1.2;
end

switch pavement_type
    case 'semi_rigid'
        base_stress = 0.75;
        type_factor = 1.0;
    case 'full_asphalt'
        base_stress = 0.80;
        type_factor = 1.05;
    case 'flexible'
        base_stress = 0.65;
        type_factor = 0.90;
    otherwise
        base_stress = 0.75;
        type_factor = 1.0;
end

sigma_std = base_stress * type_factor / k_Ne;
sigma_std = max(min(sigma_std, 1.2), 0.3);
end

function epsilon_std = calculateBaseTensileStrain(Ne, pavement_type)
% Calculate base tensile strain allowable value

if Ne <= 1e5
    base_strain = 700;
elseif Ne <= 1e6
    base_strain = 600;
elseif Ne <= 5e6
    base_strain = 500;
else
    base_strain = 400;
end

switch pavement_type
    case 'semi_rigid'
        type_factor = 1.0;
    case 'full_asphalt'
        type_factor = 1.1;
    case 'flexible'
        type_factor = 0.9;
    otherwise
        type_factor = 1.0;
end

epsilon_std = base_strain * type_factor;
epsilon_std = max(min(epsilon_std, 2000), 250);
end

function epsilon_std = calculateAsphaltTensileStrain(Ne)
% Calculate asphalt tensile strain allowable value

if Ne <= 1e5
    epsilon_std = 800;
elseif Ne <= 1e6
    epsilon_std = 700;
elseif Ne <= 5e6
    epsilon_std = 600;
else
    epsilon_std = 500;
end

epsilon_std = max(min(epsilon_std, 2500), 300);
end

function epsilon_std = calculateFlexibleBaseTensileStrain(Ne)
% Calculate flexible base tensile strain allowable value

if Ne <= 1e5
    epsilon_std = 900;
elseif Ne <= 1e6
    epsilon_std = 800;
elseif Ne <= 5e6
    epsilon_std = 700;
else
    epsilon_std = 600;
end

epsilon_std = max(min(epsilon_std, 3000), 400);
end

function D_std = calculateSubgradeDeflection(Ne, road_class)
% Calculate subgrade deflection allowable value

if Ne <= 1e5
    base_deflection = 12.0;
elseif Ne <= 1e6
    base_deflection = 10.0;
elseif Ne <= 5e6
    base_deflection = 8.0;
else
    base_deflection = 6.5;
end

if contains(road_class, {'highway', 'expressway'})
    class_factor = 0.8;
elseif contains(road_class, {'urban'})
    class_factor = 0.9;
else
    class_factor = 1.0;
end

D_std = base_deflection * class_factor;
D_std = max(min(D_std, 30.0), 4.0);
end

%% Engineering constraints

function allowable_values = apply3DEngineeringConstraints(allowable_values, traffic_level)
% Apply engineering constraints to three-dimensional indicators

% Surface tensile stress constraints
if allowable_values.surface_tensile_stress < 0.25
    fprintf('    Warning: Surface tensile stress too low (%.3f < 0.25), adjusting to 0.25 MPa\n', ...
        allowable_values.surface_tensile_stress);
    allowable_values.surface_tensile_stress = 0.25;
elseif allowable_values.surface_tensile_stress > 1.5
    fprintf('    Warning: Surface tensile stress too high (%.3f > 1.5), adjusting to 1.5 MPa\n', ...
        allowable_values.surface_tensile_stress);
    allowable_values.surface_tensile_stress = 1.5;
end

% Base tensile strain constraints
if allowable_values.base_tensile_strain < 200
    fprintf('    Warning: Base tensile strain too low (%.0f < 200), adjusting to 200 με\n', ...
        allowable_values.base_tensile_strain);
    allowable_values.base_tensile_strain = 200;
elseif allowable_values.base_tensile_strain > 2500
    fprintf('    Warning: Base tensile strain too high (%.0f > 2500), adjusting to 2500 με\n', ...
        allowable_values.base_tensile_strain);
    allowable_values.base_tensile_strain = 2500;
end

% Subgrade deflection constraints
if allowable_values.subgrade_deflection < 4.0
    fprintf('    Warning: Subgrade deflection too low (%.2f < 4.0), adjusting to 4.0 mm\n', ...
        allowable_values.subgrade_deflection);
    allowable_values.subgrade_deflection = 4.0;
elseif allowable_values.subgrade_deflection > 30.0
    fprintf('    Warning: Subgrade deflection too high (%.2f > 30.0), adjusting to 30.0 mm\n', ...
        allowable_values.subgrade_deflection);
    allowable_values.subgrade_deflection = 30.0;
end

% Three-dimensional indicator coordination check
performCoordinationCheck(allowable_values, traffic_level);
end

function performCoordinationCheck(allowable_values, traffic_level)
% Three-dimensional indicator coordination check

fprintf('    Performing three-dimensional indicator coordination check...\n');

sigma_std = allowable_values.surface_tensile_stress;
epsilon_std = allowable_values.base_tensile_strain;
D_std = allowable_values.subgrade_deflection;

% Calculate indicator severity ratios
stress_severity = 1.0 / sigma_std;
strain_severity = 1000.0 / epsilon_std;
deflection_severity = 10.0 / D_std;

% Check for abnormally strict or loose indicators
max_severity = max([stress_severity, strain_severity, deflection_severity]);
min_severity = min([stress_severity, strain_severity, deflection_severity]);
severity_ratio = max_severity / min_severity;

if severity_ratio > 3.0
    fprintf('    Warning: Large difference in indicator severity (ratio: %.1f)\n', severity_ratio);
    fprintf('      Stress severity: %.2f, Strain severity: %.2f, Deflection severity: %.2f\n', ...
        stress_severity, strain_severity, deflection_severity);
    
    if strcmp(traffic_level, 'heavy') || strcmp(traffic_level, 'extra_heavy')
        fprintf('      Note: High standards are appropriate for heavy traffic\n');
    else
        fprintf('      Note: Consider adjusting indicator balance\n');
    end
else
    fprintf('    ✓ Good three-dimensional indicator coordination (severity ratio: %.1f)\n', severity_ratio);
end
end

%% Support functions

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
    pavement_type = 'semi_rigid';
end
end

function traffic_level = inferTrafficLevel(user_input)
% Infer traffic level from user input

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

function Ne = calculateDesignTrafficVolume(traffic_level, road_class)
% Calculate design traffic volume

base_values = containers.Map(...
    {'light', 'medium', 'heavy', 'extra_heavy'}, ...
    {1e5, 1e6, 5e6, 2e7});

if isKey(base_values, traffic_level)
    Ne = base_values(traffic_level);
else
    Ne = 1e6;
end

% Adjust based on road class
if contains(road_class, 'highway')
    Ne = Ne * 1.5;
elseif contains(road_class, 'urban')
    Ne = Ne * 1.2;
end
end

function reliability_factor = getReliabilityFactor(road_class)
% Get reliability factor

if contains(road_class, 'highway')
    reliability_factor = 1.15;
elseif contains(road_class, 'urban')
    reliability_factor = 1.10;
else
    reliability_factor = 1.05;
end
end

function env_conditions = getEnvironmentalConditions(parsed_params)
% Get environmental conditions

env_conditions = struct();

if isfield(parsed_params, 'climate_zone')
    env_conditions.climate_zone = parsed_params.climate_zone;
else
    env_conditions.climate_zone = 'temperate';
end

if isfield(parsed_params, 'drainage_condition')
    env_conditions.drainage_condition = parsed_params.drainage_condition;
else
    env_conditions.drainage_condition = 'good';
end

env_conditions.freeze_thaw_cycles = inferFreezeThawCycles(env_conditions.climate_zone);
env_conditions.moisture_coefficient = getMoistureCoefficient(env_conditions.drainage_condition);
end

function freeze_cycles = inferFreezeThawCycles(climate_zone)
% Infer freeze-thaw cycles

switch climate_zone
    case {'cold', 'severe'}
        freeze_cycles = 100;
    case {'temperate', 'warm temperate'}
        freeze_cycles = 20;
    case {'subtropical', 'tropical'}
        freeze_cycles = 0;
    otherwise
        freeze_cycles = 20;
end
end

function moisture_coeff = getMoistureCoefficient(drainage_condition)
% Get moisture coefficient

switch drainage_condition
    case 'excellent'
        moisture_coeff = 1.0;
    case 'good'
        moisture_coeff = 0.9;
    case 'fair'
        moisture_coeff = 0.8;
    case 'poor'
        moisture_coeff = 0.7;
    otherwise
        moisture_coeff = 0.9;
end
end

function material_props = getMaterialProperties(parsed_params)
% Get material properties

material_props = struct();

if isfield(parsed_params, 'material')
    material_props.layer_materials = parsed_params.material;
else
    material_props.layer_materials = {'Asphalt Concrete'; 'Cement Stabilized Base'; 'Graded Aggregate'; 'Improved Soil'};
end

material_props.asphalt_type = 'SBS Modified Asphalt';
material_props.cement_content = 4.5;
material_props.aggregate_type = 'Limestone';
end

function default_criteria = getDefault3DDesignCriteria_JTG50(parsed_params)
% Get default JTG50 three-dimensional design criteria

default_criteria = struct();
default_criteria.pavement_type = 'semi_rigid';
default_criteria.road_class = 'highway';
default_criteria.traffic_level = 'heavy';
default_criteria.Ne = 5e6;
default_criteria.reliability_factor = 1.15;

% Three-dimensional control indices
default_criteria.control_indices = struct();
default_criteria.control_indices.primary = {'surface_tensile_stress', 'base_tensile_strain', 'subgrade_deflection'};
default_criteria.control_indices.critical = 'surface_tensile_stress';

% Three-dimensional allowable values
default_criteria.allowable_values = struct();
default_criteria.allowable_values.surface_tensile_stress = 0.6;
default_criteria.allowable_values.base_tensile_strain = 600;
default_criteria.allowable_values.subgrade_deflection = 8.0;

default_criteria.success = true;
default_criteria.message = 'Default JTG D50-2017 three-dimensional design criteria';
default_criteria.standard = 'JTG D50-2017';
default_criteria.version = 'Three_Dimensional_Default_v11_Fixed';
end