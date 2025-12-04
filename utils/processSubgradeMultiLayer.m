function [adjusted_thickness, boundary_conditions] = processSubgradeMultiLayer(thickness, config, load_params)
% PROCESSSUBGRADEMULTILAYER - Multi-layer elastic subgrade modeling
% Multi-layer elastic subgrade model processing function based on engineering practice
%
% Physical interpretation of modulus ratios:
% - Soft soil [1.0, 0.8, 1.2, 1.5]:
%   1.0 = Surface reference state
%   0.8 = Shallow soft interlayer (common soft clay at 0.5-1.0m depth)
%   1.2 = Transition zone, consolidation begins
%   1.5 = Deep consolidation enhancement, increased confining pressure
%
% Inputs:
%   thickness    - Original layer thickness vector [surface; base; subbase; subgrade] (cm)
%   config       - Configuration parameters structure
%   load_params  - Load parameters structure containing soil_modulus field
%
% Outputs:
%   adjusted_thickness    - Adjusted thickness vector including subgrade sublayers (cm)
%   boundary_conditions   - Boundary conditions structure
%
% Theory:
%   Based on multi-layer elastic theory with depth-dependent subgrade properties
%   Sublayer modulus formula: E_i = E_s,surface × R_i

fprintf('=========================================\n');
fprintf('Multi-layer Elastic Subgrade Model Processing\n');
fprintf('=========================================\n');

try
    % === 1. Parameter extraction and validation ===
    [Es_surface, load_pressure, load_radius] = extractAndValidateParameters(load_params);
    
    fprintf('Input parameters:\n');
    fprintf('  Surface soil modulus: %.0f MPa\n', Es_surface);
    fprintf('  Load pressure: %.2f MPa\n', load_pressure);
    fprintf('  Load radius: %.1f cm\n', load_radius);
    
    % === 2. Determine layering strategy ===
    [layer_depths, modulus_ratios, subgrade_type, physics_interpretation] = ...
        determineLayeringStrategy(Es_surface);
    
    fprintf('\nLayering strategy:\n');
    fprintf('  Subgrade classification: %s\n', subgrade_type);
    fprintf('  Layer depths: [%s] m\n', sprintf('%.1f ', layer_depths));
    fprintf('  Modulus ratios: [%s]\n', sprintf('%.1f ', modulus_ratios));
    fprintf('  Physical interpretation: %s\n', physics_interpretation);
    
    % === 3. Extract pavement structure layers ===
    pavement_thickness = extractPavementLayers(thickness);
    
    fprintf('\nPavement structure:\n');
    fprintf('  Pavement layers: %d\n', length(pavement_thickness));
    fprintf('  Pavement thickness: [%s] cm\n', sprintf('%.1f ', pavement_thickness));
    
    % === 4. Generate subgrade sublayers ===
    [sublayer_thickness_cm, sublayer_modulus, sublayer_poisson, sublayer_info] = ...
        generateSubgradeSublayers(layer_depths, modulus_ratios, Es_surface);
    
    fprintf('\nSubgrade sublayer generation:\n');
    for i = 1:length(sublayer_thickness_cm)
        fprintf('  Sublayer%d: thickness %.1fcm, depth %.1f-%.1fm, modulus %.1fMPa, ratio %.2f\n', ...
            i, sublayer_thickness_cm(i), sublayer_info.depth_ranges(i,1), ...
            sublayer_info.depth_ranges(i,2), sublayer_modulus(i), modulus_ratios(i));
    end
    
    % === 5. Combine overall structural system ===
    adjusted_thickness = [pavement_thickness; sublayer_thickness_cm];
    
    fprintf('\nOverall structural system:\n');
    fprintf('  Total layers: %d (pavement %d + subgrade %d)\n', ...
        length(adjusted_thickness), length(pavement_thickness), length(sublayer_thickness_cm));
    fprintf('  Pavement thickness: %.1f cm\n', sum(pavement_thickness));
    fprintf('  Subgrade depth: %.1f m\n', max(layer_depths));
    
    % === 6. Semi-infinite approximation validation ===
    semi_infinite_assessment = validateSemiInfiniteApproximation(...
        load_pressure, load_radius, Es_surface, max(layer_depths), sum(pavement_thickness)/100);
    
    fprintf('\nSemi-infinite approximation validation:\n');
    fprintf('  Assessment: %s\n', semi_infinite_assessment.assessment);
    fprintf('  Safety factor: %.2f\n', semi_infinite_assessment.safety_factor);
    if semi_infinite_assessment.sufficient
        fprintf('  Status: Sufficient depth for semi-infinite approximation\n');
    else
        fprintf('  Recommendation: Extend modeling depth to %.1fm\n', semi_infinite_assessment.recommended_depth);
    end
    
    % === 7. Build boundary conditions structure ===
    boundary_conditions = buildBoundaryConditions(Es_surface, subgrade_type, ...
        layer_depths, modulus_ratios, sublayer_modulus, sublayer_poisson, ...
        physics_interpretation, semi_infinite_assessment);
    
    % === 8. Model validation ===
    validateMultilayerModel(adjusted_thickness, boundary_conditions);
    
    fprintf('\nMulti-layer elastic subgrade modeling completed\n');
    fprintf('=========================================\n');
    
catch ME
    fprintf('Error: Multi-layer elastic subgrade modeling failed: %s\n', ME.message);
    if ~isempty(ME.stack)
        fprintf('   Error location: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
    end
    
    % Emergency fallback
    adjusted_thickness = thickness;
    boundary_conditions = createEmergencyBoundaryConditions();
    
    fprintf('Warning: Using emergency default parameters\n');
end
end

%% ================ Core Functions ================

function [Es_surface, load_pressure, load_radius] = extractAndValidateParameters(load_params)
% Extract and validate input parameters

% Extract surface soil modulus
if isstruct(load_params) && isfield(load_params, 'soil_modulus')
    Es_surface = load_params.soil_modulus;
elseif isstruct(load_params) && isfield(load_params, 'subgrade_modulus')
    Es_surface = load_params.subgrade_modulus;
else
    Es_surface = 50; % Default value
    fprintf('Warning: Soil modulus parameter not found, using default 50 MPa\n');
end

% Validate modulus range
Es_surface = max(10, min(Es_surface, 500)); % Limit to 10-500 MPa

% Extract load parameters
if isstruct(load_params)
    load_pressure = getfield_default(load_params, 'load_pressure', 0.7);
    load_radius = getfield_default(load_params, 'load_radius', 21.3);
else
    load_pressure = 0.7;
    load_radius = 21.3;
end

% Validate load parameters
load_pressure = max(0.1, min(load_pressure, 2.0));
load_radius = max(10, min(load_radius, 50));
end

function value = getfield_default(struct_data, field_name, default_value)
% Helper function to get field with default value
if isfield(struct_data, field_name)
    value = struct_data.(field_name);
else
    value = default_value;
end
end

function [layer_depths, modulus_ratios, subgrade_type, physics_interpretation] = determineLayeringStrategy(Es_surface)
% Determine layering strategy based on surface soil modulus
% According to engineering classification standards

if Es_surface <= 30
    % Soft soil foundation (Es ≤ 30 MPa)
    subgrade_type = 'Soft soil foundation';
    layer_depths = [0.5, 1.0, 1.5, 2.0]; % m
    modulus_ratios = [1.0, 0.8, 1.2, 1.5];
    physics_interpretation = 'Surface weathering → soft interlayer → consolidation transition → deep enhancement';
    
elseif Es_surface >= 80
    % Hard soil foundation (Es ≥ 80 MPa)
    subgrade_type = 'Hard soil foundation';
    layer_depths = [0.8, 1.6, 2.4]; % m
    modulus_ratios = [1.0, 1.1, 1.2];
    physics_interpretation = 'Relatively uniform, gradual enhancement with depth';
    
else
    % Medium foundation (30 < Es < 80 MPa)
    subgrade_type = 'Medium soil foundation';
    layer_depths = [0.6, 1.2, 1.8, 2.4]; % m
    modulus_ratios = [1.0, 0.9, 1.1, 1.3];
    physics_interpretation = 'Moderate variation, slight depth hardening';
end
fprintf('    Subgrade classification: %s\n', subgrade_type);
fprintf('    Layer depths: [%s] m\n', sprintf('%.1f ', layer_depths));
fprintf('    Modulus ratios: [%s]\n', sprintf('%.1f ', modulus_ratios));
fprintf('    Physical interpretation: %s\n', physics_interpretation);
end

function pavement_thickness = extractPavementLayers(thickness)
% Extract pavement structure layer thickness

if length(thickness) >= 3
    pavement_thickness = thickness(1:3); % Surface, base, subbase
elseif length(thickness) >= 2
    pavement_thickness = [thickness(1:2); 15]; % Add default subbase
    fprintf('Warning: Added default subbase thickness 15cm\n');
else
    pavement_thickness = [12; 30; 20]; % Complete default structure
    fprintf('Warning: Using default pavement structure [12; 30; 20] cm\n');
end

% Ensure reasonable thickness
for i = 1:length(pavement_thickness)
    pavement_thickness(i) = max(pavement_thickness(i), 5); % Minimum 5cm
end
end

function [sublayer_thickness_cm, sublayer_modulus, sublayer_poisson, sublayer_info] = ...
    generateSubgradeSublayers(layer_depths, modulus_ratios, Es_surface)
% Generate subgrade sublayer parameters

num_sublayers = length(layer_depths);
sublayer_thickness_cm = zeros(num_sublayers, 1);
sublayer_modulus = zeros(num_sublayers, 1);
sublayer_poisson = zeros(num_sublayers, 1);
depth_ranges = zeros(num_sublayers, 2);

for i = 1:num_sublayers
    % Calculate sublayer thickness
    if i == 1
        thickness_m = layer_depths(i); % First layer starts from surface
        depth_top = 0;
    else
        thickness_m = layer_depths(i) - layer_depths(i-1);
        depth_top = layer_depths(i-1);
    end
    depth_bottom = layer_depths(i);
    
    % Convert units: m → cm
    sublayer_thickness_cm(i) = thickness_m * 100;
    
    % Calculate sublayer modulus: E_i = E_s,surface × R_i
    sublayer_modulus(i) = Es_surface * modulus_ratios(i);
    
    % Assign Poisson's ratio (adjust based on soil type)
    if Es_surface <= 30
        sublayer_poisson(i) = 0.45; % Soft soil high Poisson's ratio
    elseif Es_surface >= 80
        sublayer_poisson(i) = 0.35; % Hard soil low Poisson's ratio
    else
        sublayer_poisson(i) = 0.40; % Medium subgrade
    end
    
    % Record depth range
    depth_ranges(i, :) = [depth_top, depth_bottom];
end

% Assemble additional information
sublayer_info = struct();
sublayer_info.num_sublayers = num_sublayers;
sublayer_info.total_depth_m = max(layer_depths);
sublayer_info.depth_ranges = depth_ranges;
sublayer_info.thickness_distribution = sublayer_thickness_cm;
end

function semi_infinite_assessment = validateSemiInfiniteApproximation(load_pressure, load_radius, Es_surface, total_depth, pavement_thickness)
% Validate sufficiency of semi-infinite approximation

% Calculate characteristic influence depth
P = load_pressure * 1e6; % Pa
r = load_radius / 100;   % m
Es = Es_surface * 1e6;   % Pa

% Method 1: Based on Boussinesq theory (stress attenuation to 10% of surface value)
influence_depth_boussinesq = 2.5 * r;

% Method 2: Based on pavement engineering practice
pavement_width = 4.0; % m (assumed value)
influence_depth_practice = 1.5 * pavement_width;

% Method 3: Based on elastic modulus and load simplified estimation
influence_depth_elastic = sqrt(P * r^2 / Es) * 10;

% Take maximum influence depth
influence_depths = [influence_depth_boussinesq, influence_depth_practice, influence_depth_elastic];
max_influence_depth = max(influence_depths);

% Calculate safety factor
safety_factor = total_depth / max_influence_depth;

% Depth ratio validation
depth_ratio = total_depth / pavement_thickness; % Subgrade depth/pavement thickness

% Comprehensive assessment
if safety_factor >= 2.0 && depth_ratio >= 4.0
    assessment = 'EXCELLENT - Excellent semi-infinite approximation';
    sufficient = true;
elseif safety_factor >= 1.5 && depth_ratio >= 3.0
    assessment = 'GOOD - Good semi-infinite approximation';
    sufficient = true;
elseif safety_factor >= 1.2 && depth_ratio >= 2.5
    assessment = 'ACCEPTABLE - Acceptable semi-infinite approximation';
    sufficient = true;
else
    assessment = 'INSUFFICIENT - Insufficient semi-infinite approximation';
    sufficient = false;
end

recommended_depth = max_influence_depth * 2.0;

semi_infinite_assessment = struct(...
    'assessment', assessment, ...
    'safety_factor', safety_factor, ...
    'depth_ratio', depth_ratio, ...
    'max_influence_depth', max_influence_depth, ...
    'recommended_depth', recommended_depth, ...
    'sufficient', sufficient, ...
    'boussinesq_depth', influence_depth_boussinesq, ...
    'practice_depth', influence_depth_practice, ...
    'elastic_depth', influence_depth_elastic);
end

function boundary_conditions = buildBoundaryConditions(Es_surface, subgrade_type, ...
    layer_depths, modulus_ratios, sublayer_modulus, sublayer_poisson, ...
    physics_interpretation, semi_infinite_assessment)
% Build boundary conditions structure

boundary_conditions = struct();

% Basic model information
boundary_conditions.method = 'multilayer_subgrade';
boundary_conditions.modeling_type = 'multilayer_subgrade';
boundary_conditions.theory_basis = 'Multi-layer elastic model (based on engineering practice)';

% Material properties
boundary_conditions.layer_modulus = sublayer_modulus;
boundary_conditions.layer_poisson = sublayer_poisson;

% Stratification parameters (detailed record)
boundary_conditions.stratification_params = struct(...
    'Es_surface_MPa', Es_surface, ...
    'subgrade_type', subgrade_type, ...
    'layer_depths_m', layer_depths, ...
    'modulus_ratios', modulus_ratios, ...
    'num_sublayers', length(sublayer_modulus), ...
    'total_subgrade_depth_m', max(layer_depths), ...
    'physics_interpretation', physics_interpretation);

% Boundary condition handling
boundary_conditions.bottom_boundary = 'fixed_displacement';
boundary_conditions.lateral_boundary = 'symmetric_constraint';
boundary_conditions.bottom_description = 'Fixed displacement constraint at bottom (semi-infinite simulation)';
boundary_conditions.lateral_description = 'Symmetric constraint at sides (axisymmetric assumption)';

% Semi-infinite approximation information
boundary_conditions.semi_infinite_validation = semi_infinite_assessment;

% Calculate model summary
modulus_range = sprintf('%.1f - %.1f MPa', min(sublayer_modulus), max(sublayer_modulus));
boundary_conditions.model_summary = sprintf('%s, %d sublayers, depth %.1fm, modulus range %s', ...
    subgrade_type, length(sublayer_modulus), max(layer_depths), modulus_range);

% Design recommendations
if semi_infinite_assessment.sufficient
    boundary_conditions.design_notes = 'Modeling depth sufficient for capturing main subgrade response characteristics';
else
    boundary_conditions.design_notes = sprintf('Consider extending modeling depth to %.1fm for improved semi-infinite approximation', ...
        semi_infinite_assessment.recommended_depth);
end

% Timestamp
boundary_conditions.creation_time = datestr(now, 'yyyy-mm-dd HH:MM:SS');
end

function validateMultilayerModel(adjusted_thickness, boundary_conditions)
% Model validation

fprintf('\n=== Multi-layer Model Validation ===\n');

% Geometric validation
total_layers = length(adjusted_thickness);
pavement_layers = 3; % Assume first 3 layers are pavement
subgrade_layers = total_layers - pavement_layers;

pavement_depth_cm = sum(adjusted_thickness(1:pavement_layers));
subgrade_depth_m = boundary_conditions.stratification_params.total_subgrade_depth_m;

fprintf('Geometric validation:\n');
fprintf('  Total layers: %d (pavement %d + subgrade %d)\n', total_layers, pavement_layers, subgrade_layers);
fprintf('  Pavement thickness: %.1f cm\n', pavement_depth_cm);
fprintf('  Subgrade depth: %.1f m\n', subgrade_depth_m);
fprintf('  Depth ratio: %.1f\n', subgrade_depth_m / (pavement_depth_cm/100));

% Material validation
moduli = boundary_conditions.layer_modulus;
fprintf('Material validation:\n');
fprintf('  Modulus range: %.1f - %.1f MPa\n', min(moduli), max(moduli));
fprintf('  Modulus variation: %.1f%%\n', (max(moduli) - min(moduli)) / min(moduli) * 100);

% Physical reasonableness validation
physics_ok = true;
if max(moduli) / min(moduli) > 5
    fprintf('  Warning: Modulus variation range may be too large\n');
    physics_ok = false;
end

if subgrade_depth_m / (pavement_depth_cm/100) < 2
    fprintf('  Warning: Subgrade depth may be insufficient\n');
    physics_ok = false;
end

if physics_ok
    fprintf('  Status: Physical parameters reasonable\n');
end

fprintf('===============================\n');
end

function boundary_conditions = createEmergencyBoundaryConditions()
% Create emergency boundary conditions (used when main function fails)

boundary_conditions = struct();
boundary_conditions.method = 'multilayer_subgrade';
boundary_conditions.modeling_type = 'multilayer_subgrade';
boundary_conditions.theory_basis = 'Emergency default configuration';

% Default material properties
boundary_conditions.layer_modulus = [40; 50; 60]; % Simple 3-layer subgrade
boundary_conditions.layer_poisson = [0.40; 0.40; 0.35];

% Default stratification information
boundary_conditions.stratification_params = struct(...
    'Es_surface_MPa', 50, ...
    'subgrade_type', 'Default medium foundation', ...
    'layer_depths_m', [0.6, 1.2, 1.8], ...
    'modulus_ratios', [0.8, 1.0, 1.2], ...
    'num_sublayers', 3, ...
    'total_subgrade_depth_m', 1.8);

boundary_conditions.model_summary = 'Emergency default: medium foundation, 3 sublayers, depth 1.8m';
boundary_conditions.design_notes = 'Warning: This is emergency configuration, please check input parameters and re-run';
end