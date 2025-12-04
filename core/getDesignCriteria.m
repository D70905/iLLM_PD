function design_criteria = getDesignCriteria(user_input, parsed_params, standard_type)
% GETDESIGNCRITERIA Unified interface for pavement design allowable values
%
%   Intelligently switches between JTG D50-2017 and AASHTO 1993 standards
%   based on user input, parameters, or automatic detection.
%
% SYNTAX:
%   design_criteria = getDesignCriteria(user_input, parsed_params, standard_type)
%
% INPUTS:
%   user_input     - Natural language input string
%   parsed_params  - Parsed design parameters structure
%   standard_type  - Standard type: 'JTG', 'AASHTO', 'MEPDG', 'DUAL', or 'auto'
%
% OUTPUTS:
%   design_criteria - Unified format design criteria structure with:
%                    .allowable_values (σ_std, ε_std, D_std)
%                    .control_indices
%                    .selected_standard
%
% EXAMPLES:
%   design_criteria = getDesignCriteria(input, params, 'JTG');    % Chinese standard
%   design_criteria = getDesignCriteria(input, params, 'AASHTO'); % US standard
%   design_criteria = getDesignCriteria(input, params, 'auto');   % Auto-select
%
% Author: Jingyi Xie, Tongji University
% Last Modified: 2025

fprintf('Pavement design standard selection...\n');

% Standardize inputs
if nargin < 3 || isempty(standard_type)
    standard_type = 'auto';
end

original_type = standard_type;
standard_type = upper(standard_type);

% Unify standard name format (support multiple writing styles)
standard_type = strrep(standard_type, '-', '');  % ME-PDG → MEPDG
standard_type = strrep(standard_type, '_', '');  % ME_PDG → MEPDG
standard_type = strrep(standard_type, ' ', '');  % ME PDG → MEPDG

% Display standard name conversion if changed
if ~strcmp(upper(original_type), standard_type)
    fprintf('  Standard name unified: %s → %s\n', original_type, standard_type);
end

% Automatic standard selection
if strcmp(standard_type, 'AUTO')
    standard_type = autoSelectStandard(user_input, parsed_params);
    fprintf('  Auto-selected standard: %s\n', standard_type);
else
    fprintf('  Specified standard: %s\n', standard_type);
end

% Call appropriate standard allowable value function
try
    switch standard_type
        case 'JTG'
            fprintf('  Applying JTG D50-2017 standard...\n');
            design_criteria = getJTG50DesignCriteria(user_input, parsed_params);
            
        case 'AASHTO'
            fprintf('  Applying AASHTO 1993 standard...\n');
            design_criteria = getAASHTODesignCriteria(user_input, parsed_params);
            
        case 'MEPDG'
            fprintf('  Applying MEPDG standard...\n');
            design_criteria = getMEPDGDesignCriteria(user_input, parsed_params);
            
        case 'DUAL'
            % Dual-standard mode: calculate both and compare
            fprintf('  Dual-standard mode: JTG + AASHTO...\n');
            design_criteria = getDualStandardCriteria(user_input, parsed_params);
            
        otherwise
            fprintf('  Warning: Unknown standard type, defaulting to JTG\n');
            design_criteria = getJTG50DesignCriteria(user_input, parsed_params);
    end
    
    % Add standard selection information
    design_criteria.selected_standard = standard_type;
    
    fprintf('✓ Design allowable values retrieved [%s standard]\n', standard_type);
    
catch ME
    fprintf('✗ Standard allowable value retrieval failed: %s\n', ME.message);
    
    % Fallback to JTG defaults
    design_criteria = getDefaultDesignCriteria();
    design_criteria.selected_standard = 'JTG_DEFAULT';
    design_criteria.success = false;
    design_criteria.message = sprintf('Fallback to defaults, reason: %s', ME.message);
end

% Validate output format
design_criteria = validateDesignCriteria(design_criteria);

end

%% ================ Intelligent Standard Selection ================

function standard_type = autoSelectStandard(user_input, parsed_params)
% Automatically select the most appropriate design standard

fprintf('  Analyzing input for automatic standard selection...\n');

% Priority 0: Check LTPP data source
if isfield(parsed_params, 'data_source')
    if contains(parsed_params.data_source, 'LTPP', 'IgnoreCase', true)
        standard_type = 'MEPDG';
        fprintf('    Detected LTPP data source, selecting MEPDG standard\n');
        return;
    end
end

if isfield(parsed_params, 'ltpp_section_id') || ...
   isfield(parsed_params, 'ltpp_thickness') || ...
   isfield(parsed_params, 'ltpp_modulus')
    standard_type = 'MEPDG';
    fprintf('    Detected LTPP parameter fields, selecting MEPDG standard\n');
    return;
end

% Priority 1: Explicit user specification
if contains(user_input, {'MEPDG', 'Mechanistic-Empirical', 'LTPP'}, 'IgnoreCase', true)
    standard_type = 'MEPDG';
    fprintf('    Detected MEPDG/LTPP keywords\n');
    return;
end

if contains(user_input, {'AASHTO', 'American', 'US standard'}, 'IgnoreCase', true)
    standard_type = 'AASHTO';
    fprintf('    Detected AASHTO keywords\n');
    return;
end

if contains(user_input, {'JTG', 'Chinese', 'CN standard'}, 'IgnoreCase', true)
    standard_type = 'JTG';
    fprintf('    Detected JTG keywords\n');
    return;
end

% Priority 2: Determine from parameter characteristics
if isfield(parsed_params, 'standard')
    if contains(parsed_params.standard, 'AASHTO', 'IgnoreCase', true)
        standard_type = 'AASHTO';
        fprintf('    AASHTO standard specified in parameters\n');
        return;
    elseif contains(parsed_params.standard, 'JTG', 'IgnoreCase', true)
        standard_type = 'JTG';
        fprintf('    JTG standard specified in parameters\n');
        return;
    end
end

% Priority 3: Determine from unit system
if isfield(parsed_params, 'units')
    if strcmp(parsed_params.units, 'imperial') || strcmp(parsed_params.units, 'US')
        standard_type = 'AASHTO';
        fprintf('    Detected imperial units, selecting AASHTO\n');
        return;
    elseif strcmp(parsed_params.units, 'metric') || strcmp(parsed_params.units, 'SI')
        standard_type = 'JTG';
        fprintf('    Detected metric units, selecting JTG\n');
        return;
    end
end

% Priority 4: Determine from design method
if isfield(parsed_params, 'design_method')
    if contains(parsed_params.design_method, {'ESAL', 'structural number', 'layer coefficient'}, 'IgnoreCase', true)
        standard_type = 'AASHTO';
        fprintf('    Detected AASHTO design method\n');
        return;
    end
end

% Default: Use JTG standard
standard_type = 'JTG';
fprintf('    Default to JTG standard\n');
end

%% ================ Dual-Standard Mode ================

function design_criteria = getDualStandardCriteria(user_input, parsed_params)
% Calculate and compare both JTG and AASHTO standards simultaneously

fprintf('  Computing dual-standard allowable values...\n');

% Calculate JTG standard
fprintf('\n  --- JTG D50-2017 ---\n');
criteria_JTG = getJTG50DesignCriteria(user_input, parsed_params);

% Calculate AASHTO standard
fprintf('\n  --- AASHTO 1993 ---\n');
criteria_AASHTO = getAASHTODesignCriteria(user_input, parsed_params);

% Create dual-standard structure
design_criteria = struct();
design_criteria.JTG = criteria_JTG;
design_criteria.AASHTO = criteria_AASHTO;

% Comparative analysis
design_criteria.comparison = compareStandards(criteria_JTG, criteria_AASHTO);

% Recommended standard
design_criteria.recommended_standard = recommendStandard(design_criteria.comparison);

% Use recommended standard's allowable values as primary output
if strcmp(design_criteria.recommended_standard, 'JTG')
    design_criteria.allowable_values = criteria_JTG.allowable_values;
    design_criteria.control_indices = criteria_JTG.control_indices;
else
    design_criteria.allowable_values = criteria_AASHTO.allowable_values;
    design_criteria.control_indices = criteria_AASHTO.control_indices;
end

design_criteria.pavement_type = criteria_JTG.pavement_type;
design_criteria.road_class = criteria_JTG.road_class;
design_criteria.traffic_level = criteria_JTG.traffic_level;

design_criteria.success = true;
design_criteria.message = 'Dual-standard comparison complete';
design_criteria.standard = 'JTG + AASHTO';
design_criteria.version = 'Dual_Standard_v1.0';

fprintf('\n  ✓ Dual-standard comparison complete\n');
fprintf('  Recommended: %s standard\n', design_criteria.recommended_standard);
end

function comparison = compareStandards(criteria_JTG, criteria_AASHTO)
% Compare allowable values between two standards

comparison = struct();

% Extract allowable values
jtg_vals = criteria_JTG.allowable_values;
aashto_vals = criteria_AASHTO.allowable_values;

% Calculate differences
comparison.stress_diff = (aashto_vals.surface_tensile_stress - jtg_vals.surface_tensile_stress) / ...
                         jtg_vals.surface_tensile_stress * 100;
comparison.strain_diff = (aashto_vals.base_tensile_strain - jtg_vals.base_tensile_strain) / ...
                         jtg_vals.base_tensile_strain * 100;
comparison.deflection_diff = (aashto_vals.subgrade_deflection - jtg_vals.subgrade_deflection) / ...
                             jtg_vals.subgrade_deflection * 100;

fprintf('\n  Standard comparison results:\n');
fprintf('    Surface stress: JTG=%.3f MPa, AASHTO=%.3f MPa (Diff: %+.1f%%)\n', ...
    jtg_vals.surface_tensile_stress, aashto_vals.surface_tensile_stress, comparison.stress_diff);
fprintf('    Base strain: JTG=%.0f με, AASHTO=%.0f με (Diff: %+.1f%%)\n', ...
    jtg_vals.base_tensile_strain, aashto_vals.base_tensile_strain, comparison.strain_diff);
fprintf('    Subgrade deflection: JTG=%.2f mm, AASHTO=%.2f mm (Diff: %+.1f%%)\n', ...
    jtg_vals.subgrade_deflection, aashto_vals.subgrade_deflection, comparison.deflection_diff);

% Calculate overall strictness
comparison.jtg_strictness = calculateStrictness(jtg_vals);
comparison.aashto_strictness = calculateStrictness(aashto_vals);

fprintf('    Overall strictness: JTG=%.2f, AASHTO=%.2f\n', ...
    comparison.jtg_strictness, comparison.aashto_strictness);
end

function strictness = calculateStrictness(allowable_vals)
% Calculate standard strictness (normalized metric, smaller = stricter)

stress_score = 1.0 / allowable_vals.surface_tensile_stress;
strain_score = 1000.0 / allowable_vals.base_tensile_strain;
deflection_score = 10.0 / allowable_vals.subgrade_deflection;

% Weighted average
strictness = 0.4 * stress_score + 0.3 * strain_score + 0.3 * deflection_score;
end

function recommended = recommendStandard(comparison)
% Recommend which standard to use

% Recommend based on strictness
if abs(comparison.jtg_strictness - comparison.aashto_strictness) < 0.2
    recommended = 'JTG';
    fprintf('  Standards similar, recommending JTG (Chinese engineering practice)\n');
elseif comparison.jtg_strictness > comparison.aashto_strictness
    recommended = 'JTG';
    fprintf('  JTG standard stricter, recommended for high-quality projects\n');
else
    recommended = 'AASHTO';
    fprintf('  AASHTO standard stricter, recommended for international projects\n');
end
end

%% ================ Format Validation ================

function design_criteria = validateDesignCriteria(design_criteria)
% Validate design criteria structure completeness

% Ensure required fields exist
required_fields = {'pavement_type', 'road_class', 'traffic_level', ...
                   'allowable_values', 'control_indices', 'success', 'standard'};

for i = 1:length(required_fields)
    if ~isfield(design_criteria, required_fields{i})
        fprintf('  Warning: Missing field: %s\n', required_fields{i});
        design_criteria.success = false;
    end
end

% Validate allowable values fields
if isfield(design_criteria, 'allowable_values')
    required_values = {'surface_tensile_stress', 'base_tensile_strain', 'subgrade_deflection'};
    for i = 1:length(required_values)
        if ~isfield(design_criteria.allowable_values, required_values{i})
            fprintf('  Warning: Allowable values missing field: %s\n', required_values{i});
        end
    end
end

% Validate numerical reasonableness
if isfield(design_criteria, 'allowable_values')
    vals = design_criteria.allowable_values;
    
    if vals.surface_tensile_stress < 0.1 || vals.surface_tensile_stress > 2.0
        fprintf('  Warning: Surface stress outside reasonable range\n');
    end
    
    if vals.base_tensile_strain < 100 || vals.base_tensile_strain > 3000
        fprintf('  Warning: Base strain outside reasonable range\n');
    end
    
    if vals.subgrade_deflection < 1.0 || vals.subgrade_deflection > 50.0
        fprintf('  Warning: Subgrade deflection outside reasonable range\n');
    end
end
end

function default_criteria = getDefaultDesignCriteria()
% Get universal default design criteria

default_criteria = struct();
default_criteria.pavement_type = 'semi_rigid';
default_criteria.road_class = 'highway';
default_criteria.traffic_level = 'medium';

default_criteria.control_indices = struct();
default_criteria.control_indices.primary = {'surface_tensile_stress', 'base_tensile_strain', 'subgrade_deflection'};
default_criteria.control_indices.critical = 'surface_tensile_stress';

default_criteria.allowable_values = struct();
default_criteria.allowable_values.surface_tensile_stress = 0.55;
default_criteria.allowable_values.base_tensile_strain = 650;
default_criteria.allowable_values.subgrade_deflection = 9.0;

default_criteria.success = false;
default_criteria.message = 'Using universal default values';
default_criteria.standard = 'DEFAULT';
default_criteria.version = 'Universal_Default_v1.0';
end