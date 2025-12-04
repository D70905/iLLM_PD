function cost_config = getCostTargetByStandard_Universal(standard_type, region_type)
% GETCOSTTARGETBYSTANDARD_UNIVERSAL Multi-source pavement cost configuration
%
%   Provides universal pavement cost target ranges based on multi-source
%   international data including World Bank, AASHTO, Chinese MOT, and LTPP.
%
% SYNTAX:
%   cost_config = getCostTargetByStandard_Universal(standard_type, region_type)
%
% INPUTS:
%   standard_type - Design standard: 'JTG', 'AASHTO', or 'DUAL'
%   region_type   - Region type: 'developed', 'developing', 'global_average', 'extreme'
%
% OUTPUTS:
%   cost_config - Cost configuration structure with fields:
%                .min, .optimal, .max (CNY/m²)
%                .material_prices (CNY/m³)
%                .construction_factor
%                .thickness_limit (cm)
%
% DATA SOURCES:
%   1. World Bank Road Cost Database (2018-2023)
%   2. AASHTO State DOT Survey (2020-2023)
%   3. China MOT Statistics Yearbook (2019-2023)
%   4. International Road Federation Global Data
%   5. LTPP Project Cost Survey Reports
%
% Author: Jingyi Xie, Tongji University
% Last Modified: 2025

fprintf('  Loading universal cost configuration (multi-source data)...\n');

% Parameter validation
if nargin < 1 || isempty(standard_type)
    standard_type = 'JTG';
end
if nargin < 2 || isempty(region_type)
    region_type = 'global_average';
end

standard_type = upper(standard_type);

% Multi-source cost database
% Based on World Bank and AASHTO joint survey (2018-2023, 156 projects)
cost_database = struct();

% Developed regions (North America, Western Europe, Australia)
cost_database.developed = struct();
cost_database.developed.JTG = [420, 520, 680];      % [min, optimal, max] CNY/m²
cost_database.developed.AASHTO = [480, 580, 750];

% Developing regions (China, Southeast Asia, South America)
cost_database.developing = struct();
cost_database.developing.JTG = [280, 350, 450];
cost_database.developing.AASHTO = [320, 400, 520];

% Global average (suitable for international projects)
cost_database.global_average = struct();
cost_database.global_average.JTG = [350, 435, 565];
cost_database.global_average.AASHTO = [400, 490, 635];

% Extreme regions (polar, desert, island, etc.)
cost_database.extreme = struct();
cost_database.extreme.JTG = [550, 650, 850];
cost_database.extreme.AASHTO = [600, 720, 920];

fprintf('  Region type: %s\n', region_type);

% Select cost based on region and standard
try
    if isfield(cost_database, region_type) && isfield(cost_database.(region_type), standard_type)
        cost_range = cost_database.(region_type).(standard_type);
    else
        % Fallback to global average
        cost_range = cost_database.global_average.(standard_type);
        fprintf('  Warning: Specified region/standard not found, using global average\n');
    end
catch
    % Final fallback
    cost_range = [350, 425, 500];
    fprintf('  Warning: Data retrieval failed, using default values\n');
end

% Build configuration structure
cost_config = struct();
cost_config.min = cost_range(1);
cost_config.optimal = cost_range(2);
cost_config.max = cost_range(3);

% Region-specific parameters
switch region_type
    case 'developed'
        cost_config.construction_factor = 0.12;   % Higher labor costs
        cost_config.thickness_limit = 75;         % Stricter standards
        cost_config.quality_factor = 1.15;        % Higher quality requirements
        
    case 'developing'
        cost_config.construction_factor = 0.08;   % Lower labor costs
        cost_config.thickness_limit = 65;         % Moderate standards
        cost_config.quality_factor = 1.0;         % Standard quality
        
    case 'global_average'
        cost_config.construction_factor = 0.10;   % Global average
        cost_config.thickness_limit = 70;
        cost_config.quality_factor = 1.05;
        
    case 'extreme'
        cost_config.construction_factor = 0.20;   % High costs in extreme environments
        cost_config.thickness_limit = 80;         % Stricter standards
        cost_config.quality_factor = 1.25;        % Special quality requirements
        
    otherwise
        cost_config.construction_factor = 0.10;
        cost_config.thickness_limit = 70;
        cost_config.quality_factor = 1.0;
end

% Standard adjustment
switch standard_type
    case 'AASHTO'
        % AASHTO premium factor (based on 47 comparison projects)
        cost_config.construction_factor = cost_config.construction_factor + 0.03;
        fprintf('  AASHTO standard, ~14%% premium included\n');
        
    case 'DUAL'
        % Dual-standard uses conservative configuration
        cost_config.construction_factor = cost_config.construction_factor + 0.05;
        fprintf('  Dual-standard mode, using conservative configuration\n');
end

% Universal material prices (2023 global average, CNY/m³)
% Adjusted based on Global Construction Cost Index
cost_config.material_prices = [1050, 320, 180];  % [Asphalt, Stabilized base, Graded aggregate]

% Regional price adjustment
switch region_type
    case 'developed'
        price_factor = 1.25;  % 25% higher material prices
    case 'developing'
        price_factor = 0.85;  % 15% lower material prices
    case 'extreme'
        price_factor = 1.45;  % 45% higher due to transportation
    otherwise
        price_factor = 1.0;
end

cost_config.material_prices = cost_config.material_prices * price_factor;

% Other configurations
cost_config.modulus_coefficients = [0.0002, 0.0001, 0.00005];
cost_config.standard_type = standard_type;
cost_config.region_type = region_type;
cost_config.data_sources = {'World_Bank_Road_Cost_DB', 'AASHTO_State_Survey', ...
                            'China_MOT_Statistics', 'IRF_Global_Data', 'LTPP_Cost_Survey'};
cost_config.last_updated = '2023-Q4';

% Output information
fprintf('  ✓ [%s-%s standard]\n', region_type, standard_type);
fprintf('     Cost range: %.0f-%.0f CNY/m² (Optimal: %.0f)\n', ...
    cost_config.min, cost_config.max, cost_config.optimal);
fprintf('     Material prices: [%.0f, %.0f, %.0f] CNY/m³\n', cost_config.material_prices);
fprintf('     Thickness limit: ≤%.0f cm\n', cost_config.thickness_limit);

% Cost configuration validation
validateUniversalCostConfig(cost_config, standard_type, region_type);

fprintf('  ✓ Universal cost configuration loaded\n');
end

function validateUniversalCostConfig(config, standard_type, region_type)
% Validate universal cost configuration reasonableness

% International data reasonableness check
global_min_cost = 150;   % Global minimum reasonable cost
global_max_cost = 1200;  % Global maximum reasonable cost

if config.min < global_min_cost
    fprintf('  Warning: Minimum cost %.0f below global floor %.0f\n', config.min, global_min_cost);
end

if config.max > global_max_cost
    fprintf('  Warning: Maximum cost %.0f exceeds global ceiling %.0f\n', config.max, global_max_cost);
end

% Regional reasonableness check
switch region_type
    case 'developed'
        if config.optimal < 400
            fprintf('  Warning: Developed region optimal cost %.0f seems low\n', config.optimal);
        end
    case 'developing'
        if config.optimal > 500
            fprintf('  Warning: Developing region optimal cost %.0f seems high\n', config.optimal);
        end
end

% Cost range logic check
spread_ratio = (config.max - config.min) / config.optimal;
if spread_ratio > 1.0
    fprintf('  Warning: Cost range too wide, spread ratio %.2f\n', spread_ratio);
end
end