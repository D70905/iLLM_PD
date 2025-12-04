function runCompleteOptimization()
% [Fixed Version] Complete Road Structure Optimization Process
% 
% Fixed Contents:
% 1. Remove duplicate user input
% 2. Optimize design standard selection (user can choose JTG/MEPDG)
% 3. Simplify output information
% 4. [New] Standard selection interaction
% 5. [New] Parameter parsing result display
% 
% Major Upgrades:
% - Adapt to three-dimensional state space [σ_ratio, ε_ratio, D_ratio]
% - User-friendly standard selection interface
% - Concise output format

% Check if ablation experiment mode
global ablation_mode;

% Detect comparison experiment mode
if strcmp(ablation_mode, 'subgrade_comparison')
    runSubgradeMethodComparison();
    return;
end

if ~isempty(ablation_mode)
    fprintf('Current running mode: Ablation experiment - %s\n', ablation_mode);
    return;
end

% Modified:
% Check if ablation experiment mode
global ablation_mode;

% Validate ablation mode validity
valid_modes = {'subgrade_comparison', 'no_llm_parsing', 'no_llm_guidance', 'reduced_stability', 'full_system'};

if ~isempty(ablation_mode)
    if strcmp(ablation_mode, 'subgrade_comparison')
        runSubgradeMethodComparison();
        return;
    elseif ismember(ablation_mode, valid_modes)
        fprintf('Current running mode: Ablation experiment - %s\n', ablation_mode);
        % Continue execution, but do not return
    else
        fprintf('⚠️ Invalid ablation mode: %s, reset to full_system\n', ablation_mode);
        ablation_mode = 'full_system';
    end
end

fprintf('=== Three-Dimensional State Space Intelligent Road Structure Optimization System ===\n');
fprintf('Road Structure Design Based on Large Language Model and Reinforcement Learning\n');
fprintf('Core Upgrades: 3D State Space + Reward Function + Dual-LLM System\n\n');

try
    % Get project root directory
    project_root = fileparts(mfilename('fullpath'));
    core_path = fullfile(project_root, 'core');
    utils_path = fullfile(project_root, 'utils');
    
    % Add necessary paths
    paths_to_add = {core_path, utils_path};
    for i = 1:length(paths_to_add)
        path_dir = paths_to_add{i};
        if exist(path_dir, 'dir') && ~contains(path, path_dir)
            addpath(path_dir);
            fprintf('✅ Path added: %s\n', path_dir);
        end
    end

    % Record overall start time
    overall_start_time = tic;

    % Step 0: Load configuration file
    fprintf('\nStep 0: Loading configuration file...\n');
    config = loadSimplifiedConfig();

    % Step 1: Get user input (only once)
    fprintf('\nStep 1: Getting user design requirements...\n');
    user_input = getUserInput();

    % Step 2: Parse natural language
    fprintf('\nStep 2: Parsing design requirements...\n');
    parsed_params = parseUserInput(user_input);
    
    % [New] Display parameter parsing results
    displayParsedParameters(parsed_params);

    % [Fixed] Step 3: Optimized design standard acquisition (user choice + simplified output)
    fprintf('\nStep 3: Getting design standard allowable values...\n');
    design_criteria = getDesignCriteriaWithUserChoice(user_input, parsed_params);
    
    % Verify 3D allowable value interface
    if ~isfield(design_criteria, 'allowable_values')
        error('Design standard missing allowable_values field');
    end
    
    required_3d_fields = {'surface_tensile_stress', 'base_tensile_strain', 'subgrade_deflection'};
    for i = 1:length(required_3d_fields)
        field = required_3d_fields{i};
        if ~isfield(design_criteria.allowable_values, field)
            error('3D allowable values missing field: %s', field);
        end
    end
    
    fprintf('✅ 3D allowable value interface verification passed\n');

    % Step 3.5: Confirm subgrade treatment method
    fprintf('\nStep 3.5: Confirming subgrade treatment method...\n');
    parsed_params = confirmSubgradeMethodFixed(parsed_params);

    % Step 4: Initial PDE modeling validation (3D indicator extraction)
    fprintf('\nStep 4: Initial PDE finite element modeling (3D indicator extraction)...\n');
    initial_pde_results = performInitialPDE3DModeling(parsed_params);
    
    % Verify 3D PDE result interface
    required_pde_fields = {'sigma_FEA', 'epsilon_FEA', 'D_FEA'};
    for i = 1:length(required_pde_fields)
        field = required_pde_fields{i};
        if ~isfield(initial_pde_results, field)
            fprintf('⚠️ PDE result missing field %s, supplementing default value\n', field);
            switch field
                case 'sigma_FEA'
                    initial_pde_results.sigma_FEA = 0.65;
                case 'epsilon_FEA'
                    initial_pde_results.epsilon_FEA = 500;
                case 'D_FEA'
                    initial_pde_results.D_FEA = 6.0;
            end
        end
    end
    
    fprintf('✅ 3D PDE result interface verification passed\n');

    % Check consistency between 3D standard values and PDE results
    check3DStandardsConsistency(design_criteria, initial_pde_results);

    % Display 3D state space preview
    fprintf('\n📋 === 3D State Space Preview ===\n');
    [preview_pavement, preview_subgrade] = separatePavementAndSubgrade(parsed_params);
    fprintf('Will optimize pavement structure: %d layers\n', length(preview_pavement.thickness));
    fprintf('Will protect subgrade structure: %d layers\n', preview_subgrade.num_layers);
    
    % Display 3D state calculation
    sigma_ratio = initial_pde_results.sigma_FEA / design_criteria.allowable_values.surface_tensile_stress;
    epsilon_ratio = initial_pde_results.epsilon_FEA / design_criteria.allowable_values.base_tensile_strain;
    D_ratio = initial_pde_results.D_FEA / design_criteria.allowable_values.subgrade_deflection;
    
    fprintf('3D state space initial values:\n');
    fprintf('  σ_ratio = %.3f (%.3f/%.3f MPa)\n', ...
        sigma_ratio, initial_pde_results.sigma_FEA, design_criteria.allowable_values.surface_tensile_stress);
    fprintf('  ε_ratio = %.3f (%.0f/%.0f με)\n', ...
        epsilon_ratio, initial_pde_results.epsilon_FEA, design_criteria.allowable_values.base_tensile_strain);
    fprintf('  D_ratio = %.3f (%.2f/%.2f mm)\n', ...
        D_ratio, initial_pde_results.D_FEA, design_criteria.allowable_values.subgrade_deflection);
    fprintf('====================\n\n');

    % Step 5: Start 3D state space reinforcement learning optimization
    fprintf('\nStep 5: Starting 3D state space reinforcement learning optimization...\n');
    display3DTrainingInfo(config, design_criteria);

    rl_start_time = tic;

    % Save original parameters for subgrade protection
    original_full_params = parsed_params;

    % Call 3D PPO optimization system
    fprintf('🤖 Calling 3D state space PPO optimization system...\n');
    fprintf('   State space: 3D [σ_ratio, ε_ratio, D_ratio]\n');
    fprintf('   LLM system: Dual-LLM (price + engineering)\n');
    
    [optimized_params, training_log] = runPPOOptimization(...
        parsed_params, config, design_criteria, initial_pde_results);

    % Enhanced subgrade protection verification and forced correction
    fprintf('\n🔧 Executing enhanced subgrade protection verification...\n');
    optimized_params = enforceSubgradeProtectionEnhanced(optimized_params, original_full_params);

    rl_end_time = toc(rl_start_time);
    fprintf('✅ 3D PPO optimization completed, time elapsed %.2f seconds\n', rl_end_time);

    % Step 6: Final PDE validation (3D indicators)
    fprintf('\nStep 6: Final PDE validation (3D indicator extraction)...\n');
    final_pde_results = performFinalPDE3DValidation(optimized_params);
    
    % Validate final 3D results
    if final_pde_results.success
        fprintf('✅ Final 3D indicators:\n');
        fprintf('   σ_FEA = %.4f MPa (surface layer bottom tensile stress)\n', final_pde_results.sigma_FEA);
        fprintf('   ε_FEA = %.2f με (base layer bottom tensile strain)\n', final_pde_results.epsilon_FEA);
        fprintf('   D_FEA = %.3f mm (subgrade top deflection)\n', final_pde_results.D_FEA);
    end

    % Step 7: Subgrade protection validation
    fprintf('\nStep 7: Final subgrade protection effect validation...\n');
    validateFinalSubgradeProtectionEnhanced(parsed_params, optimized_params);

    % Step 8: 3D result display and save
    fprintf('\nStep 8: 3D optimization result display and save...\n');
    overall_end_time = toc(overall_start_time);

    % Add time statistics
    training_log.reinforcement_learning_time = rl_end_time;
    training_log.total_optimization_time = overall_end_time;
    training_log.optimization_method = 'Three_Dimensional_State_Space_PPO';

    % Result display
    display3DOptimizationResults(parsed_params, optimized_params, design_criteria, ...
        initial_pde_results, final_pde_results, training_log);

    % Display complete time statistics
    display3DTimeStatistics(training_log, overall_end_time);

    fprintf('\n=== ✅ 3D State Space Optimization Process Successfully Completed ===\n');

catch ME
    fprintf('\n❌ 3D optimization process failed: %s\n', ME.message);
    if ~isempty(ME.stack)
        fprintf('Error location: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);

        % Display more error stack information
        for i = 1:min(3, length(ME.stack))
            fprintf('  Stack%d: %s (line %d)\n', i, ME.stack(i).name, ME.stack(i).line);
        end
    end

    % Provide troubleshooting advice
    provide3DTroubleshootingAdvice(ME);
end
end

%% ==================== [New] Standard Selection Functions ====================

function design_criteria = getDesignCriteriaWithUserChoice(user_input, parsed_params)
% [New] Optimized design standard acquisition - user choice + simplified output

fprintf('\n📖 === Pavement Design Standard Selection ===\n');
fprintf('Please select pavement design standard:\n');
fprintf('  1. JTG D50-2017 (Chinese Highway Asphalt Pavement Design Specification, Recommended)\n');
fprintf('  2. ME-PDG (Mechanistic-Empirical Pavement Design Guide, NCHRP 1-37A)\n');

try
    user_choice = input('Please select standard (1/2, default 1): ', 's');
    
    if isempty(user_choice) || strcmp(user_choice, '1')
        fprintf('✅ Using standard: JTG D50-2017 (Chinese Standard)\n');
        design_criteria = getDesignCriteria(user_input, parsed_params, 'JTG');
        standard_name = 'JTG D50-2017';
        
    elseif strcmp(user_choice, '2')
        fprintf('✅ Using standard: ME-PDG (Mechanistic-Empirical Pavement Design Guide)\n');
        design_criteria = getDesignCriteria(user_input, parsed_params, 'MEPDG');
        standard_name = 'ME-PDG';
        
    else
        fprintf('⚠️ Invalid selection, using default: JTG D50-2017\n');
        design_criteria = getDesignCriteria(user_input, parsed_params, 'JTG');
        standard_name = 'JTG D50-2017';
    end
    
    % Validate acquisition success
    if isfield(design_criteria, 'success') && design_criteria.success
        % Concise display of allowable values
        fprintf('\n📊 %s Design Allowable Values:\n', standard_name);
        av = design_criteria.allowable_values;
        fprintf('  ├─ Surface tensile stress σ_std: %.3f MPa\n', av.surface_tensile_stress);
        fprintf('  ├─ Base tensile strain ε_std: %.0f με\n', av.base_tensile_strain);
        fprintf('  └─ Subgrade deflection   D_std: %.2f mm\n', av.subgrade_deflection);
        fprintf('==============================\n');
    else
        fprintf('⚠️ Standard acquisition may have issues\n');
    end
    
catch ME
    fprintf('❌ Standard selection failed: %s\n', ME.message);
    fprintf('Using default: JTG D50-2017\n');
    design_criteria = getDesignCriteria(user_input, parsed_params, 'JTG');
end
end

%% ==================== [New] Parameter Parsing Result Display Function ====================

function displayParsedParameters(parsed_params)
% [New] Display detailed information of parameter parsing results

fprintf('\n📋 === Parameter Parsing Results ===\n');

% Display basic information
if isfield(parsed_params, 'road_type')
    fprintf('Road type: %s\n', parsed_params.road_type);
end

if isfield(parsed_params, 'traffic_level')
    fprintf('Traffic level: %s\n', parsed_params.traffic_level);
end

if isfield(parsed_params, 'vehicle_speed_kmh')
    fprintf('Design speed: %d km/h\n', parsed_params.vehicle_speed_kmh);
end

if isfield(parsed_params, 'subgrade_type')
    fprintf('Subgrade type: %s\n', parsed_params.subgrade_type);
end

if isfield(parsed_params, 'subgrade_treatment')
    fprintf('Subgrade treatment: %s\n', parsed_params.subgrade_treatment);
end

% Display layer structure information
fprintf('\nLayer structure parameters:\n');
fprintf('%-8s %-20s %-12s %-12s %-12s\n', 'Layer', 'Material', 'Thick(cm)', 'Modulus(MPa)', 'Poisson');
fprintf('%s\n', repmat('-', 1, 70));

num_layers = length(parsed_params.thickness);
for i = 1:num_layers
    layer_name = sprintf('Layer %d', i);
    
    if isfield(parsed_params, 'material') && i <= length(parsed_params.material)
        material_name = parsed_params.material{i};
    else
        material_name = 'Unknown Material';
    end
    
    thickness = parsed_params.thickness(i);
    modulus = parsed_params.modulus(i);
    poisson = parsed_params.poisson(i);
    
    fprintf('%-8s %-20s %-12.1f %-12.0f %-12.2f\n', ...
        layer_name, material_name, thickness, modulus, poisson);
end

% Display load parameters
if isfield(parsed_params, 'load_pressure') && isfield(parsed_params, 'load_radius')
    fprintf('\nLoad parameters:\n');
    fprintf('  Load pressure: %.2f MPa\n', parsed_params.load_pressure);
    fprintf('  Load radius: %.1f cm\n', parsed_params.load_radius);
end

% Display subgrade modeling method
if isfield(parsed_params, 'subgrade_modeling')
    fprintf('\nSubgrade modeling method: %s\n', parsed_params.subgrade_modeling);
end

% Display parsing metadata
if isfield(parsed_params, 'parsing_info')
    fprintf('\nParsing information:\n');
    if isfield(parsed_params.parsing_info, 'model_used')
        fprintf('  Model used: %s\n', parsed_params.parsing_info.model_used);
    end
    if isfield(parsed_params.parsing_info, 'success')
        if parsed_params.parsing_info.success
            fprintf('  Parsing status: ✅ Success\n');
        else
            fprintf('  Parsing status: ⚠️ Failed (using default values)\n');
            if isfield(parsed_params.parsing_info, 'error_message')
                fprintf('  Failure reason: %s\n', parsed_params.parsing_info.error_message);
            end
        end
    end
end

fprintf('========================\n\n');
end

%% ==================== [Keep Original] Other Core Functions ====================

function initial_pde_results = performInitialPDE3DModeling(parsed_params)
% Execute initial PDE modeling (3D indicator extraction)
fprintf('Executing initial PDE 3D modeling...\n');

try
    load_params = struct('load_pressure', 0.7, 'load_radius', 21.3);
    boundary_conditions = struct('method', parsed_params.subgrade_modeling);
    
    initial_pde_results = roadPDEModelingSimplified(parsed_params, load_params, boundary_conditions);
    
    if isfield(initial_pde_results, 'success') && initial_pde_results.success
        fprintf('✅ PDE 3D modeling successful\n');
        fprintf('   σ_FEA = %.4f MPa, ε_FEA = %.2f με, D_FEA = %.3f mm\n', ...
            initial_pde_results.sigma_FEA, initial_pde_results.epsilon_FEA, initial_pde_results.D_FEA);
    else
        fprintf('❌ PDE 3D modeling failed\n');
    end
    
catch ME
    fprintf('❌ PDE 3D modeling failed: %s\n', ME.message);
    initial_pde_results = createFailure3DResult(ME.message);
end
end

function final_pde_results = performFinalPDE3DValidation(optimized_params)
% Execute final PDE validation (3D indicators)
fprintf('Executing final PDE 3D validation...\n');

try
    boundary_conditions = struct();
    if isfield(optimized_params, 'subgrade_modeling')
        boundary_conditions.method = optimized_params.subgrade_modeling;
    else
        boundary_conditions.method = 'winkler_subgrade';
    end
    
    load_params = struct('load_pressure', 0.7, 'load_radius', 21.3);
    final_pde_results = roadPDEModelingSimplified(optimized_params, load_params, boundary_conditions);
    
    if isfield(final_pde_results, 'success') && final_pde_results.success
        fprintf('✅ Final PDE 3D validation successful\n');
    else
        fprintf('❌ Final PDE 3D validation failed\n');
    end
    
catch ME
    fprintf('❌ Final PDE 3D validation exception: %s\n', ME.message);
    final_pde_results = createFailure3DResult(ME.message);
end
end

function result = createFailure3DResult(error_message)
% Create failure result (3D version)
result = struct();
result.success = false;
result.message = error_message;
result.sigma_FEA = 0.65;
result.epsilon_FEA = 500;
result.D_FEA = 6.0;
end

function check3DStandardsConsistency(design_criteria, pde_results)
% Check consistency between 3D standard values and PDE results
fprintf('\n=== 3D Standard vs PDE Result Comparison ===\n');

if isfield(pde_results, 'success') && pde_results.success && isfield(design_criteria, 'allowable_values')
    av = design_criteria.allowable_values;
    
    stress_utilization = (pde_results.sigma_FEA / av.surface_tensile_stress) * 100;
    strain_utilization = (pde_results.epsilon_FEA / av.base_tensile_strain) * 100;
    deflection_utilization = (pde_results.D_FEA / av.subgrade_deflection) * 100;
    
    fprintf('3D utilization ratios:\n');
    fprintf('  Stress: %.1f%% | ', stress_utilization);
    fprintf('Strain: %.1f%% | ', strain_utilization);
    fprintf('Deflection: %.1f%%\n', deflection_utilization);
    
    if stress_utilization > 100 || strain_utilization > 100 || deflection_utilization > 100
        fprintf('⚠️ Indicators exceed limits, optimization needed\n');
    elseif stress_utilization < 50 && strain_utilization < 50 && deflection_utilization < 50
        fprintf('📈 Utilization ratio low, optimization potential exists\n');
    else
        fprintf('✅ Initial design reasonable\n');
    end
end
fprintf('============================\n');
end

function display3DTrainingInfo(config, design_criteria)
% Display 3D training information (simplified version)
fprintf('\n🎯 === 3D PPO Training Configuration ===\n');
fprintf('State space: 3D [σ_ratio, ε_ratio, D_ratio]\n');
fprintf('Training episodes: %d\n', config.ppo.max_episodes);
fprintf('Steps per episode: %d\n', config.ppo.max_steps_per_episode);
fprintf('Optimization target: Utilization 70-105%%, cost 380-460 yuan/m²\n');
fprintf('==========================\n');
end

function display3DOptimizationResults(initial_params, optimized_params, design_criteria, initial_pde, final_pde, training_log)
% Display 3D optimization results (simplified version)
fprintf('\n🎉 === Optimization Results ===\n');

pavement_layers = 3;
fprintf('Pavement structure parameters:\n');
fprintf(' Initial: Thickness[%s]cm, Modulus[%s]MPa\n', ...
    sprintf('%.1f ', initial_params.thickness(1:pavement_layers)), ...
    sprintf('%.0f ', initial_params.modulus(1:pavement_layers)));
fprintf(' Optimized: Thickness[%s]cm, Modulus[%s]MPa\n', ...
    sprintf('%.1f ', optimized_params.thickness(1:pavement_layers)), ...
    sprintf('%.0f ', optimized_params.modulus(1:pavement_layers)));

if initial_pde.success && final_pde.success && isfield(design_criteria, 'allowable_values')
    av = design_criteria.allowable_values;
    
    fprintf('\n3D indicator comparison:\n');
    fprintf(' Stress ratio: %.3f → %.3f\n', ...
        initial_pde.sigma_FEA/av.surface_tensile_stress, ...
        final_pde.sigma_FEA/av.surface_tensile_stress);
    fprintf(' Strain ratio: %.3f → %.3f\n', ...
        initial_pde.epsilon_FEA/av.base_tensile_strain, ...
        final_pde.epsilon_FEA/av.base_tensile_strain);
    fprintf(' Deflection ratio: %.3f → %.3f\n', ...
        initial_pde.D_FEA/av.subgrade_deflection, ...
        final_pde.D_FEA/av.subgrade_deflection);
end

fprintf('\nTraining statistics:\n');
fprintf(' Total episodes: %d, Successful: %d, Best reward: %.4f\n', ...
    training_log.total_episodes, training_log.successful_episodes, training_log.best_reward);
fprintf('==================\n');
end

function display3DTimeStatistics(training_log, overall_time)
% Display time statistics (simplified version)
fprintf('\n⏱️ Time Statistics:\n');
fprintf(' Reinforcement learning: %.2f seconds\n', training_log.reinforcement_learning_time);
fprintf(' Total time: %.2f seconds (%.2f minutes)\n', overall_time, overall_time/60);
end

function provide3DTroubleshootingAdvice(ME)
% Troubleshooting advice (simplified version)
fprintf('\n🔧 Troubleshooting Advice:\n');
error_msg = ME.message;

if contains(error_msg, 'state space') || contains(error_msg, '状态空间')
    fprintf(' - Check PPO state space configuration\n');
elseif contains(error_msg, 'allowable_values')
    fprintf(' - Check design standard allowable value acquisition\n');
elseif contains(error_msg, 'LLM')
    fprintf(' - Check LLM API configuration\n');
else
    fprintf(' - Check project file integrity\n');
end
end

%% ==================== [Keep Original] Support Functions ====================

function config = loadSimplifiedConfig()
config_file = 'config.json';

if exist(config_file, 'file')
    try
        fid = fopen(config_file);
        raw = fread(fid, inf);
        str = char(raw');
        fclose(fid);
        config = jsondecode(str);
        fprintf('✅ Configuration file loaded successfully\n');
        config = validateConfigSimplified(config);
    catch ME
        fprintf('⚠️ Configuration reading failed, using default configuration\n');
        config = createDefaultConfigSimplified();
    end
else
    fprintf('⚠️ Configuration file does not exist, creating default configuration\n');
    config = createDefaultConfigSimplified();
end
end

function config = validateConfigSimplified(config)
fprintf('Validating configuration parameters...\n');

if ~isfield(config, 'ppo')
    config.ppo = struct();
end

critical_ppo_fields = {'max_episodes', 'max_steps_per_episode', 'state_dimension', 'action_dimension'};
for i = 1:length(critical_ppo_fields)
    field = critical_ppo_fields{i};
    if ~isfield(config.ppo, field)
        switch field
            case 'max_episodes'
                config.ppo.max_episodes = 40;
            case 'max_steps_per_episode'
                config.ppo.max_steps_per_episode = 12;
            case 'state_dimension'
                config.ppo.state_dimension = 3;
            case 'action_dimension'
                config.ppo.action_dimension = 6;
        end
    end
end

if config.ppo.state_dimension ~= 3
    config.ppo.state_dimension = 3;
end

fprintf('✅ Configuration validation completed\n');
end

function config = createDefaultConfigSimplified()
config = struct();
config.ppo = struct(...
    'max_episodes', 40, ...
    'max_steps_per_episode', 12, ...
    'learning_rate', 3e-4, ...
    'state_dimension', 3, ...
    'action_dimension', 6, ...
    'batch_size', 24);

config.optimization = struct('use_real_pde', true, 'convergence_threshold', 0.10);
config.subgrade_protection = struct('enabled', true, 'pavement_layers', 3);
config.reward_weights = struct('safety_weight', 0.60, 'economy_weight', 0.25);
config.deepseek = struct('api_key', 'your_api_key_here', 'base_url', 'https://api.deepseek.com');

fprintf('✅ Default configuration created\n');
end

function user_input = getUserInput()
prompt = ['Please input road design requirements\n' ...
    '(e.g., Design heavy load expressway, surface layer using asphalt concrete...)\n' ...
    'Input: '];

try
    user_input = input(prompt, 's');
    if isempty(user_input)
        user_input = 'Design heavy load expressway base layer, using cement stabilized crushed stone, soft soil subgrade';
        fprintf('Using default input\n');
    end
catch
    user_input = 'Design heavy load expressway base layer, using cement stabilized crushed stone, soft soil subgrade';
    fprintf('Using default input\n');
end
end

function parsed_params = parseUserInput(user_input)
try
    if exist('parseDesignPrompt', 'file') == 2
        parsed_params = parseDesignPrompt(user_input);
        fprintf('✅ Natural language parsing successful\n');
    else
        fprintf('⚠️ parseDesignPrompt function does not exist, using default parameters\n');
        parsed_params = getDefaultParams();
    end
catch ME
    fprintf('❌ Parsing failed: %s, using default parameters\n', ME.message);
    parsed_params = getDefaultParams();
end
end

function params = getDefaultParams()
params = struct();
params.thickness = [15; 30; 20; 150];
params.modulus = [1200; 600; 200; 50];
params.poisson = [0.30; 0.25; 0.35; 0.45];
params.material = {'Asphalt Concrete'; 'Cement Stabilized Crushed Stone'; 'Graded Crushed Stone'; 'Improved Soil'};
params.load_pressure = 0.7;
params.load_radius = 21.3;
params.subgrade_modeling = 'winkler_springs';
end

function params = confirmSubgradeMethodFixed(params)
fprintf('\n🏗️  Subgrade Treatment Method Selection:\n');
fprintf('1. Winkler Spring Foundation Model (soft soil subgrade)\n');
fprintf('2. Multi-layer Elastic Model (general/hard soil subgrade)\n');

try
    user_choice = input('Please select (1/2, default 1): ', 's');
    
    if isempty(user_choice) || strcmp(user_choice, '1')
        params.subgrade_modeling = 'winkler_springs';
        fprintf('✅ Winkler Spring Foundation Model\n');
    else
        params.subgrade_modeling = 'multilayer_subgrade';
        fprintf('✅ Multi-layer Elastic Model\n');
    end
catch
    params.subgrade_modeling = 'winkler_springs';
    fprintf('✅ Default: Winkler Spring Foundation Model\n');
end
end

function [pavement_params, subgrade_params] = separatePavementAndSubgrade(parsed_params)
pavement_layers = 3;
total_layers = length(parsed_params.thickness);

pavement_params = struct();
pavement_params.thickness = parsed_params.thickness(1:min(pavement_layers, total_layers));
pavement_params.modulus = parsed_params.modulus(1:min(pavement_layers, total_layers));
pavement_params.poisson = parsed_params.poisson(1:min(pavement_layers, total_layers));

subgrade_params = struct();
if total_layers > pavement_layers
    subgrade_params.thickness = parsed_params.thickness((pavement_layers+1):end);
    subgrade_params.modulus = parsed_params.modulus((pavement_layers+1):end);
    subgrade_params.poisson = parsed_params.poisson((pavement_layers+1):end);
    subgrade_params.num_layers = total_layers - pavement_layers;
else
    subgrade_params.num_layers = 0;
end
end

function protected_params = enforceSubgradeProtectionEnhanced(optimized_params, original_params)
fprintf('🛡️ Subgrade protection...\n');
protected_params = optimized_params;
pavement_layers = 3;

if length(optimized_params.thickness) ~= length(original_params.thickness)
    protected_params.thickness = original_params.thickness;
    protected_params.modulus = original_params.modulus;
    protected_params.poisson = original_params.poisson;
    
    if length(optimized_params.thickness) >= pavement_layers
        protected_params.thickness(1:pavement_layers) = optimized_params.thickness(1:pavement_layers);
        protected_params.modulus(1:pavement_layers) = optimized_params.modulus(1:pavement_layers);
    end
end

total_layers = length(protected_params.thickness);
if total_layers > pavement_layers
    for i = (pavement_layers + 1):total_layers
        protected_params.thickness(i) = original_params.thickness(i);
        protected_params.modulus(i) = original_params.modulus(i);
        protected_params.poisson(i) = original_params.poisson(i);
    end
    fprintf('✅ Protected %d subgrade layers\n', total_layers - pavement_layers);
end
end

function validateFinalSubgradeProtectionEnhanced(initial_params, final_params)
fprintf('\n🛡️ Subgrade Protection Validation\n');

pavement_layers = 3;
total_layers = length(initial_params.thickness);

if total_layers <= pavement_layers
    fprintf('✅ Pavement structure only\n');
    return;
end

all_protected = true;
for i = (pavement_layers + 1):total_layers
    thickness_diff = abs(initial_params.thickness(i) - final_params.thickness(i));
    modulus_diff = abs(initial_params.modulus(i) - final_params.modulus(i));
    
    if thickness_diff > 1e-14 || modulus_diff > 1e-14
        fprintf('❌ Layer %d protection failed\n', i);
        all_protected = false;
    end
end

if all_protected
    fprintf('✅ Subgrade protection 100%% successful\n');
else
    error('Subgrade protection failed');
end
end