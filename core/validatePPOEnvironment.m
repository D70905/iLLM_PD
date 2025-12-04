function validatePPOEnvironment(parsed_params, config, theory_results, initial_pde_results)
% VALIDATEPPOENVIRONMENT Validate PPO training environment for multiple standards
%
%   Validates the PPO environment setup including design standards (JTG/AASHTO),
%   allowable values, PDE results, and configuration parameters before training.
%
% INPUTS:
%   parsed_params       - Parsed design parameters structure
%   config              - PPO configuration structure  
%   theory_results      - Design criteria with allowable values
%   initial_pde_results - Initial FEM analysis results
%
% Author: Jingyi Xie, Tongji University
% Last Modified: 2025

fprintf('Validating PPO environment...\n');

try
    % Extract actual standard type being used
    if isfield(theory_results, 'selected_standard')
        standard_type = theory_results.selected_standard;
    elseif isfield(theory_results, 'standard')
        standard_type = theory_results.standard;
    else
        standard_type = 'JTG';
        fprintf('  Warning: Standard type not specified, defaulting to JTG\n');
    end
    
    % Normalize standard type (handle case and spacing variations)
    if contains(standard_type, 'AASHTO', 'IgnoreCase', true)
        standard_type = 'AASHTO';
    elseif contains(standard_type, 'JTG', 'IgnoreCase', true)
        standard_type = 'JTG';
    end
    
    fprintf('  Standard: %s\n', standard_type);
    
    % Validate design criteria structure
    if ~isstruct(theory_results) || ~isfield(theory_results, 'success') || ~theory_results.success
        error('%s design criteria results invalid', standard_type);
    end
    
    % Check allowable values field
    if ~isfield(theory_results, 'allowable_values')
        error('Missing %s allowable values', standard_type);
    end
    
    allowable_values = theory_results.allowable_values;
    
    % Validate allowable value fields with unified naming
    if ~isfield(allowable_values, 'surface_tensile_stress') || ...
       isnan(allowable_values.surface_tensile_stress) || ...
       allowable_values.surface_tensile_stress <= 0
        error('%s surface tensile stress allowable value invalid', standard_type);
    end
    
    if ~isfield(allowable_values, 'base_tensile_strain') || ...
       isnan(allowable_values.base_tensile_strain') || ...
       allowable_values.base_tensile_strain <= 0
        error('%s base tensile strain allowable value invalid', standard_type);
    end
    
    if ~isfield(allowable_values, 'subgrade_deflection') || ...
       isnan(allowable_values.subgrade_deflection) || ...
       allowable_values.subgrade_deflection <= 0
        error('%s subgrade deflection allowable value invalid', standard_type);
    end
    
    % Display allowable value ranges
    fprintf('  ✓ %s design standard validation passed\n', standard_type);
    fprintf('    σ_std = %.3f MPa (Surface tensile stress)\n', allowable_values.surface_tensile_stress);
    fprintf('    ε_std = %.0f με (Base tensile strain)\n', allowable_values.base_tensile_strain);
    fprintf('    D_std = %.2f mm (Subgrade deflection)\n', allowable_values.subgrade_deflection);
    
    % Validate allowable value reasonableness for different standards
    if strcmp(standard_type, 'AASHTO')
        % AASHTO typical ranges
        if allowable_values.surface_tensile_stress < 0.3 || allowable_values.surface_tensile_stress > 2.0
            fprintf('  Warning: AASHTO surface stress outside typical range (0.3-2.0 MPa)\n');
        end
        if allowable_values.base_tensile_strain < 200 || allowable_values.base_tensile_strain > 2500
            fprintf('  Warning: AASHTO base strain outside typical range (200-2500 με)\n');
        end
        if allowable_values.subgrade_deflection < 3.0 || allowable_values.subgrade_deflection > 30.0
            fprintf('  Warning: AASHTO subgrade deflection outside typical range (3.0-30.0 mm)\n');
        end
    else
        % JTG typical ranges
        if allowable_values.surface_tensile_stress < 0.2 || allowable_values.surface_tensile_stress > 1.5
            fprintf('  Warning: JTG surface stress outside typical range (0.2-1.5 MPa)\n');
        end
        if allowable_values.base_tensile_strain < 200 || allowable_values.base_tensile_strain > 2500
            fprintf('  Warning: JTG base strain outside typical range (200-2500 με)\n');
        end
        if allowable_values.subgrade_deflection < 2.0 || allowable_values.subgrade_deflection > 30.0
            fprintf('  Warning: JTG subgrade deflection outside typical range (2.0-30.0 mm)\n');
        end
    end
    
    % Validate PDE results
    if ~isstruct(initial_pde_results) || ~isfield(initial_pde_results, 'success') || ~initial_pde_results.success
        fprintf('  Warning: PDE results invalid, will use estimated results\n');
    else
        if ~isfield(initial_pde_results, 'sigma_FEA') || isnan(initial_pde_results.sigma_FEA)
            error('PDE stress result invalid');
        end
        if ~isfield(initial_pde_results, 'epsilon_FEA') || isnan(initial_pde_results.epsilon_FEA)
            error('PDE strain result invalid');
        end
        if ~isfield(initial_pde_results, 'D_FEA') || isnan(initial_pde_results.D_FEA)
            error('PDE deflection result invalid');
        end
        fprintf('  ✓ PDE results validation passed\n');
    end
    
    % Validate design parameters
    if ~isstruct(parsed_params)
        error('Design parameter structure invalid');
    end
    
    required_fields = {'thickness', 'modulus', 'poisson'};
    for i = 1:length(required_fields)
        if ~isfield(parsed_params, required_fields{i})
            error('Missing required design parameter: %s', required_fields{i});
        end
    end
    
    fprintf('  ✓ Design parameters validation passed\n');
    
    % Validate configuration
    if ~isstruct(config)
        error('Configuration parameters invalid');
    end
    
    fprintf('  ✓ Configuration parameters validation passed\n');
    
    fprintf('✓ PPO environment validation complete [%s standard]\n', standard_type);
    
catch ME
    fprintf('✗ PPO environment validation failed: %s\n', ME.message);
    error('PPO environment validation failed, cannot proceed with training');
end
end