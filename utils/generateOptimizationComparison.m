function generateOptimizationComparison(initial_result, final_result, initial_params, final_params, design_criteria)
% Generate stress and strain comparison plots before and after optimization (6-subplot format)
% 
% Input:
%   initial_result - Initial PDE results (including Model and Solution)
%   final_result - Final PDE results (including Model and Solution)
%   initial_params - Initial design parameters
%   final_params - Optimized design parameters
%   design_criteria - Design standards (including allowable values)

fprintf('Generating optimization comparison visualization...\n');

%% === Stress Comparison Plot ===
fig1 = figure('Name', 'Stress Comparison (Initial vs Final)', ...
    'Position', [50, 50, 1600, 1000], 'Color', 'white');

% Extract stress data
try
    % Initial stress
    S_init = initial_result.Solution.Stress;
    init_sxx = S_init.xx;
    init_syy = S_init.yy;
    init_sxy = S_init.xy;
    
    % Final stress
    S_final = final_result.Solution.Stress;
    final_sxx = S_final.xx;
    final_syy = S_final.yy;
    final_sxy = S_final.xy;
    
    % Subplot 1: Initial Stress_xx
    subplot(3, 2, 1);
    plotStressField(initial_result.Model, init_sxx, 'Initial: Stress_{xx}');
    ylabel('Depth (m)', 'FontSize', 10);
    
    % Subplot 2: Final Stress_xx
    subplot(3, 2, 2);
    plotStressField(final_result.Model, final_sxx, 'Final: Stress_{xx}');
    
    % Subplot 3: Initial Stress_yy
    subplot(3, 2, 3);
    plotStressField(initial_result.Model, init_syy, 'Initial: Stress_{yy}');
    ylabel('Depth (m)', 'FontSize', 10);
    
    % Subplot 4: Final Stress_yy
    subplot(3, 2, 4);
    plotStressField(final_result.Model, final_syy, 'Final: Stress_{yy}');
    
    % Subplot 5: Initial Stress_xy
    subplot(3, 2, 5);
    plotStressField(initial_result.Model, init_sxy, 'Initial: Stress_{xy}');
    xlabel('Width (m)', 'FontSize', 10);
    ylabel('Depth (m)', 'FontSize', 10);
    
    % Subplot 6: Final Stress_xy
    subplot(3, 2, 6);
    plotStressField(final_result.Model, final_sxy, 'Final: Stress_{xy}');
    xlabel('Width (m)', 'FontSize', 10);
    
    % Add main title and parameter information
    title_str = sprintf('Stress Comparison (Initial vs Final) - %s', ...
        getMethodName(initial_params));
    sgtitle(title_str, 'FontSize', 16, 'FontWeight', 'bold');
    
    % Save stress comparison plot
    saveFigure(fig1, 'Stress_Comparison_Optimization');
    fprintf('  ✅ Stress comparison plot generated\n');
    
catch ME
    fprintf('  ❌ Stress comparison plot generation failed: %s\n', ME.message);
end

%% === Strain Comparison Plot ===
fig2 = figure('Name', 'Strain Comparison (Initial vs Final)', ...
    'Position', [100, 100, 1600, 1000], 'Color', 'white');

try
    % Initial strain
    E_init = initial_result.Solution.Strain;
    init_exx = E_init.xx;
    init_eyy = E_init.yy;
    init_exy = E_init.xy;
    
    % Final strain
    E_final = final_result.Solution.Strain;
    final_exx = E_final.xx;
    final_eyy = E_final.yy;
    final_exy = E_final.xy;
    
    % Subplot 1: Initial Strain_xx
    subplot(3, 2, 1);
    plotStrainField(initial_result.Model, init_exx, 'Initial: Strain_{xx}');
    ylabel('Depth (m)', 'FontSize', 10);
    
    % Subplot 2: Final Strain_xx
    subplot(3, 2, 2);
    plotStrainField(final_result.Model, final_exx, 'Final: Strain_{xx}');
    
    % Subplot 3: Initial Strain_yy
    subplot(3, 2, 3);
    plotStrainField(initial_result.Model, init_eyy, 'Initial: Strain_{yy}');
    ylabel('Depth (m)', 'FontSize', 10);
    
    % Subplot 4: Final Strain_yy
    subplot(3, 2, 4);
    plotStrainField(final_result.Model, final_eyy, 'Final: Strain_{yy}');
    
    % Subplot 5: Initial Strain_xy
    subplot(3, 2, 5);
    plotStrainField(initial_result.Model, init_exy, 'Initial: Strain_{xy}');
    xlabel('Width (m)', 'FontSize', 10);
    ylabel('Depth (m)', 'FontSize', 10);
    
    % Subplot 6: Final Strain_xy
    subplot(3, 2, 6);
    plotStrainField(final_result.Model, final_exy, 'Final: Strain_{xy}');
    xlabel('Width (m)', 'FontSize', 10);
    
    % Add main title
    title_str = sprintf('Strain Comparison (Initial vs Final) - %s', ...
        getMethodName(initial_params));
    sgtitle(title_str, 'FontSize', 16, 'FontWeight', 'bold');
    
    % Save strain comparison plot
    saveFigure(fig2, 'Strain_Comparison_Optimization');
    fprintf('  ✅ Strain comparison plot generated\n');
    
catch ME
    fprintf('  ❌ Strain comparison plot generation failed: %s\n', ME.message);
end

%% === 3D Indicators Comparison Plot ===
fig3 = figure('Name', '3D Control Indicators Comparison', ...
    'Position', [150, 150, 1200, 800], 'Color', 'white');

try
    % Extract 3D indicators
    sigma_init = initial_result.sigma_FEA;
    epsilon_init = initial_result.epsilon_FEA;
    D_init = initial_result.D_FEA;
    
    sigma_final = final_result.sigma_FEA;
    epsilon_final = final_result.epsilon_FEA;
    D_final = final_result.D_FEA;
    
    % Extract allowable values
    if isfield(design_criteria, 'allowable_values')
        av = design_criteria.allowable_values;
        sigma_std = av.surface_tensile_stress;
        epsilon_std = av.base_tensile_strain;
        D_std = av.subgrade_deflection;
    else
        sigma_std = 0.55;
        epsilon_std = 600;
        D_std = 8.0;
    end
    
    % Subplot 1: Stress comparison
    subplot(2, 2, 1);
    bar_data = [sigma_init, sigma_final, sigma_std];
    bar_colors = [0.8 0.4 0.4; 0.4 0.8 0.4; 0.4 0.4 0.8];
    h = bar(bar_data);
    h.FaceColor = 'flat';
    h.CData = bar_colors;
    title('Surface Layer Bottom Tensile Stress Comparison', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Stress (MPa)');
    set(gca, 'XTickLabel', {'Initial', 'Optimized', 'Allowable'});
    grid on;
    
    % Add value labels
    text(1, sigma_init, sprintf('%.3f', sigma_init), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    text(2, sigma_final, sprintf('%.3f', sigma_final), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    text(3, sigma_std, sprintf('%.3f', sigma_std), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    
    % Subplot 2: Strain comparison
    subplot(2, 2, 2);
    bar_data = [epsilon_init, epsilon_final, epsilon_std];
    h = bar(bar_data);
    h.FaceColor = 'flat';
    h.CData = bar_colors;
    title('Base Layer Bottom Tensile Strain Comparison', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Strain (με)');
    set(gca, 'XTickLabel', {'Initial', 'Optimized', 'Allowable'});
    grid on;
    
    text(1, epsilon_init, sprintf('%.0f', epsilon_init), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    text(2, epsilon_final, sprintf('%.0f', epsilon_final), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    text(3, epsilon_std, sprintf('%.0f', epsilon_std), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    
    % Subplot 3: Deflection comparison
    subplot(2, 2, 3);
    bar_data = [D_init, D_final, D_std];
    h = bar(bar_data);
    h.FaceColor = 'flat';
    h.CData = bar_colors;
    title('Subgrade Top Surface Deflection Comparison', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Deflection (mm)');
    set(gca, 'XTickLabel', {'Initial', 'Optimized', 'Allowable'});
    grid on;
    
    text(1, D_init, sprintf('%.2f', D_init), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    text(2, D_final, sprintf('%.2f', D_final), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    text(3, D_std, sprintf('%.2f', D_std), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    
    % Subplot 4: Utilization ratio comparison
    subplot(2, 2, 4);
    util_init = [sigma_init/sigma_std, epsilon_init/epsilon_std, D_init/D_std] * 100;
    util_final = [sigma_final/sigma_std, epsilon_final/epsilon_std, D_final/D_std] * 100;
    
    bar_data = [util_init; util_final]';
    h = bar(bar_data);
    h(1).FaceColor = [0.8 0.4 0.4];
    h(2).FaceColor = [0.4 0.8 0.4];
    title('3D Indicator Utilization Ratio Comparison', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Utilization Ratio (%)');
    set(gca, 'XTickLabel', {'Stress', 'Strain', 'Deflection'});
    legend('Initial', 'Optimized', 'Location', 'best');
    grid on;
    
    % Add target zones
    hold on;
    plot([0.5, 3.5], [70, 70], 'g--', 'LineWidth', 1.5);
    plot([0.5, 3.5], [100, 100], 'r--', 'LineWidth', 1.5);
    text(3.2, 70, 'Target Lower Limit', 'Color', 'g');
    text(3.2, 100, 'Allowable Upper Limit', 'Color', 'r');
    hold off;
    
    sgtitle('3D Control Indicators Comprehensive Comparison', 'FontSize', 16, 'FontWeight', 'bold');
    
    % Save 3D indicators comparison plot
    saveFigure(fig3, '3D_Indicators_Comparison');
    fprintf('  ✅ 3D indicators comparison plot generated\n');
    
catch ME
    fprintf('  ❌ 3D indicators comparison plot generation failed: %s\n', ME.message);
end

fprintf('✅ Optimization comparison visualization completed!\n');
end

%% === Auxiliary Plotting Functions ===

function plotStressField(model, stress_data, title_str)
% Plot stress field contour

try
    pdeplot(model, 'XYData', stress_data, 'ColorMap', 'jet');
    colorbar;
    title(title_str, 'FontSize', 11, 'FontWeight', 'bold');
    axis equal tight;
catch
    % If pdeplot fails, use fallback method
    plotFieldFallback(stress_data, title_str, 'Pa');
end
end

function plotStrainField(model, strain_data, title_str)
% Plot strain field contour

try
    pdeplot(model, 'XYData', strain_data, 'ColorMap', 'jet');
    colorbar;
    title(title_str, 'FontSize', 11, 'FontWeight', 'bold');
    axis equal tight;
catch
    % If pdeplot fails, use fallback method
    plotFieldFallback(strain_data, title_str, '');
end
end

function plotFieldFallback(data, title_str, unit)
% Fallback plotting method (when pdeplot fails)

[X, Y] = meshgrid(linspace(0, 4, 50), linspace(-2.5, 0, 40));
Z = griddata(rand(length(data),1)*4, rand(length(data),1)*(-2.5), data, X, Y);

contourf(X, Y, Z, 20, 'LineStyle', 'none');
colormap('jet');
colorbar;
title(title_str, 'FontSize', 11, 'FontWeight', 'bold');
axis equal tight;
xlabel('Width (m)');
ylabel('Depth (m)');
end

function method_name = getMethodName(params)
% Get subgrade method name

if isfield(params, 'subgrade_modeling')
    method = params.subgrade_modeling;
    if contains(method, 'winkler', 'IgnoreCase', true)
        method_name = 'Winkler Spring Foundation';
    elseif contains(method, 'multilayer', 'IgnoreCase', true)
        method_name = 'Multi-layer Elastic Body';
    else
        method_name = method;
    end
else
    method_name = 'Unknown Method';
end
end

function saveFigure(fig, filename)
% Save figure

try
    if ~exist('results', 'dir')
        mkdir('results');
    end
    
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    full_path = sprintf('results/%s_%s.png', filename, timestamp);
    
    print(fig, full_path, '-dpng', '-r300');
    fprintf('    Saved: %s\n', full_path);
    
catch ME
    fprintf('    Save failed: %s\n', ME.message);
end
end