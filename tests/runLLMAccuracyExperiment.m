function runLLMAccuracyExperiment()
    % LLM parameter parsing accuracy experiment
    % Test Claude and Gemini models
    
    fprintf('=== LLM Parameter Parsing Accuracy Experiment ===\n');
    fprintf('Test models: Claude-3.5-Sonnet, Gemini-2.0-Flash\n');
    fprintf('Sample count: 100 professional complete expressions\n');
    fprintf('Evaluation dimensions: 5 core indicators\n\n');
    
    try
        % Create result directory
        result_dir = 'data/llm_test_results';
        if ~exist(result_dir, 'dir')
            mkdir(result_dir);
        end
        
        % Step 1: Create 100 professional test samples
        fprintf('Step 1: Creating 100 professional test samples...\n');
        [test_dataset, ground_truth] = create100ProfessionalSamples();

        
        % Step 2: Test each model
        % Fix: Configure Claude and Gemini models
        config_keys = {'claude', 'gemini'}; % Keys in config.json
        matlab_field_names = {'claude', 'gemini'}; % MATLAB compatible field names
        model_display_names = {'Claude-3.5-Sonnet', 'Gemini-2.0-Flash'}; % Display names

        results = struct();
        
        for i = 1:length(config_keys)
            config_key = config_keys{i}; % For reading config file and API calls
            matlab_field = matlab_field_names{i}; % For MATLAB struct fields
            display_name = model_display_names{i};
            
            fprintf('\n=== Testing %s ===\n', display_name);
            
            % Test current model (pass config_key for API calls)
            model_results = testSingleModel(config_key, display_name, test_dataset, ground_truth);
            
            % Fix: Use MATLAB compatible field names to store results
            results.(matlab_field) = model_results;
            
            % Save intermediate results (use config_key as filename)
            save(fullfile(result_dir, sprintf('%s_results.mat', config_key)), 'model_results');
            
            fprintf('%s testing completed\n', display_name);
            
            % Add delay to avoid excessive API calls
            if i < length(config_keys)
                fprintf('Waiting 3 seconds before next model test...\n');
                pause(3);
            end
        end
        
        % Step 3: Generate comparison analysis
        fprintf('\nStep 3: Generating comparison analysis...\n');
        % Fix: Pass matlab_field_names parameter
        comparison_results = generateComparisonAnalysis(results, model_display_names, matlab_field_names);
        
        % Step 4: Generate experiment report
        fprintf('Step 4: Generating experiment report...\n');
        generateDetailedReport(results, comparison_results, result_dir);
        
        % Step 5: Save complete results
        save(fullfile(result_dir, 'complete_experiment_results.mat'), ...
            'results', 'comparison_results', 'test_dataset', 'ground_truth');
        
        fprintf('\n=== LLM Accuracy Experiment Completed ===\n');
        fprintf('Results saved to: %s\n', result_dir);
        
    catch ME
        fprintf('❌ Experiment failed: %s\n', ME.message);
        if ~isempty(ME.stack)
            fprintf('Error location: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
        end
        rethrow(ME);
    end
end

function [test_dataset, ground_truth] = create100ProfessionalSamples()
    % Create 100 professional complete expression samples
    
    fprintf('  Generating 100 professional samples...\n');
    
    % Sample templates and variables
    traffic_levels = {'Light Load', 'Medium Load', 'Heavy Load', 'Extra Heavy Load'};
    road_types = {'Expressway', 'Urban Arterial', 'Provincial Road', 'Industrial Road', 'Airport Road', 'Port Road'};
    surface_materials = {'SBS Modified Asphalt Concrete', 'High Modulus Asphalt Concrete', 'Dense Graded Asphalt Concrete', 'Modified Asphalt SMA', 'Heavy Duty Asphalt Concrete'};
    base_materials = {'Cement Stabilized Crushed Stone', 'Cement Stabilized Gravel', 'High Strength Cement Stabilized Crushed Stone', 'Fiber Reinforced Cement Stabilized Crushed Stone', 'Ultra High Strength Cement Stabilized Crushed Stone'};
    subbase_materials = {'Graded Crushed Stone', 'Graded Gravel', 'Natural Gravel', 'Graded Sand Gravel', 'Crushed Stone Cushion'};
    subgrade_types = {'Soft Soil', 'General Soil', 'Hard Soil', 'Improved Soil', 'Replacement Subgrade'};
    
    samples = cell(100, 1);
    detailed_ground_truth = cell(100, 1);
    
    % Generate 100 samples
    for i = 1:100
        % Randomly select basic parameters
        traffic = traffic_levels{randi(length(traffic_levels))};
        road = road_types{randi(length(road_types))};
        surface_mat = surface_materials{randi(length(surface_materials))};
        base_mat = base_materials{randi(length(base_materials))};
        subbase_mat = subbase_materials{randi(length(subbase_materials))};
        subgrade = subgrade_types{randi(length(subgrade_types))};
        
        % Adjust parameter ranges based on traffic level
        switch traffic
            case 'Light Load'
                surf_thick = randi([12, 16]);
                base_thick = randi([25, 35]);
                subbase_thick = randi([15, 25]);
                surf_mod = randi([1000, 1300]);
                base_mod = randi([300, 400]);
                subbase_mod = randi([120, 180]);
            case 'Medium Load'
                surf_thick = randi([15, 20]);
                base_thick = randi([30, 40]);
                subbase_thick = randi([18, 28]);
                surf_mod = randi([1200, 1450]);
                base_mod = randi([350, 450]);
                subbase_mod = randi([150, 200]);
            case 'Heavy Load'
                surf_thick = randi([18, 25]);
                base_thick = randi([35, 45]);
                subbase_thick = randi([22, 32]);
                surf_mod = randi([1300, 1600]);
                base_mod = randi([400, 520]);
                subbase_mod = randi([170, 220]);
            case 'Extra Heavy Load'
                surf_thick = randi([22, 28]);
                base_thick = randi([40, 50]);
                subbase_thick = randi([25, 35]);
                surf_mod = randi([1400, 1800]);
                base_mod = randi([450, 600]);
                subbase_mod = randi([180, 250]);
        end
        
        % Subgrade modulus
        switch subgrade
            case 'Soft Soil'
                subgrade_mod = randi([25, 40]);
            case 'General Soil'
                subgrade_mod = randi([45, 70]);
            case 'Hard Soil'
                subgrade_mod = randi([70, 100]);
            case 'Improved Soil'
                subgrade_mod = randi([50, 80]);
            case 'Replacement Subgrade'
                subgrade_mod = randi([60, 90]);
        end
        
        % Poisson ratios
        surface_poisson = 0.25 + (rand - 0.5) * 0.06;  % 0.22-0.28
        base_poisson = 0.25 + (rand - 0.5) * 0.08;     % 0.21-0.29
        subbase_poisson = 0.35 + (rand - 0.5) * 0.10;  % 0.30-0.40
        subgrade_poisson = 0.40 + (rand - 0.5) * 0.10; % 0.35-0.45
        
        % Construct sample text
        sample_text = sprintf(['Design %s %s asphalt pavement, surface layer thickness %dcm using %s modulus %dMPa Poisson ratio %.2f, ' ...
                              'base layer thickness %dcm using %s modulus %dMPa Poisson ratio %.2f, ' ...
                              'subbase layer thickness %dcm %s modulus %dMPa Poisson ratio %.2f, ' ...
                              'subgrade modulus %dMPa Poisson ratio %.2f %s foundation treatment'], ...
                              traffic, road, surf_thick, surface_mat, surf_mod, surface_poisson, ...
                              base_thick, base_mat, base_mod, base_poisson, ...
                              subbase_thick, subbase_mat, subbase_mod, subbase_poisson, ...
                              subgrade_mod, subgrade_poisson, subgrade);
        
        samples{i} = sample_text;
        
        % Store detailed ground truth
        detailed_ground_truth{i} = struct();
        detailed_ground_truth{i}.traffic_level = traffic;
        detailed_ground_truth{i}.road_type = road;
        detailed_ground_truth{i}.thickness = [surf_thick; base_thick; subbase_thick; 120]; % Add default subgrade thickness
        detailed_ground_truth{i}.modulus = [surf_mod; base_mod; subbase_mod; subgrade_mod];
        detailed_ground_truth{i}.poisson = [surface_poisson; base_poisson; subbase_poisson; subgrade_poisson];
        detailed_ground_truth{i}.materials = {surface_mat; base_mat; subbase_mat; subgrade};
    end
    
    % Create test dataset
    test_dataset = struct();
    test_dataset.samples = samples;
    test_dataset.total_count = 100;
    test_dataset.type = 'professional_complete';
    
    % Create ground truth
    fprintf('  Creating ground truth...\n');
    ground_truth = struct();
    ground_truth.detailed = detailed_ground_truth;
    
    fprintf('  ✅ 100 professional samples created\n');
end

function model_results = testSingleModel(model_name, display_name, test_dataset, ground_truth)
    % Test single model performance
    
    fprintf('  Starting %s test...\n', display_name);
    fprintf('  Total samples: %d\n', test_dataset.total_count);
    
    % Initialize result structure
    model_results = struct();
    model_results.model_name = model_name;
    model_results.display_name = display_name;
    model_results.total_samples = test_dataset.total_count;
    model_results.start_time = datetime('now');
    
    % Initialize counters
    counters = struct();
    counters.total_tested = 0;
    counters.api_success = 0;
    counters.parsing_success = 0;
    counters.traffic_correct = 0;
    counters.material_correct = 0;
    counters.thickness_correct = 0;
    counters.structure_correct = 0;
    
    % Track timing
    parsing_times = [];
    
    % Process samples
    samples = test_dataset.samples;
    
    for i = 1:length(samples)
        sample = samples{i};
        gt = ground_truth.detailed{i};
        
        if mod(i, 10) == 0
            fprintf('    Progress: %d/%d samples\n', i, length(samples));
        end
        
        try
            counters.total_tested = counters.total_tested + 1;
            
            % Measure parsing time
            parse_start = tic;
            result = parseDesignPrompt(sample, model_name);
            parse_time = toc(parse_start);
            parsing_times(end+1) = parse_time;
            
            % Check API success
            if isfield(result, 'parsing_info') && result.parsing_info.success
                counters.api_success = counters.api_success + 1;
                counters.parsing_success = counters.parsing_success + 1;
                
                % Evaluate parsing accuracy
                evaluateSingleSample(result, gt, counters);
            else
                % API call failed
                fprintf('      Sample %d: API call failed\n', i);
            end
            
        catch ME
            fprintf('      Sample %d: Error - %s\n', i, ME.message);
        end
        
        % Add small delay to avoid rate limiting
        if mod(i, 5) == 0
            pause(0.5);
        end
    end
    
    % Calculate accuracy rates
    accuracy_rates = struct();
    if counters.total_tested > 0
        accuracy_rates.api_stability = counters.api_success / counters.total_tested;
        accuracy_rates.traffic_level = counters.traffic_correct / counters.total_tested;
        accuracy_rates.material_param = counters.material_correct / counters.total_tested;
        accuracy_rates.thickness_param = counters.thickness_correct / counters.total_tested;
        accuracy_rates.structure_complete = counters.structure_correct / counters.total_tested;
        accuracy_rates.overall = (counters.traffic_correct + counters.material_correct + ...
                                 counters.thickness_correct + counters.structure_correct) / (4 * counters.total_tested);
    else
        accuracy_rates.api_stability = 0;
        accuracy_rates.traffic_level = 0;
        accuracy_rates.material_param = 0;
        accuracy_rates.thickness_param = 0;
        accuracy_rates.structure_complete = 0;
        accuracy_rates.overall = 0;
    end
    
    % Store results
    model_results.counters = counters;
    model_results.accuracy_rates = accuracy_rates;
    model_results.avg_parsing_time = mean(parsing_times);
    model_results.end_time = datetime('now');
    
    % Display summary
    fprintf('  %s Test Results:\n', display_name);
    fprintf('    API Success Rate: %.1f%% (%d/%d)\n', accuracy_rates.api_stability*100, counters.api_success, counters.total_tested);
    fprintf('    Traffic Level Accuracy: %.1f%%\n', accuracy_rates.traffic_level*100);
    fprintf('    Material Parameter Accuracy: %.1f%%\n', accuracy_rates.material_param*100);
    fprintf('    Thickness Parameter Accuracy: %.1f%%\n', accuracy_rates.thickness_param*100);
    fprintf('    Structure Completeness: %.1f%%\n', accuracy_rates.structure_complete*100);
    fprintf('    Overall Accuracy: %.1f%%\n', accuracy_rates.overall*100);
    fprintf('    Average Parsing Time: %.2f seconds\n', model_results.avg_parsing_time);
end

function evaluateSingleSample(result, ground_truth, counters)
    % Evaluate single sample parsing accuracy
    
    try
        % Traffic level evaluation
        if isfield(result, 'traffic_level')
            if strcmpi(result.traffic_level, ground_truth.traffic_level)
                counters.traffic_correct = counters.traffic_correct + 1;
            end
        end
        
        % Material parameter evaluation (simplified check)
        material_score = 0;
        if isfield(result, 'materials')
            if length(result.materials) >= 3
                material_score = 1;
            end
        end
        if material_score > 0.5
            counters.material_correct = counters.material_correct + 1;
        end
        
        % Thickness parameter evaluation
        thickness_score = 0;
        if isfield(result, 'thickness') && length(result.thickness) >= 3
            gt_thickness = ground_truth.thickness(1:3);
            for i = 1:3
                if abs(result.thickness(i) - gt_thickness(i)) < 3  % Within 3cm tolerance
                    thickness_score = thickness_score + 1/3;
                end
            end
        end
        if thickness_score > 0.5
            counters.thickness_correct = counters.thickness_correct + 1;
        end
        
        % Structure completeness evaluation
        structure_score = 0;
        required_fields = {'thickness', 'modulus', 'poisson'};
        for i = 1:length(required_fields)
            if isfield(result, required_fields{i})
                structure_score = structure_score + 1/3;
            end
        end
        if structure_score > 0.8
            counters.structure_correct = counters.structure_correct + 1;
        end
        
    catch ME
        fprintf('      Evaluation error: %s\n', ME.message);
    end
end

function comparison_results = generateComparisonAnalysis(results, model_display_names, matlab_field_names)
    % Generate comparison analysis results
    
    % Use MATLAB compatible field names
    num_models = length(matlab_field_names);
    
    % Create performance matrix [number of models x number of metrics]
    performance_matrix = zeros(num_models, 6);
    metrics = {'Traffic Level', 'Material Param', 'Thickness Param', 'Structure Completeness', 'API Stability', 'Overall Accuracy'};
    
    for i = 1:num_models
        matlab_field = matlab_field_names{i};
        model_result = results.(matlab_field);
        acc = model_result.accuracy_rates;
        
        performance_matrix(i, 1) = acc.traffic_level * 100;
        performance_matrix(i, 2) = acc.material_param * 100;
        performance_matrix(i, 3) = acc.thickness_param * 100;
        performance_matrix(i, 4) = acc.structure_complete * 100;
        performance_matrix(i, 5) = acc.api_stability * 100;
        performance_matrix(i, 6) = acc.overall * 100;
    end
    
    % Ranking analysis
    [~, ranking_idx] = sort(performance_matrix(:, 6), 'descend');
    ranking = model_display_names(ranking_idx);
    
    comparison_results = struct();
    comparison_results.matlab_field_names = matlab_field_names;
    comparison_results.model_display_names = model_display_names;
    comparison_results.metrics = metrics;
    comparison_results.performance_matrix = performance_matrix;
    comparison_results.ranking = ranking;
end

function generateDetailedReport(results, comparison_results, result_dir)
    % Generate detailed experiment report
    
    % 1. Generate performance comparison table
    generatePerformanceTable(comparison_results, result_dir);
    
    % 2. Generate text report
    generateTextReport(results, comparison_results, result_dir);
end

function generatePerformanceTable(comparison_results, result_dir)
    % Generate performance comparison table
    
    % Display table in console
    fprintf('\n=== LLM Parameter Parsing Accuracy Comparison Results ===\n');
    fprintf('%-20s%-12s%-12s%-12s%-15s%-12s%-12s\n', 'Model', 'Traffic Level', 'Material Param', 'Thickness Param', 'Structure Complete', 'API Stability', 'Overall Accuracy');
    fprintf('%-20s%-12s%-12s%-12s%-15s%-12s%-12s\n', repmat('-', 1, 20), repmat('-', 1, 12), repmat('-', 1, 12), repmat('-', 1, 12), repmat('-', 1, 15), repmat('-', 1, 12), repmat('-', 1, 12));
    
    for i = 1:length(comparison_results.model_display_names)
        model_name = comparison_results.model_display_names{i};
        data = comparison_results.performance_matrix(i, :);
        fprintf('%-20s%-12.1f%-12.1f%-12.1f%-15.1f%-12.1f%-12.1f\n', model_name, data(1), data(2), data(3), data(4), data(5), data(6));
    end
    fprintf('==============================================================================================================\n\n');
    
    % Save as CSV file
    csv_filename = fullfile(result_dir, 'performance_comparison.csv');
    csv_data = [{'Model'}, comparison_results.metrics; ...
                [comparison_results.model_display_names', num2cell(comparison_results.performance_matrix)]];
    writecell(csv_data, csv_filename);
    
    fprintf('    Table saved: %s\n', csv_filename);
end

function generateTextReport(results, comparison_results, result_dir)
    % Generate text report
    
    report_filename = fullfile(result_dir, 'experiment_report.txt');
    fid = fopen(report_filename, 'w');
    
    if fid == -1
        error('Cannot create report file');
    end
    
    try
        fprintf(fid, '=== LLM Parameter Parsing Accuracy Experiment Report ===\n\n');
        fprintf(fid, 'Experiment time: %s\n', datestr(now));
        fprintf(fid, 'Test samples: 100 professional complete expressions\n');
        fprintf(fid, 'Test models: %s\n', strjoin(comparison_results.model_display_names, ', '));
        fprintf(fid, 'Evaluation metrics: 5 dimensions (Traffic Level, Material Param, Thickness Param, Structure Completeness, API Stability)\n\n');
        
        % Overall ranking
        fprintf(fid, '== Overall Performance Ranking ==\n');
        for i = 1:length(comparison_results.ranking)
            model_name = comparison_results.ranking{i};
            model_idx = find(strcmp(comparison_results.model_display_names, model_name));
            overall_score = comparison_results.performance_matrix(model_idx, 6);
            fprintf(fid, '%d. %s: %.1f%%\n', i, model_name, overall_score);
        end
        fprintf(fid, '\n');
        
        % Best performance in each dimension
        fprintf(fid, '== Best Performance in Each Dimension ==\n');
        for i = 1:5
            metric_name = comparison_results.metrics{i};
            [best_score, best_idx] = max(comparison_results.performance_matrix(:, i));
            best_model = comparison_results.model_display_names{best_idx};
            fprintf(fid, '%s: %s (%.1f%%)\n', metric_name, best_model, best_score);
        end
        fprintf(fid, '\n');
        
        % Detailed analysis
        fprintf(fid, '== Detailed Analysis ==\n');
        for i = 1:length(comparison_results.matlab_field_names)
            matlab_field = comparison_results.matlab_field_names{i};
            model_result = results.(matlab_field);
            
            fprintf(fid, '%s:\n', model_result.display_name);
            fprintf(fid, '  Overall accuracy: %.1f%%\n', model_result.accuracy_rates.overall * 100);
            fprintf(fid, '  API successful calls: %d/%d (%.1f%%)\n', model_result.counters.api_success, ...
                model_result.total_samples, model_result.accuracy_rates.api_stability * 100);
            fprintf(fid, '  Average parsing time: %.2f seconds\n', model_result.avg_parsing_time);
            fprintf(fid, '  Performance in each dimension:\n');
            fprintf(fid, '    Traffic level: %.1f%%\n', model_result.accuracy_rates.traffic_level * 100);
            fprintf(fid, '    Material parameters: %.1f%%\n', model_result.accuracy_rates.material_param * 100);
            fprintf(fid, '    Thickness parameters: %.1f%%\n', model_result.accuracy_rates.thickness_param * 100);
            fprintf(fid, '    Structure completeness: %.1f%%\n', model_result.accuracy_rates.structure_complete * 100);
            fprintf(fid, '\n');
        end
        
        fclose(fid);
        fprintf('    Text report saved: %s\n', report_filename);
        
    catch ME
        fclose(fid);
        error('Report generation failed: %s', ME.message);
    end
end

% Fix: Update connection test function
function testConnection()
    % Quick connection test
    
    fprintf('⚡ === Model Connection Test ===\n');
    
    try
        sample = 'Design heavy load expressway asphalt pavement, surface layer thickness 20cm using SBS modified asphalt concrete modulus 1400MPa Poisson ratio 0.25';
        
        fprintf('Test sample: %s\n\n', sample);
     
        % Use correct configuration key names and model names
        config_keys = {'claude', 'gemini'}; % Keys in config.json
        model_names = {'Claude-3.5-Sonnet-20240620', 'Gemini-2.0-Flash'}; % Display names
        expected_models = {'claude-3-5-sonnet-20240620', 'gemini-2.0-flash'}; % Expected model names
        
        for i = 1:length(config_keys)
            fprintf('Testing %s (model: %s)...\n', model_names{i}, expected_models{i});
            
            try
                tic;
                result = parseDesignPrompt(sample, config_keys{i}); % Use config_keys
                elapsed = toc;
                
                if isfield(result, 'parsing_info') && result.parsing_info.success
                    fprintf('✅ %s parsing successful (%.2f seconds)\n', model_names{i}, elapsed);
                    if isfield(result, 'traffic_level') && isfield(result, 'road_type')
                        fprintf('   Traffic level: %s, Road type: %s\n', result.traffic_level, result.road_type);
                    else
                        fprintf('   Parsing result: %s\n', jsonencode(result));
                    end
                else
                    fprintf('❌ %s parsing failed\n', model_names{i});
                    if isfield(result, 'parsing_info') && isfield(result.parsing_info, 'error_message')
                        fprintf('   Error message: %s\n', result.parsing_info.error_message);
                    end
                end
                
            catch ME
                fprintf('❌ %s test failed: %s\n', model_names{i}, ME.message);
                if ~isempty(ME.stack)
                    fprintf('   Error location: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
                end
            end
            
            fprintf('\n');
            
            % Add delay to avoid excessive API calls
            if i < length(config_keys)
                pause(2);
            end
        end
        
    catch ME
        fprintf('❌ Connection test failed: %s\n', ME.message);
    end
    
    fprintf('Connection test completed!\n');
end