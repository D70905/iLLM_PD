classdef LLMDataLogger < handle
    % LLMDataLogger - Log LLM interaction data for ablation studies
    % For Nature Communications code availability requirements
    % Compatible with MATLAB R2020b and later versions
    
    properties
        output_dir           % Output directory
        optimization_id      % Optimization ID
        ltpp_reference      % LTPP reference data
        iteration_data      % Iteration log data
        episode_summary     % Episode summary data
        start_time          % Start timestamp
    end
    
    methods
        function obj = LLMDataLogger(ltpp_reference, optimization_id)
            % Constructor
            % Input:
            %   ltpp_reference - LTPP reference structure data
            %   optimization_id - Optimization ID string
            
            obj.optimization_id = optimization_id;
            obj.ltpp_reference = ltpp_reference;
            obj.start_time = datetime('now');
            
            % Create output directory
            obj.output_dir = fullfile('output', 'llm_logs', optimization_id);
            if ~exist(obj.output_dir, 'dir')
                mkdir(obj.output_dir);
            end
            
            % Initialize data structures
            obj.iteration_data = struct('episode', {}, 'step', {}, ...
                'params_before', {}, 'pde_results', {}, 'design_criteria', {}, ...
                'price_info', {}, 'engineering_advice', {}, 'reward', {}, ...
                'action_type', {}, 'timestamp', {});
            
            obj.episode_summary = struct('episode', {}, 'total_reward', {}, ...
                'llm_calls', {}, 'convergence_status', {}, 'duration', {});
            
            fprintf('📁 LLM Data Logger initialized: %s\n', obj.output_dir);
        end
        
        function logIteration(obj, episode, step, params_before, pde_results, ...
                design_criteria, price_info, engineering_advice, reward, action_type)
            % Log single iteration data
            
            iter_idx = length(obj.iteration_data) + 1;
            
            % Store iteration data
            obj.iteration_data(iter_idx).episode = episode;
            obj.iteration_data(iter_idx).step = step;
            obj.iteration_data(iter_idx).params_before = params_before;
            obj.iteration_data(iter_idx).pde_results = pde_results;
            obj.iteration_data(iter_idx).design_criteria = design_criteria;
            obj.iteration_data(iter_idx).price_info = price_info;
            obj.iteration_data(iter_idx).engineering_advice = engineering_advice;
            obj.iteration_data(iter_idx).reward = reward;
            obj.iteration_data(iter_idx).action_type = action_type;
            obj.iteration_data(iter_idx).timestamp = datetime('now');
        end
        
        function logEpisodeSummary(obj, episode, total_reward, llm_calls, ...
                convergence_status, duration)
            % Log episode summary
            
            ep_idx = length(obj.episode_summary) + 1;
            obj.episode_summary(ep_idx).episode = episode;
            obj.episode_summary(ep_idx).total_reward = total_reward;
            obj.episode_summary(ep_idx).llm_calls = llm_calls;
            obj.episode_summary(ep_idx).convergence_status = convergence_status;
            obj.episode_summary(ep_idx).duration = duration;
        end
        
        function finalize(obj)
            % Save all data and generate reports
            
            try
                % Save iteration data
                iteration_file = fullfile(obj.output_dir, 'iteration_log.mat');
                iteration_data = obj.iteration_data; %#ok<NASGU>
                save(iteration_file, 'iteration_data');
                
                % Save episode summary
                summary_file = fullfile(obj.output_dir, 'episode_summary.mat');
                episode_summary = obj.episode_summary; %#ok<NASGU>
                save(summary_file, 'episode_summary');
                
                % Save LTPP reference
                ltpp_file = fullfile(obj.output_dir, 'ltpp_reference.mat');
                ltpp_reference = obj.ltpp_reference; %#ok<NASGU>
                save(ltpp_file, 'ltpp_reference');
                
                % Generate CSV reports (for analysis in Python/R)
                obj.exportToCSV();
                
                % Generate summary report
                obj.generateSummaryReport();
                
                fprintf('✅ LLM log data saved successfully\n');
                fprintf('   Output directory: %s\n', obj.output_dir);
                
            catch ME
                % ✅ 修正：使用 fprintf 替代 warning
                fprintf(2, '⚠️ Warning: Failed to save log data: %s\n', ME.message);
            end
        end
        
        function exportToCSV(obj)
            % Export data to CSV format for external analysis
            
            try
                % Export iteration data
                if ~isempty(obj.iteration_data)
                    csv_file = fullfile(obj.output_dir, 'iterations.csv');
                    
                    % Create table
                    T = table();
                    T.Episode = [obj.iteration_data.episode]';
                    T.Step = [obj.iteration_data.step]';
                    T.Reward = [obj.iteration_data.reward]';
                    T.ActionType = {obj.iteration_data.action_type}';
                    
                    % Write to CSV
                    writetable(T, csv_file);
                end
                
                % Export episode summary
                if ~isempty(obj.episode_summary)
                    csv_file = fullfile(obj.output_dir, 'episodes.csv');
                    
                    T = table();
                    T.Episode = [obj.episode_summary.episode]';
                    T.TotalReward = [obj.episode_summary.total_reward]';
                    T.LLMCalls = [obj.episode_summary.llm_calls]';
                    T.ConvergenceStatus = {obj.episode_summary.convergence_status}';
                    T.Duration = [obj.episode_summary.duration]';
                    
                    writetable(T, csv_file);
                end
                
            catch ME
                % ✅ 修正：使用 fprintf 替代 warning
                fprintf(2, '⚠️ Warning: Failed to export CSV: %s\n', ME.message);
            end
        end
        
        function generateSummaryReport(obj)
            % Generate text summary report
            
            try
                report_file = fullfile(obj.output_dir, 'summary_report.txt');
                fid = fopen(report_file, 'w');
                
                if fid == -1
                    error('Cannot create report file');
                end
                
                fprintf(fid, '=== LLM-Guided PPO Optimization Summary ===\n\n');
                fprintf(fid, 'Optimization ID: %s\n', obj.optimization_id);
                fprintf(fid, 'Start Time: %s\n', datestr(obj.start_time));
                fprintf(fid, 'Total Iterations: %d\n', length(obj.iteration_data));
                fprintf(fid, 'Total Episodes: %d\n', length(obj.episode_summary));
                
                if ~isempty(obj.episode_summary)
                    rewards = [obj.episode_summary.total_reward];
                    fprintf(fid, '\nReward Statistics:\n');
                    fprintf(fid, '  Mean: %.4f\n', mean(rewards));
                    fprintf(fid, '  Std: %.4f\n', std(rewards));
                    fprintf(fid, '  Max: %.4f\n', max(rewards));
                    fprintf(fid, '  Min: %.4f\n', min(rewards));
                end
                
                % ✅ 新增：LLM 调用统计
                if ~isempty(obj.iteration_data)
                    action_types = {obj.iteration_data.action_type};
                    hybrid_count = sum(strcmp(action_types, 'hybrid'));
                    ppo_only_count = sum(strcmp(action_types, 'ppo_only'));
                    
                    fprintf(fid, '\nLLM Usage Statistics:\n');
                    fprintf(fid, '  Hybrid actions: %d\n', hybrid_count);
                    fprintf(fid, '  PPO-only actions: %d\n', ppo_only_count);
                    fprintf(fid, '  LLM usage rate: %.1f%%\n', ...
                        100 * hybrid_count / (hybrid_count + ppo_only_count));
                end
                
                fprintf(fid, '\n=== End of Report ===\n');
                fclose(fid);
                
            catch ME
                % ✅ 修正：使用 fprintf 替代 warning
                fprintf(2, '⚠️ Warning: Failed to generate report: %s\n', ME.message);
                % 确保文件句柄关闭
                if exist('fid', 'var') && fid ~= -1
                    fclose(fid);
                end
            end
        end
        
        % ✅ 新增：数据验证方法
        function is_valid = validateData(obj)
            % Validate logged data integrity
            is_valid = true;
            
            if isempty(obj.iteration_data)
                fprintf(2, '⚠️ Warning: No iteration data logged\n');
                is_valid = false;
            end
            
            if isempty(obj.episode_summary)
                fprintf(2, '⚠️ Warning: No episode summary logged\n');
                is_valid = false;
            end
            
            if is_valid
                fprintf('✅ Data validation passed\n');
            end
        end
        
        % ✅ 新增：获取统计摘要
        function stats = getStatistics(obj)
            % Get statistical summary of logged data
            stats = struct();
            
            if ~isempty(obj.episode_summary)
                rewards = [obj.episode_summary.total_reward];
                stats.mean_reward = mean(rewards);
                stats.std_reward = std(rewards);
                stats.max_reward = max(rewards);
                stats.min_reward = min(rewards);
                stats.total_episodes = length(obj.episode_summary);
            else
                stats.mean_reward = NaN;
                stats.std_reward = NaN;
                stats.max_reward = NaN;
                stats.min_reward = NaN;
                stats.total_episodes = 0;
            end
            
            if ~isempty(obj.iteration_data)
                stats.total_iterations = length(obj.iteration_data);
                action_types = {obj.iteration_data.action_type};
                stats.hybrid_actions = sum(strcmp(action_types, 'hybrid'));
                stats.ppo_only_actions = sum(strcmp(action_types, 'ppo_only'));
            else
                stats.total_iterations = 0;
                stats.hybrid_actions = 0;
                stats.ppo_only_actions = 0;
            end
        end
    end
end