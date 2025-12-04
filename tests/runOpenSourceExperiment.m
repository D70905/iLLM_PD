% runOpenSourceExperiment_Fixed.m
% 修正版：解决了 URL 重复导致的 404 错误，并支持切换 Qwen
% 目标：响应 Nature Communications 编辑要求，验证开源模型

clear; clc; close all;

% 添加核心路径
addpath(genpath(pwd));

%% 1. 基础参数定义 (保持与论文一致的 LTPP Case)
design_params = struct();
design_params.thickness = [15; 25; 20; 150]; % cm
design_params.modulus = [2500; 1200; 400; 50]; % MPa
design_params.poisson = [0.35; 0.40; 0.40; 0.45];
design_params.pavement_type = 'semi_rigid';
design_params.traffic_level = 'heavy';

% 获取设计规范
design_criteria = getJTG50DesignCriteria('Heavy traffic highway design', design_params);

% 初始 PDE 计算
load_params = struct('load_pressure', 0.7, 'load_radius', 21.3);
bc = struct('method', 'multilayer_subgrade', 'soil_modulus', 50);
initial_pde = roadPDEModelingSimplified(design_params, load_params, bc);

%% 2. 定义实验配置：选择开源模型

config_opensource = getDefaultOptimizedConfig();

% =======================================================
% 【模型选择开关】请在这里修改 model 名称
% =======================================================

% 选项 A: 使用 Llama 3 (Meta) - 需先运行 'ollama pull llama3'
% model_name = 'llama3'; 
% exp_name = 'OpenSource_Llama3';

% 选项 B: 使用 Qwen 2.5 (Alibaba) - 需先运行 'ollama pull qwen2.5:7b'
model_name = 'qwen2.5:7b'; 
exp_name = 'OpenSource_Qwen2.5';

config_opensource.experiment_name = exp_name;

% =======================================================
% 【关键修复】Ollama API 配置
% =======================================================
% 1. API Key: 构造一个假的 sk- 开头且足够长的 Key，绕过代码校验
config_opensource.llm_api_config.api_key = 'sk-ollama-local-host-dummy-key-for-nc-test'; 

% 2. Base URL: 【重要】去掉末尾的 /v1，避免 RoadStructurePPO 拼接出双重 v1/v1
config_opensource.llm_api_config.base_url = 'http://localhost:11434'; 

% 3. 模型名称
config_opensource.llm_api_config.model = model_name;

% 4. 其他参数
config_opensource.llm_api_config.max_tokens = 500;
config_opensource.timeout_seconds = 60; % 本地推理需给予更多时间

%% 3. 执行实验

fprintf('================================================\n');
fprintf('   Starting Open-Source Model Verification\n');
fprintf('   Target Model: %s\n', model_name);
fprintf('================================================\n');

try
    % 调用核心优化函数
    [opt_params_open, log_open] = runPPOOptimization(...
        design_params, config_opensource, design_criteria, initial_pde);
    
    % 保存数据
    save(['results_' exp_name '.mat'], 'opt_params_open', 'log_open');
    fprintf('✅ Experiment completed successfully.\n');
    
    % 检查是否真的调用了 LLM (检查日志中的调用次数)
    if isfield(log_open, 'price_llm_details')
        total_calls = sum(log_open.price_llm_details) + sum(log_open.engineering_llm_details);
        fprintf('📊 Total LLM API Calls Made: %d\n', total_calls);
        if total_calls == 0
            fprintf('⚠️ Warning: No successful API calls recorded. Check Ollama status.\n');
        else
            fprintf('🎉 Success! Real interactions with %s confirmed.\n', model_name);
        end
    end
    
catch ME
    fprintf('❌ Experiment failed. \nError: %s\n', ME.message);
end

%% 4. 结果可视化 (生成回复编辑的图)

figure('Position', [100, 100, 800, 600], 'Color', 'w');
hold on; grid on;

% 绘制奖励曲线
if exist('log_open', 'var') && ~isempty(log_open.episode_rewards)
    plot(log_open.episode_rewards, 'o-', 'LineWidth', 2, 'Color', '#0072BD', ...
        'DisplayName', sprintf('iLLM-PD (Powered by %s)', model_name));
    
    xlabel('Training Episodes', 'FontSize', 12);
    ylabel('Total Reward', 'FontSize', 12);
    title(['Performance Consistency: ' model_name ' (Local Inference)'], 'FontSize', 14);
    legend('Location', 'southeast');
    
    % 添加水印证明是本地运行
    text(1, min(log_open.episode_rewards), ...
        sprintf('Local Inference via Ollama\nAPI: localhost:11434'), ...
        'FontSize', 10, 'Color', [0.5 0.5 0.5]);
end

saveas(gcf, ['Response_to_Editor_' exp_name '_Validation.png']);
fprintf('\n📊 Validation plot generated: Response_to_Editor_%s_Validation.png\n', exp_name);

%% 辅助函数：配置结构体
function config = getDefaultOptimizedConfig()
    config = struct();
    config.ablation_mode = 'full_system'; 
    config.max_training_episodes = 10;
    
    config.ppo = struct();
    config.ppo.max_episodes = 10;
    config.ppo.max_steps_per_episode = 6;
    config.ppo.learning_rate = 0.003;
    config.ppo.batch_size = 32;
    
    config.deepseek = struct(); 
    config.deepseek.guidance_enabled = true;
    
    config.llm_api_config = struct(); 
end