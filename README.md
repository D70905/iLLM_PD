# iLLM-PD: An Autonomous Agent for Road Structure Design Powered by Large Language Model and Reinforcement Learning

**Official implementation code for the manuscript submitted to *Nature Communications*.**

**Author:** Jingyi Xie (Tongji University)  
**Contact:** 2410820@tongji.edu.cn  
**Year:** 2025

---

## 🚀 Overview

iLLM-PD represents a novel framework that integrates Large Language Models (LLMs) with Proximal Policy Optimization (PPO) to automate pavement structure design. By synergizing the semantic reasoning of LLMs with the decision-making capabilities of RL, iLLM-PD achieves multi-objective optimization for safety, economy, and carbon footprint.

**Key Features:**

- **Hybrid Intelligence:** Combines LLM engineering knowledge with RL optimization.
- **Model Agnostic:** Supports both proprietary models (GPT-4, DeepSeek) and open-source models (Llama 3, Qwen 2.5) via local inference.
- **Physics-Informed:** Incorporates Finite Element Analysis (FEA) based on PDE Toolbox to ensure mechanical validity.

---

## 🛠️ Requirements

- **MATLAB**: Version 2024b or later (Required for PDE Toolbox features).
- **Deep Learning Toolbox**: For the PPO agent neural networks.
- **Partial Differential Equation Toolbox**: For pavement mechanical response simulation.
- **(Optional) Ollama**: For running open-source LLMs locally (free & offline).

---

## 📂 Repository Structure

```text
iLLM-PD/
├── core/                             # Core algorithms (PPO agent, FEA environment)
├── utils/                            # Helper functions for data processing
├── tests/                            # Scripts to reproduce paper figures
│   ├── ablation/                     # Ablation study scripts
│   ├── runLLMAccuracyExperiment.m    # LLM parsing accuracy tests 
│   ├── runOpenSourceExperiment.m     # Comparison between open-source models and commercial model
│   └── runSubgradeExperiment.m       # Comparison of two types of soil foundation handling method
├── config.json                       # Configuration file (API keys & Hyperparameters)
├── runCompleteOptimization.m         # Main entry point for design optimization
└── README.md                         # Documentation
```

---

## ⚙️ Configuration

Before running, please go to the websites of each model and apply for the corresponding API key by yourself, and set your parameters:

```json
{
    "llm_api_config": {
        "api_key": "YOUR_API_KEY_HERE",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat"
    },
    "ppo": {
        "max_episodes": 15
    }
}
```

> **Note:** For local models, set `"api_key": "sk-ollama..."`, `"base_url": "http://localhost:11434"`, and `"model": "llama3"` or `"qwen2.5:7b"`.

---

## 🏃‍♂️ Usage (Running the Code)

### Option A: Quick Verification with Open-Source Models (Recommended for Reviewers)

This method allows you to verify the framework using free, local models (Llama 3 or Qwen 2.5) without needing an API key.

1. **Install Ollama:** Download from [ollama.com](https://ollama.com/).

2. **Pull a Model:** Open your terminal/command prompt and run:
   ```bash
   ollama pull llama3
   # OR
   ollama pull qwen2.5:7b
   ```

3. **Run the Script:**
   - Open MATLAB and navigate to the `tests/` folder.
   - Open `runOpenSourceExperiment.m`.
   - Click **Run**. The script is pre-configured to connect to your local Ollama instance (`localhost:11434`).

> **Note:** The `config.json` provided in this repo contains placeholder values. Please update them with your local settings.

### Option B: Full System Optimization

To run the complete pipeline with custom configurations (standards, subgrade models, or commercial APIs):

1. **Configure API:** Update `config.json` with your API key (for commercial models) or local settings.

2. **Modify the Model Selection:** Change the `"active_llm"` field in `config.json` to the model name you need.
   ```json
   {
       "active_llm": "deepseek"
   }
   ```

3. **Run the Script:**
   - Open MATLAB and navigate to the root directory.
   - Open `runCompleteOptimization.m`.
   - Click **Run**.

4. **Customization:**
   - The script uses **JTG D50-2017** (Chinese Standard) and **Multi-layer Elastic System** for the subgrade by default.
   - You can manually modify the input arguments in `runCompleteOptimization.m` to switch to **AASHTO 1993** or the **Winkler Spring Model** as needed.

---

## 📊 Example Data & Results

The repository includes the standard **LTPP validation case (SHRP ID: 1001)** data used in the manuscript.

| Item | Description |
|------|-------------|
| **Input** | Heavy traffic highway, semi-rigid pavement design |
| **Expected Output** | Optimized layer thicknesses and moduli satisfying ME-PDG standards |
| **Performance** | Detailed convergence plots and structural parameters are saved in the `output/` directory after execution |

---

## ⚖️ Code Availability Statement

The source code is available under the MIT License. Commercial LLM APIs (e.g., GPT-4, DeepSeek) require separate subscriptions from their respective providers. Open-source models (Llama 3, Qwen) are available free of charge via Ollama.

---

## 📧 Contact

For any questions regarding the code or paper, please open an issue in this repository or contact the corresponding author.
