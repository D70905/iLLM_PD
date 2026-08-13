"""Debug: check if delivered_cost_cny is in env info after episode."""
from stable_baselines3 import PPO
from rl.env import PavementEnv, EnvConfig

CKPT = 'experiments/ltpp_data/deliverables/rl_models/B_semi_rigid_1024ts/checkpoints/ckpt_final_step_001024/ppo_model.zip'
model = PPO.load(CKPT)

cfg = EnvConfig(
    pavement_type='semi_rigid',
    E_subgrade=59, max_episode_steps=20,
    llm_enabled=False, fea_verbose=False, fea_keep_runs=False,
)
env = PavementEnv(cfg)
obs, info = env.reset()
for _ in range(20):
    a, _ = model.predict(obs, deterministic=True)
    obs, r, done, tr, info = env.step(a)
env.close()

print('delivered_design in info:', 'delivered_design' in info)
print('delivered_cost_cny:', info.get('delivered_cost_cny', 'NOT FOUND'))
print('delivered_dsr:', info.get('delivered_dsr', 'NOT FOUND'))
print('DSR:', info.get('dsr'))
print('SCR:', info.get('scr_running'))
print('Best cost tracked:', env._best_cost if env._best_cost != float('inf') else 'inf (never set)')
print('Best design tracked:', env._best_design is not None)