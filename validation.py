from omegaconf import DictConfig, OmegaConf
import hydra
import torch
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import os

from envs.car_racing import CarRacingWithInfoWrapper
from agents.agents import get_agents
from models.VAE.CNN_VAE import ConvVAE
from utils.helppers import numpy2img_tensor

@hydra.main(config_path='config', config_name='val', version_base='1.2')
def main(config: DictConfig):
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    print('------ Configuration ------')
    print(OmegaConf.to_yaml(config))
    print('---------------------------')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    width, height = config.envs.img_size
    env = gym.make('CarRacing-v3', render_mode='human')
    env = CarRacingWithInfoWrapper(env, width=width, height=height)

    state_dim = 64
    action_dim = 3
    agent = get_agents(agent_cfg=config.agent, state_dim=state_dim, action_dim=action_dim, device=device)
    agent.load(config.agent.ckpt_path)

    vae = ConvVAE(latent_dim=64).to(device).eval()
    vae.load_ckpt(config.encoder.ckpt_path)

    writer = SummaryWriter(log_dir=config.get("log_dir", "./runs_evaluate"))
    obs, info = env.reset()

    max_episodes = config.max_episodes
    max_steps = config.max_steps

    episode_rewards = []
    
    log_interval = config.get("reconstructed_log_interval", 50)

    for episode in range(max_episodes):
        obs, vehicle_info = env.reset()
        episode_reward = 0

        print(f"Evaluation Episode {episode} started.")
        for step in range(max_steps):
            obs_img = obs["image"].copy()
            obs_img = numpy2img_tensor(obs_img).unsqueeze(0).to(device)
            state = vae.obs_to_z(obs_img)

            action = agent.select_action(state=state, evaluate=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            env.render()

            if step % log_interval == 0:
                reconstucted_img = vae.sample(state).squeeze(0).float().cpu().detach()
                global_step = episode * max_steps + step
                writer.add_image("Reconstructed/Eval", reconstucted_img, global_step)
                writer.add_histogram("Action/Eval_Distribution", action, global_step)

            if terminated or truncated:
                print(f"Evaluation Episode {episode}: Step {step} terminated.")
                break

            obs = next_obs

        episode_rewards.append(episode_reward)
        writer.add_scalar("Reward/Eval_Episode", episode_reward, episode)
        print(f"Evaluation Episode {episode}: Reward = {episode_reward:.2f}")

    writer.close()
    env.close()
    print("Evaluation completed.")

if __name__ == '__main__':
    main()
