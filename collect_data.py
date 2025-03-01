import os
import cv2
import numpy as np
import pygame
import torch
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from omegaconf import DictConfig, OmegaConf
import hydra

from envs.car_racing import CarRacingWithInfoWrapper
from utils.timers import Timer as Timer
# from utils.timers import TimerDummy as Timer


@hydra.main(config_path='config', config_name='collect_data', version_base='1.2')
def main(config: DictConfig):
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    print('------ Configuration ------')
    print(OmegaConf.to_yaml(config))
    print('---------------------------')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 環境のセットアップ
    width, height = config.envs.img_size
    env = gym.make('CarRacing-v3', render_mode="human")  # 画面表示を有効化
    env = TimeLimit(env, max_episode_steps=config.max_steps)
    env = CarRacingWithInfoWrapper(env, width=width, height=height)

    # pygame の初期化
    pygame.init()
    screen = pygame.display.set_mode((400, 300))  # 小さいウィンドウを作成
    pygame.display.set_caption("CarRacing Manual Control")
    clock = pygame.time.Clock()

    # 画像保存用のディレクトリ
    save_dir = config.save_dir
    os.makedirs(save_dir, exist_ok=True)

    max_episodes = config.max_episodes
    max_steps = config.max_steps

    episode_count = 0
    while episode_count < max_episodes:
        obs, info = env.reset()
        step = 0
        episode_reward = 0
        done = False

        while not done and step < max_steps:
            # 画面をクリアして更新（pygame の描画が止まらないように）
            screen.fill((0, 0, 0))
            # pygame.display.flip()

            # 長押しに対応した操作方式
            keys = pygame.key.get_pressed()
            action = np.array([0.0, 0.0, 0.0])  # [steer, gas, brake]

            # ハンドルの感度調整（滑らかにするための微調整）
            if keys[pygame.K_LEFT]:
                action[0] = -0.3  # 左へハンドル
            if keys[pygame.K_RIGHT]:
                action[0] = 0.3   # 右へハンドル
            
            # アクセルとブレーキの調整
            if keys[pygame.K_UP]:
                action[1] = 0.3  # アクセル
            if keys[pygame.K_DOWN]:
                action[2] = 0.3  # ブレーキ

            # 終了イベントの処理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    env.close()
                    return

            # 環境をステップ実行
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            # 画像を保存
            if isinstance(obs, dict):
                obs = obs["image"]  # CarRacingWithInfoWrapper の場合

            if isinstance(obs, np.ndarray):
                img_path = os.path.join(save_dir, f"ep{episode_count:03d}_step{step:04d}.png")
                cv2.imwrite(img_path, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
            else:
                print(f"Warning: obs is not a valid numpy array. Type: {type(obs)}")

            step += 1
            clock.tick(30)  # 30 FPS に制限

        print(f"Episode {episode_count + 1}/{max_episodes} finished. Total reward: {episode_reward:.2f}")
        episode_count += 1

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
