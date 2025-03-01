import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.Actor.actor import ActorSAC
from models.Critic.critic import Critic

# SACエージェントクラス（latent表現のみを使用）
class SACAgent:
    def __init__(self, state_dim: int, action_dim: int, actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 gamma=0.99, tau=0.005, hidden_dim=256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau

        # Actor, Critic, およびターゲットCriticは latent を入力とする
        self.actor = ActorSAC(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(self.device)
        self.critic = Critic(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 温度パラメータαの設定
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        # ターゲットエントロピーは通常 -action_dim として設定
        self.target_entropy = -action_dim

    def select_action(self, state, evaluate=False):
        """
        エージェントが環境との相互作用時にアクションを選択するための関数
        入力は encoder の出力 z のみを使用
        """
        if evaluate:
            with torch.no_grad():
                # 評価時は平均値（tanh適用済み）を利用
                _, _, action = self.actor.sample(state)
            return action.cpu().numpy()[0]
        else:
            with torch.no_grad():
                action, _, _ = self.actor.sample(state)
            return action.cpu().numpy()[0]

    def update(self, buffer, batch_size=64):
        """
        リプレイバッファからサンプルを取得し、ネットワークの更新を行う
        バッファは "state", "action", "reward", "next_state", "done" を格納している前提
        """
        sample = buffer.sample(batch_size)
        state = torch.FloatTensor(sample["state"]).to(self.device)
        action = torch.FloatTensor(sample["action"]).to(self.device)
        reward = torch.FloatTensor(sample["reward"]).to(self.device)
        next_state = torch.FloatTensor(sample["next_state"]).to(self.device)
        done = torch.FloatTensor(sample["done"]).to(self.device)

        # ターゲットQ値の計算
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - torch.exp(self.log_alpha) * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q

        # 現在のQ値の計算
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actorの更新
        action_new, log_prob, _ = self.actor.sample(state)
        q1_new, q2_new = self.critic(state, action_new)
        actor_loss = (torch.exp(self.log_alpha) * log_prob - torch.min(q1_new, q2_new)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 温度パラメータαの更新
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ターゲットネットワークのソフト更新
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": torch.exp(self.log_alpha).item()
        }
    
    def save(self, filepath, episode=None):
        """
        モデルのチェックポイントを保存する
        """
        checkpoint = {
            "episode": episode,
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha,
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load(self, filepath):
        """
        保存されたチェックポイントをロードする
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
        self.log_alpha = checkpoint["log_alpha"].to(self.device)

        print(f"Checkpoint loaded from {filepath}")
