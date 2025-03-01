from omegaconf import DictConfig
from .sac import SACAgent

def get_agents(agent_cfg: DictConfig, state_dim: int, action_dim: int, device: str = 'cpu'):
    if agent_cfg.name == "SAC":
        return SACAgent(state_dim=state_dim,
                        action_dim=action_dim,
                        actor_lr=agent_cfg.actor_lr,
                        critic_lr=agent_cfg.critic_lr,
                        alpha_lr=agent_cfg.alpha_lr,
                        gamma=agent_cfg.gamma,
                        tau=agent_cfg.tau,
                        hidden_dim=agent_cfg.hidden_dim)
    
    else:
        raise NotImplementedError(f"Unknown agent: {agent_cfg.name}")