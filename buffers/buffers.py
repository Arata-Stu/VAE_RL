from omegaconf import DictConfig
from .off_policy_buffer import OffPolicyEncoderBuffer

def get_buffers(buffer_cfg: DictConfig, state_dim: int, action_dim: int):
    if buffer_cfg.type == "off_policy":
        return OffPolicyEncoderBuffer(size=int(buffer_cfg.size),
                                      state_dim=state_dim,
                                      action_dim=action_dim,
                                      n_step=buffer_cfg.n_step,
                                      gamma=buffer_cfg.gamma)
    else:
        raise ValueError(f"Unexpected buffer type: {buffer_cfg.type}")