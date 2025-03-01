import numpy as np
from collections import deque

class OffPolicyEncoderBuffer:
    def __init__(self, size, state_dim=64, action_dim=2, n_step=3, gamma=0.99):
        self.size = size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_step = n_step
        self.gamma = gamma
        self.temp_buffer = deque(maxlen=n_step)  # dequeを使用

        self.buffer = {
            "state": np.zeros((size, state_dim), dtype=np.float32),  
            "action": np.zeros((size, action_dim), dtype=np.float32),
            "reward": np.zeros((size, 1), dtype=np.float32),
            "next_state": np.zeros((size, state_dim), dtype=np.float32),
            "done": np.zeros((size, 1), dtype=np.bool_)
        }
        self.position = 0
        self.full = False

    def add(self, state, action, reward, next_state, done):
        """
        N-step のために一時バッファにデータを保存し、n_stepに到達したらリプレイバッファに追加
        """
        self.temp_buffer.append((state, action, reward, next_state, done))

        if len(self.temp_buffer) >= self.n_step:
            self._store_n_step_transition()
        
        if done:  # エピソード終了時は残りのデータも登録
            while self.temp_buffer:
                self._store_n_step_transition()

    def _store_n_step_transition(self):
        state, action, _, _, _ = self.temp_buffer[0]  # 初期ステップ
        reward = 0
        discount = 1
        next_state, done = self.temp_buffer[-1][4:]  # 最後のステップの遷移先
        
        for _, _, r, _, d in self.temp_buffer:
            reward += discount * r
            discount *= self.gamma
            if d:
                break


        idx = self.position
        self.buffer["state"][idx] = state
        self.buffer["action"][idx] = action
        self.buffer["reward"][idx] = reward
        self.buffer["next_state"][idx] = next_state
        self.buffer["done"][idx] = done

        self.position = (self.position + 1) % self.size
        if self.position == 0:
            self.full = True

        self.temp_buffer.popleft()  # 先頭要素を削除 (deque の popleft で高速化)

    def sample(self, batch_size):
        """
        バッチサンプリング
        """
        max_idx = self.size if self.full else self.position
        indices = np.random.choice(max_idx, batch_size, replace=False)
        return {key: self.buffer[key][indices] for key in self.buffer}

    def __len__(self):
        return self.size if self.full else self.position
