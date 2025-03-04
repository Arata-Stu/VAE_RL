import torch
import torch.nn as nn
import torch.nn.functional as F
# from utils.timers import CudaTimer as Timer
from utils.timers import TimerDummy as Timer

class ConvVAE(nn.Module):
    
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_size = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        ## input img 3*64*64
        # encoder
        self.enc_conv1 = nn.Conv2d(3,32,kernel_size=4,stride=2, padding=0) # 3*64*64 -> 32*31*31
        self.enc_conv2 = nn.Conv2d(32,64,kernel_size=4,stride=2, padding=0) # 32*31*31 -> 64*14*14
        self.enc_conv3 = nn.Conv2d(64,128,kernel_size=4,stride=2, padding=0) # 64*14*14 -> 128*6*6
        self.enc_conv4 = nn.Conv2d(128,256,kernel_size=4,stride=2, padding=0) # 128*6*6 -> 256*2*2
        
        # z
        self.mu = nn.Linear(1024, latent_dim)
        self.logvar = nn.Linear(1024, latent_dim)
        
        # decoder
        self.dec_conv1 = nn.ConvTranspose2d(latent_dim, 128, kernel_size=5, stride=2, padding=0)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=0)
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2, padding=0)
        self.dec_conv4 = nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2, padding=0)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.latent(mu, logvar)
        out = self.decode(z)
        
        return out, mu, logvar   
        
    def encode(self, x):
        with Timer("encode"):
            batch_size = x.shape[0]
            
            out = F.relu(self.enc_conv1(x))
            out = F.relu(self.enc_conv2(out))
            out = F.relu(self.enc_conv3(out))
            out = F.relu(self.enc_conv4(out))
            out = out.view(batch_size,1024)
            
            mu = self.mu(out)
            logvar = self.logvar(out)
            
        return mu, logvar
        
    def decode(self, z):
        with Timer("decode"):
            batch_size = z.shape[0]
            
            out = z.view(batch_size, self.latent_size, 1, 1)
            out = F.relu(self.dec_conv1(out))
            out = F.relu(self.dec_conv2(out))
            out = F.relu(self.dec_conv3(out))
            out = torch.sigmoid(self.dec_conv4(out))
            
        return out
        
    def latent(self, mu, logvar):
        with Timer("latent"):
            sigma = torch.exp(0.5*logvar)
            eps = torch.randn_like(logvar).to(self.device)
            z = mu + eps * sigma
        return z
    
    def obs_to_z(self, x):
        mu, logvar = self.encode(x)
        z = self.latent(mu, logvar)
        return z

    def sample(self, z):
        out = self.decode(z)
        return out
    
    def vae_loss(self, out, y, mu, logvar):
        BCE = F.binary_cross_entropy(out, y, reduction="sum")
        KL = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KL, BCE, KL

    def get_latent_size(self):
        return self.latent_size

    def set_device(self, device):
        self.device = device
        self.to(device)  # モデル全体を指定デバイスに移動
    
    def load_ckpt(self, ckpt_path: str):
        """
        PyTorch Lightning のチェックポイントからモデルの重みをロードする。

        - `ckpt['state_dict']` に含まれる "model." のプレフィックスを削除して適用。
        - デバイスに適応。

        Args:
            ckpt_path (str): チェックポイントのファイルパス
        """
        print(f"Loading checkpoint from {ckpt_path}")
        
        try:
            # チェックポイントをロード
            ckpt = torch.load(ckpt_path, map_location=self.device)

            # "model." プレフィックスを削除
            state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
            
            # モデルの state_dict を更新
            self.load_state_dict(state_dict)

            # デバイス設定
            self.to(self.device)

            print(f"Checkpoint successfully loaded from {ckpt_path}")
        
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
