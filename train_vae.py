import lightning as pl
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import hydra
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.callbacks import ModelCheckpoint

from models.VAE.CNN_VAE import ConvVAE
from data.dataset import create_dataloaders

class VAETrainer(pl.LightningModule):
    def __init__(self, latent_dim: int, learning_rate: float=1e-3):
        super(VAETrainer, self).__init__()
        self.model = ConvVAE(latent_dim=latent_dim)
        self.learning_rate = learning_rate
        self.save_hyperparameters()
    
    def loss_function(self, recon_x, x, mu, log_var):
        bce_kl, bce, kl = self.model.vae_loss(recon_x, x, mu, log_var)
        return bce_kl
    
    def training_step(self, batch, batch_idx):
        x = batch
        recon_x, mu, log_var = self.model(x)
        loss = self.loss_function(recon_x, x, mu, log_var)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        recon_x, mu, log_var = self.model(x)
        loss = self.loss_function(recon_x, x, mu, log_var)
        self.log("val_loss", loss, prog_bar=True)
        
        if batch_idx == 0:
            n = min(8, x.size(0))
            orig = x[:n]
            recon = recon_x[:n]
            comparison = torch.cat([orig, recon], dim=0)
            grid = vutils.make_grid(comparison, nrow=n, normalize=True, padding=2)
            self.logger.experiment.add_image("Reconstruction", grid, self.global_step)
        
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

@hydra.main(config_path="config", config_name="train_vae", version_base="1.2")
def main(config: DictConfig):
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    print("------ Configuration ------")
    print(OmegaConf.to_yaml(config))
    print("---------------------------")
    
    # データローダーの作成
    train_loader, val_loader = create_dataloaders(data_dir=config.data_dir,
                                                  batch_size=config.batch_size,
                                                  img_size=config.img_size,
                                                  num_workers=config.num_workers)

    
    model = VAETrainer(latent_dim=config.latent_dim, learning_rate=config.learning_rate)
    if config.ckpt_path is not None:
        model = VAETrainer.load_from_checkpoint(config.ckpt_path, latent_dim=config.latent_dim, learning_rate=config.learning_rate)
    
    # チェックポイントの設定 (Top 3 の val_loss のモデルを保存)
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename="{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,  # 上位3つのモデルを保存
        save_last=True,  # 最後のエポックも保存
        verbose=True
    )
    
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=10,
        callbacks=[checkpoint_callback]  # コールバックに追加
    )
    
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
