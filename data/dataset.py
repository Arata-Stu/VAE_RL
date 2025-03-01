import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class CarRacingDataset(Dataset):
    def __init__(self, data_dir, img_size=(64, 64), augment=True):
        """
        :param data_dir: 画像が保存されているディレクトリ
        :param img_size: 画像のリサイズサイズ（デフォルト: 64x64）
        :param augment: データ拡張を適用するか (バリデーションではFalse)
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.image_paths = sorted(glob.glob(os.path.join(data_dir, "*.png")))  # PNG画像をすべて取得

        # 画像前処理（データ拡張は学習用データのみ適用）
        transform_list = [
            transforms.Resize(img_size),
            transforms.ToTensor()
        ]

        if augment:
            transform_list.insert(1, transforms.RandomHorizontalFlip())
            transform_list.insert(2, transforms.RandomRotation(10))

        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # 画像をRGB形式で読み込む
        return self.transform(image)  # 正規化した Tensor を返す


def create_dataloaders(data_dir, batch_size=128, img_size=(64, 64), num_workers=4, val_split=0.2):
    """
    学習用とバリデーション用のデータローダーを作成する関数
    """
    dataset = CarRacingDataset(data_dir, img_size, augment=True)

    # データセットを `train` と `val` に分割
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # バリデーションデータにはデータ拡張を適用しない
    val_dataset.dataset = CarRacingDataset(data_dir, img_size, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader
