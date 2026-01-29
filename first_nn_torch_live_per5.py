import pathlib

import torch
import cv2

INPUT_SHAPE = (3, 224, 224)
IMAGE_PATH = "../animals"

class Dataset(torch.utils.data.Dataset):
    """Represent the dataset as an object"""
    def __init__(self, image_path: pathlib.Path):
        self.class_names = []
        self.images = []
        for folder in image_path.iterdir():
            if folder.is_dir():
                self.class_names.append(folder.name)
                for img in folder.iterdir():
                    img = cv2.imread(img)
                    if img is not None:
                        img = cv2.resize(img, INPUT_SHAPE[1:])
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = img / 255.0
                        # Move channels first
                        img = img.transpose([2, 0, 1])
                        img = torch.tensor(
                            img,
                            dtype=torch.float32,
                            device='mps', # or 'cuda' or ignore it
                        )
                        label = torch.tensor(
                            self.class_names.index(folder),
                            dtype=torch.float32,
                            device='mps', # or 'cuda'  or ignore it
                        )
                        self.images.append((img, label))


class Model(torch.nn.Module):
    """Represent the CNN as an object."""
    def __init__(self, input_shape: (int, int, int)):
        """input_shape expects channels first."""
        super().__init__()
        self.zp1 = torch.nn.ZeroPad2d((2, 1, 2, 1))
        self.conv1 = torch.nn.Conv2d(
            in_channels = 3, 
            out_channels = 48, 
            kernel_size = 11,
            stride = 4,
        )
        self.relu = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        # Linear layers aka Dense or Fully-Connected layers
        # This section is known as MLP = Multi-Layer Perceptron
        self.linear1 = torch.nn.Linear(
            in_features = 145200, # Waaay too big
            out_features = 2048,
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Execute the forward pass of the CNN."""
        print(f"{'Input Shape:':>30} {x.shape}")
        y = self.zp1(x)
        print(f"{'After 1st Zero-Padding:':>30} {y.shape}")
        y = self.conv1(y)
        y = self.relu(y)
        print(f"{'After 1st convolution:':>30} {y.shape}")
        y = self.flatten(y)
        print(f"{'After Flatten:':>30} {y.shape}")
        y = self.relu(self.linear1(y))
        print(f"{'After 1st Linear Layer:':>30} {y.shape}")


def main():
    dataset = Dataset(pathlib.Path(IMAGE_PATH))

    # x = torch.rand(INPUT_SHAPE)
    # x = torch.unsqueeze(x, 0)
    # model = Model(INPUT_SHAPE)
    # model(x)

if __name__ == "__main__":
    main()