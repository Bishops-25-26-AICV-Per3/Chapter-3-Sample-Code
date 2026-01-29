import pathlib

import torch
import cv2

INPUT_SHAPE = (3, 224, 224)
IMAGES_PATH = pathlib.Path("../animals")
BATCH_SIZE = 32

class Dataset(torch.utils.data.Dataset):
    """Represent the dataset as an object."""
    def __init__(self, image_path: pathlib.Path):
        """Provide a path where the images are, 1 folder for each category.
        
        This assumes that train and validation are mingled."""
        print("Loading images...")
        self.class_names = []
        self.images = []
        for path in image_path.iterdir():
            if path.is_dir():
                self.class_names.append(path.name)
                for image_file in path.iterdir():
                    img = cv2.imread(image_file)
                    if img is not None:
                        label = torch.tensor(
                            self.class_names.index(path.name),
                            dtype = torch.float32,
                            device = 'mps', # Or 'cuda' or leave this out
                        )
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, INPUT_SHAPE[1:])
                        img = img / 255.0
                        img = img.transpose([2, 0, 1]) # Move channels to first
                        img = torch.tensor(
                            img,
                            dtype = torch.float32,
                            device = 'mps', # Or 'cuda' or leave this out
                        )
                        self.images.append((img, label))
        print("Images loaded.")

    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
            return self.images[idx]

class Model(torch.nn.Module):
    """Represent my CNN as an object."""
    def __init__(self, input_shape: (int, int, int)):
        """input_shape is expected to be channels first."""
        super().__init__()
        self.zp1 = torch.nn.ZeroPad2d((2, 1, 2, 1))
        self.conv1 = torch.nn.Conv2d(
            in_channels = 3,
            out_channels = 48,
            kernel_size = 11,
            stride = 4,
        )
        self.relu = torch.nn.ReLU()
        self.zp2 = torch.nn.ZeroPad2d(2)
        self.conv2 = torch.nn.Conv2d(
            in_channels = 48,
            out_channels = 128,
            kernel_size = 5,
            stride = 1,
        )
        self.maxpool = torch.nn.MaxPool2d(
            kernel_size = 3,
            stride = 2,
        )

        ... # More convolution blocks here

        self.flatten = torch.nn.Flatten()
        # Linear layer also called Fully-Connected or Dense layer
        # This section is also called MLP = Multi-Layer Perceptron
        self.linear1 = torch.nn.Linear(
            in_features = 0,
            out_features = 0,
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass of the CNN."""
        print(f"{'Input Shape:':>30}", x.shape)
        y = self.zp1(x)
        print(f"{'After 1st Zero-Pad':>30}", y.shape)
        y = self.relu(self.conv1(y))
        print(f"{'After 1st Convolution:':>30}", y.shape)
        y = self.zp2(y)
        print(f"{'After 2nd Zero-Pad':>30}", y.shape)
        y = self.relu(self.conv2(y))
        print(f"{'After 2nd Convolution:':>30}", y.shape)
        y = self.maxpool(y)
        print(f"{'After 1st MaxPool:':>30}", y.shape)
        y = self.flatten(y)
        print(f"{'After Flatten:':>30}", y.shape)
        return y


def get_dataloaders(dataset: Dataset, train_prop: float, batch_size: int
        ) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    """Split the dataset and prepare for training"""
    generator = torch.Generator().manual_seed(37)
    train_set, validation_set = torch.utils.data.random_split(
            dataset,
            lengths = [train_prop, 1-train_prop],
            generator = generator,
    )
    train = torch.utils.data.DataLoader(train_set, batch_size = batch_size)
    validation = torch.utils.data.DataLoader(validation_set, 
            batch_size = batch_size)
    return train, validation

def main():
    dataset = Dataset(IMAGES_PATH)
    print(f"Found {len(dataset)} images.")
    train, validation = get_dataloaders(dataset, 0.8, BATCH_SIZE)
    model = Model(INPUT_SHAPE).to('mps')
    x = next(iter(train))[0]
    model(x)

if __name__ == "__main__":
    main()


