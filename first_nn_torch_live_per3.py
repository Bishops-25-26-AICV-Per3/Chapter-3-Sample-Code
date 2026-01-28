import torch

INPUT_SHAPE = (3, 224, 224)

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
            in_features = ,
            out_features = ,
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



def main():
    model = Model(INPUT_SHAPE)
    x = torch.rand(INPUT_SHAPE)
    x = torch.unsqueeze(x, 0)
    y = model(x)

if __name__ == "__main__":
    main()


