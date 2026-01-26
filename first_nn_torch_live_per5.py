import torch

INPUT_SHAPE = (3, 224, 224)

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

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Execute the forward pass of the CNN."""
        print(f"{'Input Shape:':>30} {x.shape}")
        y = self.zp1(x)
        print(f"{'After 1st Zero-Padding:':>30} {y.shape}")
        y = self.conv1(y)
        y = self.relu(y)
        print(f"{'After 1st convolution:':>30} {y.shape}")


def main():
    x = torch.rand(INPUT_SHAPE)
    model = Model(INPUT_SHAPE)
    model(x)

if __name__ == "__main__":
    main()