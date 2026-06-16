import torch
import torch.nn as nn
 

class ResBlock(nn.Module):
    """Basic residual block: two 3x3 convs with a skip connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int =1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # If shape changes (stride != 1 or channel count differs), project the skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias = False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)
    
class BaselineCNN(nn.Module):
    """
    ResNet-style CNN trained from random initialisation.
    Architecture follows the README plan:
    Conv(3->64) -> BN -> ReLU -> MaxPool
    ResBlock(64->64)   x2
    ResBlock(64->128, stride=2)  x2
    ResBlock(128->256, stride=2) x2
    GlobalAvgPool -> FC(256->num_classes)
    """

    def __init__(self, num_classes: int = 21):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding=3, bias =False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64),
        )
        self.layer2 = nn.Sequential(
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128),
        )
        self.layer3 = nn.Sequential(
            ResBlock(128, 256, stride=2),
            ResBlock(256,256),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
    
if __name__ == "__main__":
    #Sanity check: does the forward run produce the right output shape?
    model = BaselineCNN(num_classes=21)
    dummy_input = torch.randn(4, 3, 224, 224) #Batch of 4 RGB images, 224x224
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") #Expected [4, 21]

    num_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: {num_params:,}")

'''num_classes = 21 corresponding to 20 breeds and 1 cofounder slot
might swap for softmax threshholding '''