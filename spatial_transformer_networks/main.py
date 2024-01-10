import argparse
import numpy as np
import matplotlib.pyplot as plt

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Loading the data
train_loader = torch.utils.data.DataLoader(datasets.MNIST(root="../data", train=True, download=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.381,))
                                                              ])),
                                           batch_size=64, shuffle=True, num_workers=4)

test_loader = torch.utils.data.DataLoader(datasets.MNIST(root="../data", train=False, download=True,
                                                         transform=transforms.Compose([
                                                             transforms.ToTensor(),
                                                             transforms.Normalize((0.1307,), (0.381,))
                                                         ])),
                                          batch_size=64, shuffle=True, num_workers=4)
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # NOTE(Abid): Localization network.
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),  # [b, 1, 28, 28] -> [b, 8, 22, 22]
            nn.MaxPool2d(2, stride=2),       # [b, 8, 22, 22] -> [b, 8, 11, 11]
            nn.ReLU(True),                   # [b, 8, 11, 11] -> [b, 8, 11, 11]
            nn.Conv2d(8, 10, kernel_size=5), # [b, 8, 11, 11] -> [b, 10, 7,  7]
            nn.MaxPool2d(2, stride=2),       # [b, 10, 7,  7] -> [b, 10, 3,  3]
            nn.ReLU(True)                    # [b, 10, 3,  3] -> [b, 10, 3,  3]
        )

        # NOTE(Abid): Affine 2D transformation regressor.
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32), # [b, 90] -> [b, 32]
            nn.ReLU(True),             # [b, 32] -> [b, 32]
            nn.Linear(32, 3 * 2)       # [b, 32] -> [b,  6]
        )

        # NOTE(Abid): Setting the weights and biases to identity transform.
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # NOTE(Abid): Spatial Transformer.
    def stn(self, x: Tensor) -> Tensor:
        xs = self.localization(x)              # [b, 1, 28, 28] -> [b, 10,  3,  3]
        xs = xs.view(-1, 10 * 3 * 3)           # [b, 1,  3,  3] -> [b, 10 * 3 * 3]
        theta = self.fc_loc(xs)                # [b, 90]        -> [b, 6]
        theta = theta.view(-1, 2, 3)           # [b,  6]        -> [b, 2, 3]

        grid = F.affine_grid(theta, x.size())  # [b, 28, 28, 2]
        x = F.grid_sample(x, grid)             # [b, 1, 28, 28]

        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.stn(x)

        x = F.relu(F.max_pool2d(self.conv1(x), 2))                  # [b,  1, 28, 28] -> [b, 10, 24, 24] -> [b, 10, 12, 12]
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)) # [b, 10, 12, 12] -> [b, 20,  8,  8] -> [b, 20,  4,  4]
        x = x.view(-1, 20 * 4 * 4)                                  # [b, 20,  4,  4] -> [b, 20 * 4 * 4]
        x = F.relu(self.fc1(x))                                     # [b, 320]        -> [b, 50]
        x = F.dropout(x, training=self.training)                    # [b, 50]         -> [b, 50]
        x = self.fc2(x)                                             # [b, 50]         -> [b, 10]
        return F.log_softmax(x, dim=1)                              # [b, 10]         -> [b, 10]

def train(epoch: int) -> None:
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 500 == 0:
            print(f"Train epoch: {epoch} [{ batch_idx * len(data) }/{ len(train_loader.dataset) } ({(100. *batch_idx / len(train_loader)):.0f})]\tLoss: {loss.item():.6f}")


def test() -> None:
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += F.nll_loss(output, target, reduction="sum").item()

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print(f"Test set: Avg. loss: {test_loss:.4f}, Accu: {correct}/{len(test_loader.dataset)}, {(100. * correct / len(test_loader.dataset)):.4f}%")

def convert_to_image_np(inputs: Tensor) -> Tensor:
    inputs = inputs.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inputs = std * inputs + mean
    inputs = np.clip(inputs, 0, 1)

    return inputs

def visualize_stn() -> None:
    with torch.no_grad():
        data = next(iter(test_loader))[0].to(device)

        input_ten = data.cpu()
        trans_input_ten = model.stn(data).cpu()

        ori_grid = convert_to_image_np(torchvision.utils.make_grid(input_ten))

        trans_grid = convert_to_image_np(torchvision.utils.make_grid(trans_input_ten))

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(ori_grid)
        axes[0].set_title("Original images")

        axes[1].imshow(trans_grid)
        axes[1].set_title("Transformed images")

if __name__ == "__main__":
    plt.ion()

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train and save the trained model")
    should_train = parser.parse_args().train

    model = Net()
    model = model.to(device)
    if should_train: 

        optimizer = optim.SGD(model.parameters(), lr=0.01)
        for epoch in range(1, 16):
            train(epoch)
            test()

        torch.save(model.state_dict(), "stn.pth")
    else: model.load_state_dict(torch.load("stn.pth"))

    visualize_stn()

    plt.ioff()
    plt.show()
