import torch
from train import FeedForwardNet, load_mnist_data

if __name__ == "__main__":
    model = FeedForwardNet()
    state_dict = torch.load("feed_forward_net.pth")
    model.load_state_dict(state_dict)
    model.eval()

    _, test_data = load_mnist_data()
