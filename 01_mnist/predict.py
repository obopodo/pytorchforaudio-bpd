import matplotlib.pyplot as plt
import torch
from torch import nn
from train import FeedForwardNet, load_mnist_data

SAMPLE_INDEX = 1  # Index of the sample to predict
SHOW_PICTURE = True
class_mapping = [str(i) for i in range(10)]


def plot_sample(sample: torch.Tensor, target: int, pred: str):
    plt.imshow(sample.squeeze(), cmap="gray")
    plt.title(f"Expected: {target}, Predicted: {pred}")
    plt.axis("off")
    plt.show()


def predict(model: nn.Module, sample: torch.Tensor, target: int, class_mapping: list) -> tuple:
    model.eval()
    with torch.no_grad():
        preds: torch.Tensor = model(sample)
        # preds is Tensor(n_samples, n_classes) of probabilities -> [[0.1, 0.9, ..., 0.001]] 1st dim == 1 in this case
        predicted_class = preds.argmax(dim=1).item()
        expected_class = target

    if SHOW_PICTURE:
        plot_sample(sample, expected_class, class_mapping[predicted_class])

    return class_mapping[predicted_class], class_mapping[expected_class]


if __name__ == "__main__":
    model = FeedForwardNet()
    state_dict = torch.load("feed_forward_net.pth")
    model.load_state_dict(state_dict)

    _, test_data = load_mnist_data(which="test")
    # test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=1)
    sample, target = test_data[SAMPLE_INDEX][0], test_data[SAMPLE_INDEX][1]

    predicted, expected = predict(model, sample, target, class_mapping)
    print(f"Predicted: {predicted}, Expected: {expected}")
