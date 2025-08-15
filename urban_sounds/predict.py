from pathlib import Path

import numpy as np
import torch

from urban_sounds.cnn import CNNNet
from urban_sounds.dataset import UrbanSoundDataset
from urban_sounds.preprocess import TransformPipeline


def predict(model: CNNNet, sample: torch.Tensor, target: int) -> tuple:
    model.eval()
    with torch.no_grad():
        preds: torch.Tensor = model.softmax(model(sample))
        # preds is Tensor(n_samples, n_classes) of probabilities -> [[0.1, 0.9, ..., 0.001]] 1st dim == 1 in this case
        predicted_class = preds.argmax(dim=1).item()
        expected_class = target

    return predicted_class, expected_class


if __name__ == "__main__":
    ANNOTATIONS_FILE = Path(__file__).parent / "UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = Path(__file__).parent / "UrbanSound8K/audio"
    MODEL_PATH = Path(__file__).parent.parent / "checkpoint.pth"

    transform_pipeline = TransformPipeline(
        target_sample_rate=22050,
        num_samples=22050,
        n_fft=1024,
        n_mels=64,
    )

    test_dataset = UrbanSoundDataset(
        annotations_file=ANNOTATIONS_FILE,
        audio_dir=AUDIO_DIR,
        transformation=transform_pipeline,
        is_test=True,
        test_fold=10,
    )
    device = test_dataset.device

    model = CNNNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))

    # Prediction
    np.random.seed(0)
    for _ in range(5):
        i = np.random.randint(0, len(test_dataset))
        sample, target = test_dataset[i]
        sample = sample.unsqueeze(0).to(device)
        print(f"Sample: {sample.shape}, Target: {target}")
        predicted_class, expected_class = predict(model, sample, target)
        print(f"Predicted: {predicted_class}, Expected: {expected_class}")
