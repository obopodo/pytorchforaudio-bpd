from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from urban_sounds.cnn import CNNNet
from urban_sounds.dataset import UrbanSoundDataset
from urban_sounds.preprocess import TransformPipeline


def predict(model: CNNNet, sample: torch.Tensor) -> tuple:
    model.eval()
    with torch.no_grad():
        predicted_probas: torch.Tensor = model.softmax(model(sample))
        # preds is Tensor(n_samples, n_classes) of probabilities -> [[0.1, 0.9, ..., 0.001]] 1st dim == 1 in this case
        predicted_class = predicted_probas.argmax(dim=1).item()
    return predicted_class, predicted_probas


if __name__ == "__main__":
    from urban_sounds.constants import ANNOTATIONS_FILE, AUDIO_DIR

    MODEL_PATH = Path(__file__).parent / "models/checkpoint_20250817_175507.pth"

    assert MODEL_PATH.exists(), f"Model file {MODEL_PATH} does not exist."

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
        test_fold=9,
    )

    device = test_dataset.device
    test_data_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    print("Dataset loaded")

    model = CNNNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))

    all_preds = []
    all_probas = []
    all_targets = []

    for i in tqdm(range(len(test_dataset)), total=len(test_dataset)):
        sample, target = test_dataset[i]
        sample = sample.unsqueeze(0).to(device)
        predicted_class, predicted_probas = predict(model, sample)
        all_preds.append(predicted_class)
        all_targets.append(target)
        all_probas.append(predicted_probas.cpu().numpy()[0])

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probas = np.array(all_probas)

    accuracy = accuracy_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds, average="macro")
    precision = precision_score(all_targets, all_preds, average="macro")

    # For multiclass ROC AUC, need probability scores and one-hot targets
    all_targets_onehot = np.eye(10)[all_targets]
    all_preds_onehot = np.zeros((len(all_preds), 10))

    roc_auc = roc_auc_score(all_targets_onehot, all_probas, average="macro", multi_class="ovr")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
