from pathlib import Path

import pandas as pd
import torchaudio
from torch.utils.data import DataLoader, Dataset


class UrbanSoundDataset(Dataset):
    def __init__(self, annotations_file: str, audio_dir: str, transform=None):
        """
        Parameters:
        -----------
        - annotations_file: Path to the CSV file containing annotations.
        - audio_dir: Directory where audio files are stored.
        - transform: Optional transform to be applied on a sample.
        """
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = Path(audio_dir)
        self.transform = transform
        assert self.audio_dir.exists(), f"audio_dir {self.audio_dir} does not exist."

    def __len__(self):
        return len(self.annotations)

    def _get_audio_sample_path_and_label(self, idx):
        row = self.annotations.iloc[idx]
        filename = row["slice_file_name"]
        foldname = f"fold{row['fold']}"
        filepath = self.audio_dir / foldname / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} does not exist.")
        label = row["classID"]
        return filepath, label

    def __getitem__(self, idx):
        filepath, label = self._get_audio_sample_path_and_label(idx)
        waveform, sample_rate = torchaudio.load(str(filepath), format="wav")
        if self.transform:
            waveform = self.transform(waveform)
        return waveform, label


if __name__ == "__main__":
    ANNOTATIONS_FILE = Path(__file__).parent / "UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = Path(__file__).parent / "UrbanSound8K/audio"

    assert ANNOTATIONS_FILE.exists(), f"annotations_file {ANNOTATIONS_FILE} does not exist."
    assert AUDIO_DIR.exists(), f"audio_dir {AUDIO_DIR} does not exist."

    dataset = UrbanSoundDataset(
        annotations_file=ANNOTATIONS_FILE,
        audio_dir=AUDIO_DIR,
    )
    print("Dataset size:", len(dataset))

    waveform, label = dataset[0]
    print(waveform.shape, label)
