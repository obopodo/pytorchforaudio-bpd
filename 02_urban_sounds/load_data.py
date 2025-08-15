from pathlib import Path

import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import Resample


class UrbanSoundDataset(Dataset):
    """Dataset for UrbanSound8K audio files.
    https://urbansounddataset.weebly.com/urbansound8k.html
    """

    def __init__(self, annotations_file: str, audio_dir: str, transform=None, target_sample_rate=16000):
        """
        Pytorch Dataset implementation for UrbanSound8K audio files.

        Parameters:
        -----------
        - annotations_file: Path to the CSV file containing annotations.
        - audio_dir: Directory where audio files are stored.
        - transform: Optional transform to be applied on a sample.
        - target_sample_rate: Sample rate to which audio files will be resampled.
        """
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = Path(audio_dir)
        self.transform = transform
        self.target_sample_rate = target_sample_rate
        assert self.audio_dir.exists(), f"audio_dir {self.audio_dir} does not exist."

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get audio sample and its label by index."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset.")

        filepath, label = self._get_audio_sample_path_and_label(idx)
        waveform, sample_rate = self._load_audio(str(filepath))
        waveform = self._resample(waveform, sample_rate)
        if self.transform:
            waveform = self.transform(waveform)
        waveform = self._mixdown(waveform)
        return waveform, label

    def _get_audio_sample_path_and_label(self, idx) -> tuple[Path, int]:
        row = self.annotations.iloc[idx]
        filename = row["slice_file_name"]
        foldname = f"fold{row['fold']}"
        filepath = self.audio_dir / foldname / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} does not exist.")
        label = row["classID"]
        return filepath, label

    def _load_audio(self, filepath) -> tuple[torch.Tensor, int]:
        """Load audio file and return waveform and sample rate."""
        waveform, sample_rate = torchaudio.load(filepath, format="wav")
        return waveform, sample_rate

    def _resample(self, waveform, input_sample_rate):
        """Resample the waveform to the target sample rate if necessary."""
        if input_sample_rate != self.target_sample_rate:
            resample_transform = Resample(orig_freq=input_sample_rate, new_freq=self.target_sample_rate)
            waveform = resample_transform(waveform)
        return waveform

    def _mixdown(self, waveform):
        """Convert stereo to mono if necessary by averaging channels."""
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform


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
