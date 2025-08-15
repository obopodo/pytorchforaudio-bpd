import wave
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram, Resample

from urban_sounds.preprocess import TransformPipeline
from urban_sounds.utils import get_device


class UrbanSoundDataset(Dataset):
    """Dataset for UrbanSound8K audio files.
    https://urbansounddataset.weebly.com/urbansound8k.html
    """

    def __init__(
        self,
        annotations_file: str,
        audio_dir: str,
        transformation: nn.Module,
    ):
        """
        Pytorch Dataset implementation for UrbanSound8K audio files.

        Parameters:
        -----------
        - annotations_file: Path to the CSV file containing annotations.
        - audio_dir: Directory where audio files are stored.
        - transformation: Spectral transformation to be applied on a sample.
        """
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = Path(audio_dir)
        self.device = get_device()
        self.transformation = transformation.to(self.device) if transformation else None
        assert self.audio_dir.exists(), f"audio_dir {self.audio_dir} does not exist."

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get audio sample and its label by index."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset.")

        filepath, label = self._get_audio_sample_path_and_label(idx)
        waveform, sample_rate = self._load_audio(str(filepath))
        waveform = waveform.to(self.device)
        waveform = self._transform(waveform, sample_rate)  # -> Tensor (n_channels, n_mels, ceil(n_samples / hop_size))
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

    def _transform(self, waveform, sample_rate):
        """Apply the transform to the waveform if specified."""
        if self.transformation:
            waveform = self.transformation(waveform, sample_rate)
        return waveform


if __name__ == "__main__":
    ANNOTATIONS_FILE = Path(__file__).parent / "UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = Path(__file__).parent / "UrbanSound8K/audio"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050  # time in seconds == NUM_SAMPLES / SAMPLE_RATE, so here it's 1 second

    device = get_device()
    print(f"Using device: {device}")

    transform_pipeline = TransformPipeline(
        target_sample_rate=SAMPLE_RATE,
        num_samples=NUM_SAMPLES,
        n_fft=1024,
        n_mels=64,
    )
    # resample_transform = Resample(orig_freq=44100, new_freq=SAMPLE_RATE)

    dataset = UrbanSoundDataset(
        annotations_file=ANNOTATIONS_FILE,
        audio_dir=AUDIO_DIR,
        transformation=transform_pipeline,
    )
    print("Dataset size:", len(dataset))

    waveform, label = dataset[1]
    print("Mel Spectrogram shape:", waveform.shape, "\nLabel:", label)
    a = 1
