import wave
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import MelSpectrogram, Resample

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
        target_sample_rate: int = 16000,
        num_samples: int = 22050,
    ):
        """
        Pytorch Dataset implementation for UrbanSound8K audio files.

        Parameters:
        -----------
        - annotations_file: Path to the CSV file containing annotations.
        - audio_dir: Directory where audio files are stored.
        - transformation: Spectral transformation to be applied on a sample.
        - target_sample_rate: Sample rate to which audio files will be resampled.
        - num_samples: Number of samples to be extracted from each audio file.
            If <num_samples> is greater than the length of the audio file, the audio file will be padded with zeros.
        """
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = Path(audio_dir)
        self.device = get_device()
        self.transformation = transformation.to(self.device) if transformation else None
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
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
        waveform = self._resample(waveform, sample_rate)  # -> Tensor (num_channels, original_num_samples)
        waveform = self._mixdown(waveform)  # -> Tensor (1, original_num_samples)
        waveform = self._crop_or_pad(waveform)  # -> Tensor (1, self.num_samples)
        waveform = self._transform(waveform)  # -> Tensor (?,?)
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

    def _crop_or_pad(self, waveform):
        """Crop or pad the waveform to the target number of samples.

        Right side padding will be applied if the waveform is shorter than the target length.
        """
        if waveform.shape[1] > self.num_samples:
            waveform = waveform[:, : self.num_samples]
        elif waveform.shape[1] < self.num_samples:
            padding_size = self.num_samples - waveform.shape[1]
            # The padding size by which to pad some dimensions of input are
            # described starting from the last dimension and moving forward.
            # So len(pad)/2 dimensions of input will be padded
            # in this case we need to pad only the last dim, so the pad argument could look just like (left_pad, right_pad)
            # which is equal to (left_pad, right_pad, 0, 0)
            waveform = nn.functional.pad(waveform, (0, padding_size))
        return waveform

    def _mixdown(self, waveform):
        """Convert stereo to mono if necessary by averaging channels."""
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    def _transform(self, waveform):
        """Apply the transform to the waveform if specified."""
        if self.transformation:
            waveform = self.transformation(waveform)
        return waveform


if __name__ == "__main__":
    ANNOTATIONS_FILE = Path(__file__).parent / "UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = Path(__file__).parent / "UrbanSound8K/audio"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050  # time in seconds == NUM_SAMPLES / SAMPLE_RATE, so here it's 1 second

    device = get_device()
    print(f"Using device: {device}")

    mel_spectrogram_transform = MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64,
    )
    # resample_transform = Resample(orig_freq=44100, new_freq=SAMPLE_RATE)

    dataset = UrbanSoundDataset(
        annotations_file=ANNOTATIONS_FILE,
        audio_dir=AUDIO_DIR,
        transformation=mel_spectrogram_transform,
        target_sample_rate=SAMPLE_RATE,
        num_samples=NUM_SAMPLES,
    )
    print("Dataset size:", len(dataset))

    waveform, label = dataset[1]
    # print(waveform.shape, label)
    a = 1
