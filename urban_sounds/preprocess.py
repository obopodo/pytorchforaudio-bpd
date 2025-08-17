import torch
from torch import nn
from torchaudio import functional as AF
from torchaudio.transforms import MelSpectrogram


class TransformPipeline(nn.Module):
    def __init__(
        self,
        target_sample_rate: int = 22050,
        num_samples: int = 22050,
        n_fft: int = 1024,
        n_mels: int = 64,
    ):
        super().__init__()
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=n_fft,
            hop_length=n_fft // 2,
            n_mels=n_mels,
        )
        self.num_samples = num_samples
        self.target_sample_rate = target_sample_rate

    def forward(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Apply resampling (if needed), mixdown, length fix, and mel spectrogram.

        Parameters
        ----------
        waveform: Tensor[C, T]
            Audio waveform.
        sample_rate: int
            The actual sample rate of this waveform.
        """
        waveform = self._resample(waveform, sample_rate)
        waveform = self._mixdown(waveform)
        waveform = self._crop_or_pad(waveform)
        return self.mel_spectrogram(waveform)

    def _resample(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if sample_rate != self.target_sample_rate:
            # Use functional resample so we can handle varying input sample rates per item
            waveform = AF.resample(waveform, orig_freq=sample_rate, new_freq=self.target_sample_rate)
        return waveform

    def _mixdown(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert stereo to mono if necessary by averaging channels."""
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    def _crop_or_pad(self, waveform: torch.Tensor) -> torch.Tensor:
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
