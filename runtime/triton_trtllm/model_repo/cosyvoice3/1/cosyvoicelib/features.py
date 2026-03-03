from matcha.utils.audio import mel_spectrogram
from numpy.typing import NDArray
import numpy as np


def extract_speech_feat(speech: NDArray[np.float32]) -> NDArray[np.float32]:
    speech_feat = (
        mel_spectrogram(
            speech,
            n_fft=1920,
            num_mels=80,
            sampling_rate=24000,
            hop_size=480,
            win_size=1920,
            fmin=0,
            fmax=8000,
        )
        .squeeze(dim=0)
        .transpose(0, 1)
    )
    return speech_feat.unsqueeze(dim=0).numpy()
