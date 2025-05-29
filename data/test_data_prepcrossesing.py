import torch
import math
import pytest
import matplotlib.pyplot as plt

# Replace with your actual module names
from PreProcesesAudio import Resampler, MelSpec

@pytest.fixture
def sampling_rates():
    return {
        "orig": 20000,
        "target": 24000
    }

@pytest.fixture
def dummy_signal(sampling_rates):
    sr = sampling_rates["orig"]
    duration = 1.0  # seconds
    t = torch.linspace(0, duration, int(sr * duration), dtype=torch.float32)
    freq = 440  # Hz (A4 note)
    signal = torch.sin(2 * math.pi * freq * t)
    return signal.unsqueeze(0)  # shape: (1, time)

def test_pipeline_resample_and_melspec(dummy_signal, sampling_rates):
    orig_sr = sampling_rates["orig"]
    target_sr = sampling_rates["target"]

    # Step 1: Resample
    resampler = Resampler(orig_freq=orig_sr, new_freq=target_sr)
    resampled = resampler.forward(dummy_signal)
    assert isinstance(resampled, torch.Tensor)
    assert resampled.ndim == 2  # (1, new_time)
    assert resampled.shape[1] > 0, "Resampled waveform is empty"
    
    # Step 2: Mel Spectrogram
    melspec = MelSpec(sampling_rate=target_sr)
    mel_output = melspec.forward(resampled)
    assert mel_output.ndim == 3, f"Expected output to be 3D, got {mel_output.ndim}D shape {mel_output.shape}"
    assert torch.isfinite(mel_output).all(), "Mel spectrogram contains NaNs or Infs"
    assert mel_output.shape[1] > 0 and mel_output.shape[2] > 0, "Mel spectrogram has invalid shape"

    # Value range sanity check
    assert mel_output.min() > -100 and mel_output.max() < 100, \
        f"Unexpected mel value range: {mel_output.min()} to {mel_output.max()}"

    # Save visualization
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_output[0].cpu().numpy(), aspect='auto', origin='lower')
    plt.colorbar()
    plt.title("Mel Spectrogram Output (Test)")
    plt.tight_layout()
    plt.savefig("test_figs/test.png")
    plt.close()

    print("✅ Resampler and MelSpec test passed!")

