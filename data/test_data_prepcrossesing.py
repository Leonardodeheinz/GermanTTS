import torch
import math
import pytest
import matplotlib.pyplot as plt

# You would import your pipeline here
from PreProcesesAudio import Resampler, MelSpec

@pytest.fixture
def dummy_signal():
    sampling_rate = 24000
    duration = 1.0  # seconds
    t = torch.linspace(0, duration, int(sampling_rate * duration))
    freq = 440  # Hz (A4 note)

    signal = torch.sin(2 * math.pi * freq * t)
    signal = signal.unsqueeze(0)  # Add batch dimension
    return signal

def test_pipeline_basic(dummy_signal):
    # Fake pipeline for now if you need to test structure
    # Remove this and use your real pipeline
    def pipeline(x):
        return x.unsqueeze(1)  # simulate (batch, 1, time)

    output = pipeline(dummy_signal)

    # Basic checks
    assert output.ndim == 3, f"Expected output to be 3D, got {output.ndim}D shape {output.shape}"
    assert torch.isfinite(output).all(), "Output contains NaNs or Infs!"
    assert output.min() > -20 and output.max() < 20, f"Output values too extreme: {output.min()} to {output.max()}"

    # (Optional) Visual check
    plt.imshow(output.squeeze(0).cpu().numpy(), aspect='auto', origin='lower')
    plt.colorbar()
    plt.title("Mel Spectrogram Output (Test)")
    plt.savefig("mel_test_output.png")  # Save to file during tests
    plt.close()

    print("✅ Basic pipeline test passed!")

