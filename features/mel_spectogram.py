import jax.numpy as jnp
from jax.scipy.signal import stft
from flax import nnx 
from jax.random import PRNGKey, normal
import matplotlib.pyplot as plt

# abeondoned for now !!

class MelSpec(nnx.Module):
    """
    

    Args:
        sample_rate: int --
    """
    
    def __init__(
        self,
        sample_rate: int = 24_000,
        n_fft:int = 1024,
        hop_length:int = 256,
        n_mels: int = 80,
        f_min: int = 0,
        f_max: int = None,        
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self._mel_filter_bank()

    def _mel_filter_bank(self):
        """Create a Mel filter bank matrix using JAX."""
        # Compute the Mel frequencies
        mel_frequencies = jnp.linspace(self._hz_to_mel(self.f_min), 
                                       self._hz_to_mel(self.f_max), 
                                       self.n_mels + 2)
        hz_frequencies = self._mel_to_hz(mel_frequencies)
        bin_frequencies = jnp.floor((self.n_fft + 1) * hz_frequencies / self.sr).astype(int)

        # Create filter bank
        filters = jnp.zeros((self.n_mels, self.n_fft // 2 + 1))
        for i in range(self.n_mels):
            filters = filters.at[i, bin_frequencies[i]: bin_frequencies[i + 1]].set(
                jnp.linspace(0, 1, bin_frequencies[i + 1] - bin_frequencies[i] + 1)
            )
            filters = filters.at[i, bin_frequencies[i + 1]: bin_frequencies[i + 2]].set(
                jnp.linspace(1, 0, bin_frequencies[i + 2] - bin_frequencies[i + 1] + 1)
            )
        return filters
    
    def _hz_to_mel(self, hz):
        return 2595 * jnp.log10(1 + hz / 700)
    
    def _mel_to_hz(self):
        pass
    
    
    def transform(self, wav):
        """Convert a WAV sample to a Mel spectrogram batch using JAX STFT."""
        f, t, Zxx = stft(wav, fs=self.sr, nperseg=self.n_fft, noverlap=self.n_fft - self.hop_length)
        spectrogram = jnp.abs(Zxx) ** 2  # Power spectrogram

        # Apply Mel filter bank
        mel_spec = jnp.dot(self.mel_filter, spectrogram)
        mel_spec = jnp.log1p(mel_spec)  # Log scaling (log1p for numerical stability)
        return mel_spec, t
    
def plot_mel_spectrogram(mel_spec, t, sr=22050, n_mels=80):
    """Plot the Mel spectrogram using matplotlib."""
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec, aspect='auto', origin='lower', cmap='magma', extent=[t.min(), t.max(), 0, n_mels])
    plt.colorbar(label="Log Magnitude")
    plt.xlabel("Time (s)")
    plt.ylabel("Mel Frequency Bins")
    plt.title("Mel Spectrogram")
    plt.show()

    
    
    
key = PRNGKey(0)
wav = normal(key, (22050,))  # Fake 1-second audio at 22050 Hz
converter = MelSpec()
mel_spec, time_axis = converter.transform(wav)

print(mel_spec.shape)  # (n_mels, Frames)

# Plot the Mel spectrogram
plot_mel_spectrogram(mel_spec, time_axis, sr=22050, n_mels=80)