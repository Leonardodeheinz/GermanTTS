from jax import Array
from jax.scipy.signal import stft
from flax import nnx 


class MelSpec(nnx.Module):
    """
    

    Args:
        sample_rate: int --
    """
    
    def __init__(
        self,
        sample_rate: int = 24_000,
        n_fft:int = 1024,
        hop_length= 256,
        n_mels=100,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def __call__(self, audio: jax.array, **kwargs) -> jax.array:
        return log_mel_spectrogram(audio, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length)
        
    @jax.jit
    def stft(self):
        pass
    
    def hz_to_mel(self):
        pass
    
    def mel_to_hz(self):
        pass