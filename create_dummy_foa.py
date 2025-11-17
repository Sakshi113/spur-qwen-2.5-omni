import numpy as np
import soundfile as sf

SAMPLING_RATE = 16000
DURATION_SECONDS = 5
NUM_CHANNELS = 4  # FOA has 4 channels

# Create 5 seconds of white noise
dummy_audio = np.random.randn(DURATION_SECONDS * SAMPLING_RATE, NUM_CHANNELS).astype('float32')

# Save it as a 4-channel WAV file
sf.write('dummy_foa_4ch.wav', dummy_audio, SAMPLING_RATE)

print("Created 'dummy_foa_4ch.wav' with shape:", dummy_audio.T.shape) # .T to show (C, T)