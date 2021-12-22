import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

sample_rate, samples = wavfile.read('Rapp.wav')
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

plt.pcolormesh(times, frequencies, spectrogram)
plt.imshow(spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


#https://nlml.github.io/neural-networks/detecting-bpm-neural-networks/
# the repo https://github.com/nlml/bpm/tree/0f7b415fb89613804bfba2ca8b382c5fac2eb309

#http://tommymullaney.com/projects/rhythm-games-neural-networks
