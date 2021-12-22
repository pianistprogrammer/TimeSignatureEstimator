# import modules
import librosa
import IPython.display as ipd

# read audio file
x, sr = librosa.load('rap.wav')
ipd.Audio(x, rate=sr)

# approach 1 - onset detection and dynamic programming
tempo, beat_times = librosa.beat.beat_track(x, sr=sr, start_bpm=60, units='time')
clicks = librosa.clicks(beat_times, sr=sr, length=len(x))
ipd.Audio(x + clicks, rate=sr)
