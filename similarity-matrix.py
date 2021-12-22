
from libfmp.b import FloatingBox
import numpy as np
import os
import sys
import librosa
from scipy import signal
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import IPython.display as ipd
import pandas as pd
from numba import jit

import libfmp.c6
import libfmp.c4
import libfmp.c3
import libfmp.c2
import libfmp.b

sys.path.append('..')

# Generate normalized feature sequence
K = 4
M = 100
r = np.arange(M)
b1 = np.zeros((K, M))
b1[0, :] = r
b1[1, :] = M-r
b2 = np.ones((K, M))
X = np.concatenate((b1, b1, np.roll(b1, 2, axis=0), b2, b1), axis=1)
X = libfmp.c3.normalize_feature_sequence(X, norm='2', threshold=0.001)

# Compute SSM
S = np.dot(np.transpose(X), X)

# Visualization
cmap = 'gray_r'
fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 0.05],
                                          'height_ratios': [0.2, 1]}, figsize=(4.5, 5))
libfmp.b.plot_matrix(X, Fs=1, ax=[ax[0, 0], ax[0, 1]], cmap=cmap,
                     xlabel='Time (frames)', ylabel='', title='Feature sequence')
libfmp.b.plot_matrix(S, Fs=1, ax=[ax[1, 0], ax[1, 1]], cmap=cmap,
                     title='SSM', xlabel='Time (frames)', ylabel='Time (frames)', colorbar=True)
plt.tight_layout()

cmap = libfmp.b.compressed_gray_cmap(alpha=-1000)
fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 0.05], 
                                          'height_ratios': [0.2, 1]}, figsize=(4.5, 5))
libfmp.b.plot_matrix(X, Fs=1, ax=[ax[0,0], ax[0,1]], cmap=cmap,
            xlabel='Time (frames)', ylabel='', title='Feature sequence')
libfmp.b.plot_matrix(S, Fs=1, ax=[ax[1,0], ax[1,1]], cmap=cmap,
            title='SSM', xlabel='Time (frames)', ylabel='Time (frames)', colorbar=True);
plt.tight_layout()


# def compute_sm_dot(X, Y):
    
#     S = np.dot(np.transpose(X), Y)
#     return S


# def plot_feature_ssm(X, Fs_X, S, Fs_S, ann, duration, color_ann=None,
#                      title='', label='Time (seconds)', time=True,
#                      figsize=(5, 6), fontsize=10, clim_X=None, clim=None):
#     """Plot SSM along with feature representation and annotations (standard setting is time in seconds)

#     Notebook: C4/C4S2_SSM.ipynb

#     Args:
#         X: Feature representation
#         Fs_X: Feature rate of ``X``
#         S: Similarity matrix (SM)
#         Fs_S: Feature rate of ``S``
#         ann: Annotaions
#         duration: Duration
#         color_ann: Color annotations (see :func:`libfmp.b.b_plot.plot_segments`) (Default value = None)
#         title: Figure title (Default value = '')
#         label: Label for time axes (Default value = 'Time (seconds)')
#         time: Display time axis ticks or not (Default value = True)
#         figsize: Figure size (Default value = (5, 6))
#         fontsize: Font size (Default value = 10)
#         clim_X: Color limits for matrix X (Default value = None)
#         clim: Color limits for matrix ``S`` (Default value = None)

#     Returns:
#         fig: Handle for figure
#         ax: Handle for axes
#     """
#     cmap = libfmp.b.compressed_gray_cmap(alpha=-10)
#     fig, ax = plt.subplots(3, 3, gridspec_kw={'width_ratios': [0.1, 1, 0.05],
#                                               'wspace': 0.2,
#                                               'height_ratios': [0.3, 1, 0.1]},
#                            figsize=figsize)
#     libfmp.b.plot_matrix(X, Fs=Fs_X, ax=[ax[0, 1], ax[0, 2]], clim=clim_X,
#                          xlabel='', ylabel='', title=title)
#     ax[0, 0].axis('off')
#     libfmp.b.plot_matrix(S, Fs=Fs_S, ax=[ax[1, 1], ax[1, 2]], cmap=cmap, clim=clim,
#                          title='', xlabel='', ylabel='', colorbar=True)
#     ax[1, 1].set_xticks([])
#     ax[1, 1].set_yticks([])
#     libfmp.b.plot_segments(ann, ax=ax[2, 1], time_axis=time, fontsize=fontsize,
#                            colors=color_ann,
#                            time_label=label, time_max=duration*Fs_X)
#     ax[2, 2].axis('off'), ax[2, 0].axis('off')
#     libfmp.b.plot_segments(ann, ax=ax[1, 0], time_axis=time, fontsize=fontsize,
#                            direction='vertical', colors=color_ann,
#                            time_label=label, time_max=duration*Fs_X)
#     return fig, ax


# # Waveform
# fn_wav = 'rap.wav'

# Fs = 22050
# x, Fs = librosa.load(fn_wav, Fs)
# x_duration = (x.shape[0])/Fs

# # Chroma Feature Sequence
# N, H = 4096, 1024
# chromagram = librosa.feature.chroma_stft(
#     y=x, sr=Fs, tuning=0, norm=2, hop_length=H, n_fft=N)
# X, Fs_X = libfmp.c3.smooth_downsample_feature_sequence(
#     chromagram, Fs/H, filt_len=41, down_sampling=10)

# # Annotation
# filename = 'FMP_C4_Audio_Brahms_HungarianDances-05_Ormandy.csv'
# fn_ann = os.path.join('..', 'data', 'C4', filename)
# ann, color_ann = libfmp.c4.read_structure_annotation(
#     fn_ann, fn_ann_color=filename)
# ann_frames = libfmp.c4.convert_structure_annotation(ann, Fs=Fs_X)

# # SSM
# X = libfmp.c3.normalize_feature_sequence(X, norm='2', threshold=0.001)
# S = compute_sm_dot(X, X)
# fig, ax = plot_feature_ssm(X, 1, S, 1, ann_frames, x_duration*Fs_X, color_ann=color_ann,
#                            clim_X=[0, 1], clim=[0, 1], label='Time (frames)',
#                            title='Chroma feature (Fs=%0.2f)' % Fs_X)



# float_box = libfmp.b.FloatingBox()

# # MFCC-based feature sequence
# N, H = 2048, 1024
# X_MFCC = librosa.feature.mfcc(y=x, sr=Fs, hop_length=H, n_fft=N)
# coef = np.arange(0, 20)
# X_MFCC_upper = X_MFCC[coef, :]
# X, Fs_X = libfmp.c3.smooth_downsample_feature_sequence(
#     X_MFCC_upper, Fs/H, filt_len=41, down_sampling=10)
# X = libfmp.c3.normalize_feature_sequence(X, norm='2', threshold=0.001)
# S = compute_sm_dot(X, X)
# ann_frames = libfmp.c4.convert_structure_annotation(ann, Fs=Fs_X)
# fig, ax = plot_feature_ssm(X, 1, S, 1, ann_frames, x_duration*Fs_X, color_ann=color_ann,
#                            title='MFCC (20 coefficients, Fs=%0.2f)' % Fs_X, label='Time (frames)')
# float_box.add_fig(fig)


# # MFCC-based feature sequence only using coefficients 4 to 14
# coef = np.arange(4, 15)
# X_MFCC_upper = X_MFCC[coef, :]
# X, Fs_X = libfmp.c3.smooth_downsample_feature_sequence(
#     X_MFCC_upper, Fs/H, filt_len=41, down_sampling=10)
# X = libfmp.c3.normalize_feature_sequence(X, norm='2', threshold=0.001)
# S = compute_sm_dot(X, X)
# ann_frames = libfmp.c4.convert_structure_annotation(ann, Fs=Fs_X)
# fig, ax = plot_feature_ssm(X, 1, S, 1, ann_frames, x_duration*Fs_X,
#                            color_ann=color_ann, label='Time (frames)',
#                            title='MFCC (coefficients 4 to 14, Fs=%0.2f)' % Fs_X)
# float_box.add_fig(fig)

# float_box.show()


# Tempogram feature sequence
# nov, Fs_nov = libfmp.c6.compute_novelty_spectrum(
#     x, Fs=Fs, N=2048, H=512, gamma=100, M=10, norm=1)
# nov, Fs_nov = libfmp.c6.resample_signal(nov, Fs_in=Fs_nov, Fs_out=100)


# N, H = 1000, 100
# X, T_coef, F_coef_BPM = libfmp.c6.compute_tempogram_fourier(
#     nov, Fs_nov, N=N, H=H, Theta=np.arange(30, 601))
# octave_bin = 12
# tempogram_F = np.abs(X)
# output = libfmp.c6.compute_cyclic_tempogram(
#     tempogram_F, F_coef_BPM, octave_bin=octave_bin)
# X = output[0]
# F_coef_scale = output[1]
# Fs_X = Fs_nov/H
# X = libfmp.c3.normalize_feature_sequence(X, norm='2', threshold=0.001)
# S = compute_sm_dot(X, X)
# ann_frames = libfmp.c4.convert_structure_annotation(ann, Fs=Fs_X)
# fig, ax = plot_feature_ssm(X, 1, S, 1, ann_frames, x_duration*Fs_X, color_ann=color_ann,
#                            title='Tempogram (Fs=%0.2f)' % Fs_X, label='Time (frames)')


if (__name__ == '__main__'):
    Statements only executed when run as a script
