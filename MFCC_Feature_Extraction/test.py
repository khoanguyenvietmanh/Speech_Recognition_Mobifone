from MFCC import mfcc
from MFCC import get_delta_feature
from MFCC import logfbank
import scipy.io.wavfile as wav
from Gaussian_Mixture_Model.GMM import GMM
import numpy as np
import librosa


if __name__ == "__main__":
    (rate, sig) = wav.read("english.wav")
    (signal, sampling_rate) = librosa.load("english.wav", sr=8000)


    print(rate)
    print(sampling_rate)
    # mfcc_feat = mfcc(sig, rate)
    # delta_mfcc_feat = get_delta_feature(mfcc_feat, N=2)
    # ddelta_mfcc_feat = get_delta_feature(delta_mfcc_feat, N=2, order=2)
    # fbank_feat = logfbank(sig, rate)

    print(signal)

    print(sig)




