import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import rfft, irfft, fftfreq, fft, ifft
from scipy.signal import butter, lfilter
from scipy.interpolate import interp1d

def get_dtype(s1, s2):
    return (np.empty(0, dtype=s1.dtype)*np.empty(0, dtype=s2.dtype)).dtype

def padto_lensum_pow2(s1, s2):
    dtype = get_dtype(s1, s2)
    vlen = int(2**np.ceil(np.log2(len(s1)+len(s2))))
    
    spad1 = np.zeros(vlen, dtype=dtype)
    spad2 = np.zeros(vlen, dtype=dtype)
    
    spad1[:len(s1)] = s1
    spad2[:len(s2)] = s2
    
    return spad1, spad2

def xcorr(s1,s2):
    s1, s2 = padto_lensum_pow2(s1, s2)

    f_s1 = fft(s1)
    f_s2 = fft(s2)
    f_s2c = np.conj(f_s2)
    f_s = f_s1 * f_s2c
    
    #denom = abs(f_s)
    #denom[denom < 1e-6] = 1e-6
    #f_s = f_s / denom  # This line is the only difference between GCC-PHAT and normal cross correlation
    return np.abs(ifft(f_s))[:]

def pcorr(s1, s2):
    s1, s2 = padto_lensum_pow2(s1, s2)

    G_a = fft(s1)
    G_b = fft(s2)
    conj_b = np.ma.conjugate(G_b)
    R = G_a*conj_b
    R /= np.absolute(R)
    #r = np.absolute(ifft(R))#.real
    r = (ifft(R)).real
    return r

def phasedelay(s1, s2, use_xcorr=False, subsample_rate=None, return_severity=False):
    '''
    s1 [xxxxxxxxxxxxx]    signal 
    s2 [___xxxxxxxxxxxxx] signal delay with 3 samples
    the above will return -3
    
    s1 [___xxxxxxxxxxxxx] signal delay with 3 samples
    s2 [xxxxxxxxxxxxx]    signal
    the above will return 3
    '''
    theCorr = xcorr if use_xcorr else pcorr
    corr = theCorr(s1,s2)
    diff = np.argmax(corr)
    if diff > len(s1):
        diff = diff - len(corr)
        corrmaxval = corr[diff]

    #Use spline interpolation to finer-tune the maximum correlated point
    if subsample_rate is not None:
        #The size around the maximum point to be evaluated
        slice_offset = 6
        idx = np.arange(diff-slice_offset, diff+slice_offset).astype('int')
        corrslice = corr.take(idx, mode='wrap')

        subidx = np.linspace(diff-2, diff+2, 2*2*subsample_rate)
        corrslice_subsampled = interp1d(idx, corrslice, kind='cubic')(subidx)

        diff = subidx[np.argmax(corrslice_subsampled)]
        corrmaxval = np.max(corrslice_subsampled)

    return -diff, corrmaxval

def butter_bandpass_response(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass_response(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

#***************************************************************************************
# Some tests
#
#***************************************************************************************
import unittest
class TestFft(unittest.TestCase):
    def test_xcorr(self):pass
    def test_pcorr(self): pass
    def test_phasediff(self): pass
    def test_phasediff_subsample(self): pass
    def test_butter_bandpass_response(self): pass
    def test_butter_bandpass_filter(self): pass
if __name__ == "__main__":
    unittest.main()
