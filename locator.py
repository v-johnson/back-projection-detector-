import obspy
import numpy as np
import time

def fft_shift(t,waveform,tDelay):

    samples = len(t)
    tstart = t[0]
    tend = t[-1]

    # 1. Take the FFT
    fftData = np.fft.fft(waveform)

    # 2. Construct the phase shift
    samplePeriod = (tend - tstart) / (samples)
    tDelayInSamples = tDelay / samplePeriod
    N = fftData.shape[0]
    k = np.linspace(0, N-1, N)
    timeDelayPhaseShift = np.exp(((-2*np.pi*1j*k*tDelayInSamples)/(N)) + (tDelayInSamples*np.pi*1j))

    # 3. Do the fftshift on the phase shift coefficients
    timeDelayPhaseShift = np.fft.fftshift(timeDelayPhaseShift)

    # 4. Multiply the fft data with the coefficients to apply the time shift
    fftWithDelay = np.multiply(fftData, timeDelayPhaseShift)

    # 5. Do the IFFT
    return  np.fft.ifft(fftWithDelay)

def tt(ice_thick, station_loc, ref_station_loc, trial_loc, wave_speed):
    d1 = np.sqrt(ice_thick**2 + (station_loc - trial_loc)**2)
    d2 = np.sqrt(ice_thick**2 + (ref_station_loc - trial_loc)**2)
    return (d1 - d2) / wave_speed

