import obspy
import numpy as np
import time
import h5py
import matplotlib.pyplot as plt

'''
This function is responsible for the frequency shift in data. 
'''
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

'''
This function defines the backprojection routine. 
                   |||INPUTS|||
|NAME|                |TYPE|         |DESCRIP.|
locations             list           Every trial location based on station distance min/max [m]
traces                list           All traces for given stream
chunk_length          int            Length of time for chunk of data [s]
chunk_index           int            Index of chunk of the total, total = 86400/chunk_length
w                     int            Adjustment factor, default is 25
ice_thickness         int            Thickness of ice, default is 3450 [m]
station_location_x    list           All station distances
shear_wave_speed      int            Wave speed for shear waves, default is 2000 [m/s]
'''

def backproject(locations, traces, chunk_length, chunk_index, w, ice_thickness, station_location_x, shear_wave_speed):
    big_matrix = np.zeros((chunk_length-w, len(locations)))
    t = traces[0].times()
    
    start_time = time.perf_counter()
    
    for i in np.arange(0,len(locations)):
        shifted_traces = []
        for j in range(0, len(traces)):
            current_trace = traces[j][((chunk_length * chunk_index) + 1): chunk_length * (chunk_index + 1)]
            print('Current_trace is set')
            current_time = t[((chunk_length * chunk_index) + 1): chunk_length * (chunk_index + 1)]
            print('Current_time is set')
            tr_avg = (np.convolve(np.abs(current_trace), np.ones(w), 'valid') / w)
#             tr_norm = current_trace[(w-1):] / tr_avg[0:]
            print('tr_avg has been calculated.')
            tr_norm = tr_avg[0:] / np.mean(tr_avg[0:])
#             tr_norm = current_trace  # for synthetic data + noise
            print('tr_norm has been calculated.')
            tr_shift = fft_shift(current_time,\
                                tr_norm, tt(ice_thickness, station_location_x[j], station_location_x[0], \
                                locations[i], shear_wave_speed))
            shifted_traces.append(tr_shift)
            print('Added to shifted_traces')
            print('In BackProject Fn          Finished trace loop %d in %f seconds'%(j,time.perf_counter()-start_time))
        stack = sum(shifted_traces)
        if len(big_matrix[:,i]) != len(np.real(stack)):
            print("WARNING: SIZE MISMATCH")
            continue
        big_matrix[:,i] = np.real(stack)
        print('In BackProject fn     Finished location loop %d in %f seconds'%(i,time.perf_counter()-start_time))
    return big_matrix

'''
This function runs backprojection on time chunks of data.
                    |||INPUTS|||
|NAME|                 |TYPE|             |DESCRIP.|
locations              list               Every trial location based on station distance min/max [m]
traces                 list               All traces for given stream
station_location_x     list               All station distances [m]
name                   string             Desired name description for file, ie. how long the time interval is.
shear_wave_speed       int                Wave speed for shear waves, default is 2000 [m/s]
'''
def run_back_projection(locations, traces, station_location_x, name, shear_wave_speed = 2000, \
                        ice_thickness = 3450, w = 25, chunk_len = 1080):
  
    t = traces[0].times()
    num_iterations = int(np.floor(len(t) / chunk_len))

    start_time = time.perf_counter()

    for chunk_index in range(0,num_iterations+1):
        big_matrix = backproject(locations, traces, chunk_len, \
                                 chunk_index, w, ice_thickness, station_location_x, shear_wave_speed)
        print('Min of matrix = %f' %np.min(big_matrix) + ' Max of big matrix = %f' %np.max(big_matrix))
        h5f = h5py.File('output/BP_%s_%d_of_%d.h5' %(name,chunk_index,num_iterations), 'w')
        print('H5py file has been created.')
        h5f.create_dataset('Backprojection Array', data=big_matrix)       
        h5f.close()
        del(big_matrix)
        print('Big matrix has been deleted')
        print('Finished time loop %d in %f seconds'%(chunk_index,time.perf_counter()-start_time))