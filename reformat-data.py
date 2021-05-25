import numpy as np
import obspy
from obspy import read
import os
from obspy.core.utcdatetime import UTCDateTime
import time
from sys import getsizeof
import matplotlib.pyplot as plt
import h5py
import multiprocessing as mp 
from multiprocessing import cpu_count


sample_rate = 1000
file_list = os.listdir("data") 
dt_zero = UTCDateTime("2019-01-14T00:00:00")
window_length = 1000
num_stations = 99
#file_list = file_list[0:2]


start_time = time.perf_counter()
# for t in range(0, int(86400/window_length)):
time = range(0, int(86400/window_length))
def write_section_files(t):
    print(t)
    dt = dt_zero + (t*window_length)
    # st = obspy.Stream()
    data_array = np.zeros((num_stations, window_length * sample_rate - 1))
    for i, file_name in enumerate(file_list):
        st_new = obspy.read('data/%s'%file_name)
        st_new.trim(dt + 1/sample_rate, dt + window_length - 1/sample_rate)
        st_new = st_new.select(component = 'Z') #I changed this to only use comp Z
        print(st_new.__str__(extended=True))
        # st += st_new
        data_array[i,:] = st_new[0].data
        del(st_new)
        #print('Loaded file in %f seconds'%(time.perf_counter()-start_time))
    # st.write("data-short/Time_%s.mseed" %dt, format = "MSEED")
    
    h5f = h5py.File('data-short/Time_%s.h5' %dt, 'w')
    h5f.create_dataset('Vertical Component Seismogram', data = data_array)       
    h5f.close()
    print("Files have been written.")
    # dt += window_length
    # del(st)
    #print('Wrote file in %f seconds' %(time.perf_counter()-start_time))
    # print('Loaded file in %f seconds'%(time.perf_counter()-start_time))
    
nprocs = mp.cpu_count()
print(f"Number of CPU cores: {nprocs}")
pool = mp.Pool(processes=3)
print("Actively using three cores.")

# write_section_files(time[47])
pool.map(write_section_files, time)
print("Full process has completed for this day.")