# Finding the fourier transform of an MxN image using the 2D Discrete Fourier Transform

import numpy as np
from PIL import Image
from tqdm import tqdm

image = Image.open('mn8.png').convert('L')
im = np.array(image)
M = im.shape[0]
N = im.shape[1]

def freq_bin( u,v ):
    
    # Create frequency bins for each column in image
    
    YFreq_bins = np.zeros(N)
    totfreq_bin = 0
    for c in range(N):
        column_correlation = 0
        for r in range(M):
            column_correlation += im[r,c]* (np.cos((2*np.pi*u*r)/M) - np.complex(0,np.sin((2*np.pi*u*r)/M)))
        YFreq_bins[c] = column_correlation
        totfreq_bin += column_correlation* (np.cos((2*np.pi*v*c)/N) - np.complex(0,np.sin((2*np.pi*v*c)/N)))
        
    return totfreq_bin

#VERY slow without FFT algorithm
def DFT(image):
    DFTar = np.zeros([M+10,N+10])
    for u in tqdm(range(M+10)):
        for v in range(N+10):
            DFTar[u,v] = np.log(np.abs(freq_bin(u,v))) * 12

    DFTim = Image.fromarray(DFTar)
    DFTim.show()
    return DFTar


