import math

class SignalProcessor ( object ):
    def __init__ ( self , data ):
        self . data = data

class SineWaveFunction ( SignalProcessor ):
    def __init__ ( self , amplitude , frequency ):
        super () . __init__ ([])
        self . amplitude = amplitude
        self . frequency = frequency
    def __call__(self,duration):
        self.data = []
        for i in range(duration):
            self.data.append(self.amplitude * math.sin(2*math.pi*self.frequency*i))
        print(f"Sine wave data:{self.data}")
    
    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        self.index = 0 
        return self
    
    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        current_value = self.data[self.index]
        self.index += 1
        return current_value
    
    def __eq__(self,b):
        count = 0
        if len(self.data) != len(b.data):
            raise ValueError("Two signals are not equal in length")
        else:
            for i in range(len(self.data)):
                if abs(self.data[i]-b.data[i])<0.01:
                    count += 1
            return count

class SquareWaveFunction ( SineWaveFunction ):
    def __init__ ( self , amplitude , frequency ):
        super (SquareWaveFunction,self) . __init__ ( 1 , frequency )
        self . amplitude = amplitude
    
    def __call__(self,duration):
        super(). __call__(duration)
        for i in range(len(self.data)):
            if self.data[i] >= 0:
                self.data[i] = self.amplitude
            else:
                self.data[i] = -self.amplitude
        print(f"Square wave data:{self.data}")
    

SG1 = SineWaveFunction(amplitude =2.0 ,frequency =0.1)
SG1 (duration = 5)

SG = SineWaveFunction(amplitude =2.0 ,frequency =0.1)
SG ( duration =5 )
print (len( SG ) )

SG = SineWaveFunction(amplitude =2.0 ,frequency =0.1)
SG(duration =5 )
print([val for val in SG]) 

SW = SquareWaveFunction(amplitude =3.0 , frequency =0.1)
SW(duration =5 )
print(len( SW ) )
print([val for val in SW]) 

SG1 = SineWaveFunction (amplitude =2.0 , frequency =0.1)
SG1 (duration =5)
SG2 = SineWaveFunction ( amplitude =2.0 , frequency =0.15 )
SG2 ( duration =5 )
print ( SG1 == SG2 )
# Code commented out as it raises error
# SG3 = SineWaveFunction ( amplitude =2.0 , frequency =0.1 )
# SG3(duration =3)
# print(SG1 == SG3) 

class CompositeSignalFunction ( SignalProcessor ):
    def __init__ ( self , inputs ):
        super () . __init__ ([])
        self.inputs = inputs
    
    def __call__(self,duration):
        self.data = [0]*duration
        for waves in self.inputs:
            waves(duration)
        for i in range(duration):
            for waves in self.inputs:
                self.data[i]+=waves.data[i]
        print(f"Composite signal data:{self.data}")


SG = SineWaveFunction ( amplitude = 1.0 , frequency = 0.1)
SW = SquareWaveFunction ( amplitude = 0.5 , frequency = 0.05)
CSG = CompositeSignalFunction ( inputs = [ SG , SW ] )
CSG ( duration = 80 )

def get_fft_spectrum(signal_data):
    fft_vals = np.fft.fft(signal_data)
    # We only need the first half (positive frequencies)
    half_n = len(signal_data) // 2
    magnitude = np.abs(fft_vals)[:half_n]
    return magnitude

import matplotlib.pyplot as plt
import numpy as np
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

time_axis = range(80)

ax1.plot(time_axis, SG.data, label='Sine Wave', color='blue')
ax1.plot(time_axis, SW.data, label='Square Wave', color='green')
ax1.plot(time_axis, CSG.data, label='Composite Signal', color='red')

ax1.set_title('Time Domain Analysis')
ax1.set_xlabel('Time (samples)')
ax1.set_ylabel('Amplitude')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)


fft_sine = get_fft_spectrum(SG.data)
fft_square = get_fft_spectrum(SW.data)
fft_composite = get_fft_spectrum(CSG.data)

# Create frequency axis (Normalized Frequency 0 to 0.5)
freqs = np.linspace(0, 0.5, len(fft_sine))

ax2.plot(freqs, fft_sine, label='Sine FFT', color='blue')
ax2.plot(freqs, fft_square, label='Square FFT', color='green')
ax2.plot(freqs, fft_composite, label='Composite FFT', color='red')

ax2.set_title('Frequency Domain Analysis (FFT)')
ax2.set_xlabel('Normalized Frequency')
ax2.set_ylabel('Magnitude')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.savefig('signal_analysis_plot.png', dpi=300, bbox_inches='tight')
print("Figure saved as 'signal_analysis_plot.png'")