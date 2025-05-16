import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import tkinter as tk
from tkinter import filedialog

def visualize_spectrum(fits_file):
    """
    可视化光谱数据，绘制Flux vs Wavelength曲线
    """
    with fits.open(fits_file) as hdul:
        data = hdul[1].data
        flux = data['flux'] if 'flux' in data.columns.names else data.field(0)
        if 'wavelength' in data.columns.names:
            wavelength = data['wavelength']
        elif 'loglam' in data.columns.names:
            wavelength = 10 ** data['loglam']
        else:
            wavelength = np.arange(len(flux))
    
    plt.figure(figsize=(10, 6))
    plt.plot(wavelength, flux, 'b-', label='Flux')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux')
    plt.title('Spectrum Visualization')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    root = tk.Tk()
    root.withdraw()
    fits_file = filedialog.askopenfilename(title='Select FITS file', filetypes=[('FITS files', '*.fits')])
    if fits_file:
        visualize_spectrum(fits_file)

if __name__ == '__main__':
    main() 