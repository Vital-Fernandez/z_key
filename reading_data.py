import numpy as np
import matplotlib.pyplot as plt
import lime
from pathlib import Path

# Folder with NIRSPEC spectra
data_folder = Path(f'./data')

# CEERs 1027 redshift
z_obj = 7.8334

# Multiple spectra from the same galaxy observed with diference dispensers
spec_list = data_folder.glob('*.fits')

for spec_address in spec_list:

    # Create LiMe spectrum
    spec = lime.Spectrum.from_file(spec_address, 'nirspec', redshift=z_obj)

    # Convert units as necessary
    spec.unit_conversion(wave_units_out='Angstrom', flux_units_out='FLAM')

    # Visualize the spectrum
    spec.plot.spectrum(rest_frame=True)

    # Compute the FWHM curve for the instrument observation (removing data mask)
    wave_obs = spec.wave.data
    deltalamb_arr = np.diff(wave_obs)
    R_arr = wave_obs[1:] / deltalamb_arr
    FWHM_arr = wave_obs[1:] / R_arr

    # Plot the FWHM_arr curve
    fig, ax = plt.subplots()
    # ax.plot(wave_obs[1:], R)
    ax.plot(wave_obs[1:], FWHM_arr)
    plt.show()
