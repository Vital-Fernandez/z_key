import numpy as np
import matplotlib.pyplot as plt
import lime
from pathlib import Path
from lime.model import gaussian_model


def compute_z_key(redshift, lines_lambda, wave_matrix, amp_arr, sigma_arr):

    # Compute the observed line wavelengths
    obs_lambda = lines_lambda * (1 + redshift)
    obs_lambda = obs_lambda[(obs_lambda > wave_matrix[0, 0]) & (obs_lambda < wave_matrix[0, -1])]

    if obs_lambda.size > 0:

        # Compute indexes ion array
        idcs_obs = np.searchsorted(wave_matrix[0, :], obs_lambda)

        # Compute lambda arrays:
        sigma_lines = sigma_arr[idcs_obs]
        mu_lines = wave_matrix[0, :][idcs_obs]

        # Compute the Gaussian bands
        x_matrix = wave_matrix[:idcs_obs.size, :]
        gauss_matrix = gaussian_model(x_matrix, amp_arr, mu_lines[:, None], sigma_lines[:, None])
        gauss_arr = gauss_matrix.sum(axis=0)

        # Set maximum to 1:
        idcs_one = gauss_arr > 1
        gauss_arr[idcs_one] = 1

    else:
        gauss_arr = None

    return gauss_arr


# Folder with NIRSPEC spectra
data_folder = Path(f'./data')

# CEERs 1027 redshift
z_obj = 7.8334

# Multiple spectra from the same galaxy observed with diference dispensers
spec_list = data_folder.glob('*.fits')
spec_list = list(spec_list)

# Recover a spectrum
s = spec_list[5]
spec = lime.Spectrum.from_file(s, 'nirspec', redshift=z_obj)
spec.unit_conversion(wave_units_out='Angstrom', flux_units_out='FLAM', norm_flux=1e-17)
spec.plot.spectrum()
# Get spectra and its mask
wave_obs = spec.wave.data
flux_obs = spec.flux.data
mask = ~spec.flux.mask if np.ma.isMaskedArray(spec.flux) else np.ones(flux_obs.size)
flux_scaled = (flux_obs - np.nanmin(flux_obs)) / (np.nanmax(flux_obs) - np.nanmin(flux_obs))

# Compute the resolution params
deltalamb_arr = np.diff(wave_obs)
R_arr = wave_obs[1:] / deltalamb_arr
FWHM_arr = wave_obs[1:] / R_arr
sigma_arr = np.zeros(wave_obs.size)
sigma_arr[:-1] = FWHM_arr / (2 * np.sqrt(2 * np.log(2)))
sigma_arr[-1] = sigma_arr[-2]

# Lines selection
full_df = lime.line_bands(vacuum=True)
candidate_lines = ["H1_1216A", "He2_1640A", "Ne5_3427A", "O2_3727A", "H1_4342A",
                   "H1_4863A", "O3_4960A", "O3_5008A", "H1_6565A", "S2_6718A", "He1_10833A"]
lines_df = full_df.loc[candidate_lines]
theo_lambda = lines_df.wavelength.to_numpy()

# Parameters for the brute analysis
z_arr = np.linspace(0, 10, 100 * 10)
wave_matrix = np.tile(wave_obs, (lines_df.index.size, 1))
F_sum = np.zeros(z_arr.size)

for i, z_i in enumerate(z_arr):

    # Generate the redshift key
    gauss_arr = compute_z_key(z_i, theo_lambda, wave_matrix, 1, sigma_arr)

    # Compute flux cumulative sum
    F_sum[i] = 0 if gauss_arr is None else np.sum(flux_obs[mask] * gauss_arr[mask])


z_max = z_arr[np.argmax(F_sum)]
print(np.max(F_sum), z_max)

# Plot the addition:
fig, ax = plt.subplots()
ax.step(z_arr, F_sum, where='mid', color='tab:blue')
ax.axvline(z_obj, color='red', linestyle='--', alpha=0.5)
ax.axvline(z_max, color='blue', linestyle='--', alpha=0.5)
plt.show()

gauss_arr_max = compute_z_key(z_max, theo_lambda, wave_matrix, 1, sigma_arr)

# Plot keys at max:
fig, ax = plt.subplots()
ax.step(wave_obs[mask], flux_scaled[mask], where='mid', color='tab:blue')
ax.step(wave_obs[mask], gauss_arr_max[mask], where='mid', color='tab:orange')
plt.show()