import numpy as np
import matplotlib.pyplot as plt
import lime
from pathlib import Path


def wavelength_to_index(wave, wave_theo):
    if wave_theo < wave[0] or wave_theo > wave[-1]:
        return None
    return np.argmin(np.abs(wave - wave_theo))


def calculate_sum(wave, flux, wave_theo):
    sum = 0
    for wave_t in wave_theo:
        idx = wavelength_to_index(wave, wave_t)
        if idx is None:
            continue
        sum += flux[idx]
    return sum


def log_likelihood(theta, x, y, lines):
    z = theta
    sum = 0
    for line in lines:
        w = line * (1 + z)
        idx = wavelength_to_index(x, w)
        if idx is None:
            continue
        sum += y[idx]
    return sum


def log_prior(theta):
    z = theta
    if z < 0 or z > 10:
        return -np.inf
    return 0


def log_posterior(theta, x, y, lines):
    return log_likelihood(theta, x, y, lines) + log_prior(theta)




# Folder with NIRSPEC spectra
data_folder = Path(f'./data')

# CEERs 1027 redshift
z_obj = 7.8334

# Multiple spectra from the same galaxy observed with diference dispensers
spec_list = data_folder.glob('*.fits')
spec_list = list(spec_list)

s = spec_list[1]

spec = lime.Spectrum.from_file(s, 'nirspec', redshift=z_obj)

# Convert units as necessary
spec.unit_conversion(wave_units_out='Angstrom', flux_units_out='FLAM')

# Visualize the spectrum
spec.plot.spectrum(rest_frame=True)

# Compute the FWHM curve for the instrument observation (removing data mask)
wave_obs = spec.wave.data
deltalamb_arr = np.diff(wave_obs)
R_arr = wave_obs[1:] / deltalamb_arr
FWHM_arr = wave_obs[1:] / R_arr

l = spec.wave.data
f = spec.flux.data

full_df = lime.line_bands(vacuum=True)

# Cropping to the "main lines"
candidate_lines = ["H1_1216A", "He2_1640A", "Ne5_3427A", "O2_3727A", "H1_4342A",
                   "H1_4863A", "O3_4960A", "O3_5008A", "H1_6565A", "S2_6718A",
                   "He1_10833A"]
# candidate_lines = ["O2_3727A", "H1_4863A", "O3_4960A", "O3_5008A", "H1_6565A"]
lines_df = full_df.loc[candidate_lines]
wave_theo = lines_df.wavelength.to_numpy()

z = spec.redshift

log_posterior(z_obj + 0.3, spec.wave.data, spec.flux.data, wave_theo)

sums = []
zs = np.linspace(z_obj - 5, z_obj + 5, 10000)
for z in zs:
    sums.append(calculate_sum(spec.wave.data, spec.flux.data, wave_theo * (1 + z)))

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(zs, sums)
ax.axvline(z_obj, color='r', linestyle='--')

ax.set_xlabel('Redshift')
ax.set_ylabel('Metric')

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(l, f, alpha=0.5)
plt.show()

running_sum = 0
for w in wave_theo:
    corrected_w = w * (1 + spec.redshift + 0.0)
    w_idx = wavelength_to_index(spec.wave.data, corrected_w)
    if w_idx is None:
        continue
    running_sum += f[w_idx]

    ax.axvline(corrected_w, color='grey', linestyle='--', zorder=0, alpha=0.5)

ax.set_ylabel("Intensity", fontsize=14, labelpad=10)
ax.set_xlabel("Observed Wavelength (Angstrom)", fontsize=14, labelpad=10)
ax.tick_params(axis='both', which='major', labelsize=12)
plt.show()
