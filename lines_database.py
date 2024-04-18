import lime

# Pandas dataframe with all the:
full_df = lime.line_bands(vacuum=True)

# Cropping to the "main lines"
candidate_lines = ["H1_1216A", "He2_1640A", "Ne5_3427A", "O2_3727A", "H1_4342A",
                   "H1_4863A", "O3_4960A", "O3_5008A", "H1_6565A", "S2_6718A",
                   "He1_10833A"]
lines_df = full_df.loc[candidate_lines]
print(lines_df)

# The theoretical wavelengths (vacuum) are in the first column:
wave_theo = lines_df.wavelength.to_numpy()

# For this galaxy I think only
#"O2_3727A", "H1_4863A", "O3_4960A", "O3_5008A", "H1_6565A" are visible