import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/hp_obp.csv")

jump_col = "jump_height_(imp-mom)_[cm]_mean_cmj"
bat_col = "bat_speed_mph"
level_col = "playing_level"

df_clean = df[[jump_col, bat_col, level_col]].dropna()

df_clean[jump_col] = pd.to_numeric(df_clean[jump_col], errors="coerce")
df_clean[bat_col] = pd.to_numeric(df_clean[bat_col], errors="coerce")

df_clean = df_clean.dropna()

corr = df_clean[jump_col].corr(df_clean[bat_col])
print("Overall Correlation (r):", round(corr, 3))
print("R^2:", round(corr**2, 3))

levels = df_clean[level_col].unique()

plt.figure()

for level in levels:
    subset = df_clean[df_clean[level_col] == level]
    plt.scatter(subset[jump_col], subset[bat_col], label=level)

# Regression line (overall)
x = df_clean[jump_col]
y = df_clean[bat_col]
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.plot(x, poly1d_fn(x))

plt.xlabel("Jump Height (cm)")
plt.ylabel("Bat Speed (mph)")
plt.title("Jump Height vs Bat Speed by Playing Level")
plt.legend()
plt.show()
