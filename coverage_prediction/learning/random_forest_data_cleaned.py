import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 📌 Caricamento del dataset
print("Caricamento del dataset...")
df = pd.read_csv('output_data.csv')

# 🔹 1. Unione delle distanze e interpolazione
df['distance'] = df[['Vector_128', 'Vector_222']].mean(axis=1)
df['distance'] = df['distance'].interpolate(method='linear', limit_direction='both')

# 🔹 2. Unione delle coordinate X e interpolazione
df['coord_x'] = df[['Vector_129', 'Vector_223']].mean(axis=1)
df['coord_x'] = df['coord_x'].interpolate(method='linear', limit_direction='both')

# 🔹 3. Unione delle coordinate Y e interpolazione
df['coord_y'] = df[['Vector_130', 'Vector_224']].mean(axis=1)
df['coord_y'] = df['coord_y'].interpolate(method='linear', limit_direction='both')

# 🔹 4. SINR UL e interpolazione
df['SINR'] = df[['Vector_131']].mean(axis=1)
df['SINR'] = df['SINR'].interpolate(method='linear', limit_direction='both')

# 🔹 Verifica della presenza di NaN dopo la pulizia
print("\nNumero di NaN dopo la pulizia:")
print(df[['distance', 'coord_x', 'coord_y', 'SINR']].isna().sum())

# 📊 Grafici più chiari: Prima vs Dopo (a coppie)
fig, axs = plt.subplots(4, 2, figsize=(16, 12))
fig.suptitle("Distribuzione dei Valori nei Vettori - Prima e Dopo la Pulizia", fontsize=14)

time_axis = np.arange(len(df))

# 📌 1. Distanza
axs[0, 0].scatter(time_axis, df['Vector_128'], alpha=0.5, s=2, label='Vector_128')
axs[0, 0].scatter(time_axis, df['Vector_222'], alpha=0.5, s=2, label='Vector_222')
axs[0, 0].set_title("Distanza - Prima della Pulizia")
axs[0, 0].legend()

axs[0, 1].scatter(time_axis, df['distance'], alpha=0.5, s=2, color='r', label='Distanza Pulita')
axs[0, 1].set_title("Distanza - Dopo la Pulizia")
axs[0, 1].legend()

# 📌 2. Coordinata X
axs[1, 0].scatter(time_axis, df['Vector_129'], alpha=0.5, s=2, label='Vector_129')
axs[1, 0].scatter(time_axis, df['Vector_223'], alpha=0.5, s=2, label='Vector_223')
axs[1, 0].set_title("Coordinata X - Prima della Pulizia")
axs[1, 0].legend()

axs[1, 1].scatter(time_axis, df['coord_x'], alpha=0.5, s=2, color='r', label='Coordinata X Pulita')
axs[1, 1].set_title("Coordinata X - Dopo la Pulizia")
axs[1, 1].legend()

# 📌 3. Coordinata Y
axs[2, 0].scatter(time_axis, df['Vector_130'], alpha=0.5, s=2, label='Vector_130')
axs[2, 0].scatter(time_axis, df['Vector_224'], alpha=0.5, s=2, label='Vector_224')
axs[2, 0].set_title("Coordinata Y - Prima della Pulizia")
axs[2, 0].legend()

axs[2, 1].scatter(time_axis, df['coord_y'], alpha=0.5, s=2, color='r', label='Coordinata Y Pulita')
axs[2, 1].set_title("Coordinata Y - Dopo la Pulizia")
axs[2, 1].legend()

# 📌 4. SINR
axs[3, 0].scatter(time_axis, df['Vector_131'], alpha=0.5, s=2, label='Vector_131')
axs[3, 0].set_title("SINR - Prima della Pulizia")
axs[3, 0].legend()

axs[3, 1].scatter(time_axis, df['SINR'], alpha=0.5, s=2, color='r', label='SINR Pulito')
axs[3, 1].set_title("SINR - Dopo la Pulizia")
axs[3, 1].legend()

plt.tight_layout()
plt.show()
