import pandas as pd

# Citește fișierul
df = pd.read_csv('/home/alin/Desktop/ContinualLearning/datasets/French/french_anonymized_2/val.csv')

# Convertește label la numeric, valorile non-numerice devin NaN
df['label'] = pd.to_numeric(df['label'], errors='coerce')

# Șterge rândurile cu NaN în coloana label
df = df.dropna(subset=['label'])

# Convertește înapoi la int
df['label'] = df['label'].astype(int)

# Salvează
df.to_csv('/home/alin/Desktop/ContinualLearning/datasets/French/french_anonymized_2/val.csv', index=False)

print(f"Rânduri rămase: {len(df)}")