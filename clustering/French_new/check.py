import pandas as pd

# Încarcă datele
df = pd.read_csv('/home/alin/Desktop/ContinualLearning/datasets/French/french_anonymized_2/test_1.csv')

print("Label column dtype:", df['label'].dtype)
print("\nUnique labels:")
print(df['label'].unique())

# Verifică labels non-numerice
non_numeric = df[~df['label'].astype(str).str.match(r'^\d+$')]
print(f"\nNon-numeric labels: {len(non_numeric)}")
print(non_numeric.head(10))

# Verifică CSV structure
print("\nFirst few rows:")
print(df.head())
print("\nColumns:", df.columns.tolist())