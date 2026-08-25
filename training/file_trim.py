import pandas as pd

df = pd.read_csv("../data/raw/Amazon_Reviews_20K_records.csv")

df_5000 = df.head(3500)

df_5000.to_csv("../data/raw/Amazon_Reviews_3500records.csv")