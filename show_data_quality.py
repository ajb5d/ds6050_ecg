import pandas as pd

datasets = ['train', 'test', 'val']

result = []
for d in datasets:
    dat = pd.read_csv(f"data/{d}_data_quality.csv")
    dat['source'] = d
    result.append(dat)

dat = pd.concat(result)

for col in dat.columns:
    if dat[col].dtype == "bool":
        m = dat[col].mean()
        print(f"{col:10}       {m*100:0.2f}%")

