import src.DataLoader as dl
data = dl.DataLoader("data")
dataset = data.loadAllDataCsv()
print(dataset["AAPL"].head())