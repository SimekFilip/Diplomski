import pandas as pd

# input_data = pd.read_csv("../data/NYSE_prices_daily_1990-2004.csv")
# input_data = pd.read_csv("../data/Google.csv")
# C:\Filip\FER\3.GODINA\6.SEMESTAR\ZAVRŠNI RAD\MeanVarPortofolio\data
input_data = pd.read_csv(r"C:\Filip\FER\3.GODINA\6.SEMESTAR\ZAVRŠNI RAD\MeanVarPortofolio\data\Google.csv")
print(len(input_data))
input_data = input_data.values[:, 1:]
input_target = [1 for i in range(len(input_data))]
print(len(input_target))
print(input_target)
print(input_data)
print("--------------------")
train_input = input_data[0:1000, :]
test_input = input_data[1000:, :]
print(train_input)
print(test_input)

train_target = [1 for i in range(len(train_input))]
test_target = [1 for i in range(len(test_input))]
print(len(train_target))
print(len(test_target))
print(train_target)
print(test_target)