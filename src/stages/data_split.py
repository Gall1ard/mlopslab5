import pandas as pd
from sklearn.model_selection import train_test_split

def data_split():
    # splitting final dataset
    df = pd.read_csv("./data/final_iris.csv")
    train_dataset, test_dataset = train_test_split(df,
                                                   test_size=0.3,
                                                   random_state=42)
    
    train_dataset.to_csv("./data/train_iris.csv", index=False)
    test_dataset.to_csv("./data/test_iris.csv", index=False)

if __name__ == "__main__":
    data_split()