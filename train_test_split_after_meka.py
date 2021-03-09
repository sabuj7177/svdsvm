import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def train_test_split_data(dataset_name, output_path):
    df = pd.read_csv(dataset_name, header=None)
    print("Data read complete")
    col_num = df.shape[1] - 1
    X = df.iloc[:, :col_num]
    y = df.iloc[:, -1]
    X = X.to_numpy()
    y = y.to_numpy()
    y = y.reshape((len(y), 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("train test split complete")

    concatenated_train = np.concatenate((X_train, y_train), axis=1)
    pd.DataFrame(concatenated_train).to_csv(output_path+"/"+output_path+"_train.csv", index=False, header=False)
    print("Train data save complete")

    concatenated_test = np.concatenate((X_test, y_test), axis=1)
    pd.DataFrame(concatenated_test).to_csv(output_path+"/"+output_path+"_test.csv", index=False, header=False)


if __name__ == '__main__':
    # train_test_split_data("covtype/covtype_after_meka.csv", "covtype")
    # train_test_split_data("webspam/webspam_after_meka.csv", "webspam")
    train_test_split_data("susy/susy_after_meka.csv", "susy")