import numpy as np
import pandas as pd


def scale(X):
    nom = (X-X.min(axis=0))
    denom = X.max(axis=0) - X.min(axis=0)
    # denom[denom==0] = 1
    return nom/denom


def make_sparse_matrix_to_dense(file_name, dataset_name, data_num, col_num, delimiter):
    input_file = open(dataset_name + "/" + file_name, "r+")
    X = np.zeros((data_num, col_num))
    y = np.zeros((data_num, 1))
    count = 0

    for i in range(data_num):
        data_line = input_file.readline().strip()
        split_row = data_line.split(delimiter)
        y[count, 0] = float(split_row[0])
        if dataset_name == 'covtype' or dataset_name == 'susy':
            if y[count, 0] != 1:
                y[count, 0] = -1
        data_len = len(split_row)
        for j in range(data_len - 1):
            temp = split_row[j + 1]
            split_by_colon = temp.split(':')
            X[count, int(split_by_colon[0]) - 1] = float(split_by_colon[1])
        count += 1

    if dataset_name == 'susy':
        X = scale(X)
    concatenated_data = np.concatenate((X, y), axis=1)
    pd.DataFrame(concatenated_data).to_csv(dataset_name+"/"+dataset_name+"_dense.csv", header=False, index=False)


if __name__ == '__main__':
    # make_sparse_matrix_to_dense("covtype.libsvm.binary.scale", "covtype", 581012, 54, ' ')
    # make_sparse_matrix_to_dense("webspam_wc_normalized_unigram.svm", "webspam", 350000, 254, ' ')
    make_sparse_matrix_to_dense("SUSY", "susy", 5000000, 18, ' ')
