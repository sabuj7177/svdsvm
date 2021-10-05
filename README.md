# SVDSVM

This work provides a method(SVDSVM) to compute linear support vector classification (SVM) distributedly and efficiently. Our algorithm uses singular value decomposition (SVD) for matrix factorization purposes. After matrix factorization, we apply the dual ascent method to train the final SVM classifier. Using stochastic SVD instead of householder QR decomposition, this method reduces the algorithm's per iteration time, resulting in an overall shorter training time compared to QRSVM, a fast distributed SVM algorithm. Our method improves the overall time complexity of the algorithm, and its storage and communication complexity is similar to QRSVM. We provide an implementation of our algorithm using openMPI and Armadillo library. Our evaluation with benchmark datasets shows that our method reduces per iteration time of dual ascent step around 5x compared to QRSVM, resulting in an overall 50% time reduction. However, singular value decomposition is not entirely lossless. It results in a small accuracy drop which we found around 0.5-1.5% for different datasets. Thus, our method provides a significant performance boost by sacrificing small accuracy compared to QRSVM.


### This is the official source code of SVDSVM. Ubuntu system is necessary for running this project.

## Installation guide: 

Run this commands to make the environment of this project.

    sudo apt-get update
    sudo apt-get -y install build-essential
    sudo snap install cmake --classic
    sudo apt-get -y install libboost-dev
    sudo apt -y install libblas-dev liblapack-dev
    wget http://sourceforge.net/projects/arma/files/armadillo-9.900.3.tar.xz
    tar -xvf armadillo-9.900.3.tar.xz
    cd armadillo-9.900.3
    ./configure
    make
    sudo make install
    cd ..
    sudo apt-get -y install libopenmpi-dev

Download covtype, webspam, susy from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html

Run `data_generation_for_meka.py` to convert the data into dense format. Run MEKA code on the csv generated file to get 
new csv file. Run `train_test_split_after_meka.py` for splitting the data into train and test set. 

Run this command to distribute the data.

    ./distribute.sh <dataset_name> <data_num> <col_num> <processor>
Example:

    ./distribute.sh susy_train.csv 4000000 128 64

Compile:

    mpicxx -o  MPIexecV1 dist_MPI_SVDSVM_MEKA_V2.cpp -larmadillo -std=c++14

Run:

    mpirun -np <processor> ./MPIexecV1 1 0.1 0.001 covtype
