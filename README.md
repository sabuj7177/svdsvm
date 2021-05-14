# SVDSVM

This is the official source code of SVDSVM. Ubuntu system is necessary for running this project.

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
