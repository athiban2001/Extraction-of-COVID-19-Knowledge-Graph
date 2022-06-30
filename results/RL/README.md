# Representation Learning Results

## OpenKE installation instructions

1. Install [PyTorch](https://pytorch.org/get-started/locally/)

2. Change current directory to this directory
```
cd results/RL
```

3. Clone the OpenKE-PyTorch branch:
```
git clone -b OpenKE-PyTorch https://github.com/thunlp/OpenKE --depth 1
cd OpenKE
cd openke
```

4. Compile C++ files
```
bash make.sh
```

## Dataset Folder

dataset folder contains the results from running the representation learning which will be used for TransD training.