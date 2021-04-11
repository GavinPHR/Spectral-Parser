<p align="center">
  <img width="180" src="logo.png">
</p>

<p align="center">
  <img height="20" src="https://img.shields.io/badge/python-%3E%3D3.6-blue"> <img height="20" src="https://img.shields.io/github/license/GavinPHR/Spectral-Parser"> 
</p>

A high-performance implementation of Spectral Learning of Latent-Variable PCFGs [(Cohen et al., 2013)](https://www.aclweb.org/anthology/N13-1015/). This work was done for my undergraduate dissertation at the University of Edinburgh, supervised by Dr. Shay Cohen.

The codebase is only tersely commented. If you want to know more about the algorithm, how it is implemented, and its performance, you should consult my dissertation `dissertation.pdf`. 

### Pre-requisites

Install the requirements:
```
pip3 install -r requirements.txt
```

### Global Configuration File

`spectral_parser/config.py` contains all the configurable variables, including file paths, output directory, hyperparameters etc. You need to configure this file before training/testing.

### Training and Testing

Please run the commands from the `spectral_parser/` directory.

To train:
```
python3 train.py
```
The parameters will be saved to the output directory.

To parse a gold file:
```
python3 test.py
```
The candidate parse file will be saved to the output direcoty and be called `parse.txt`. Note that the progress bar is not indicative of actual progress because the file is parsed in chunks with multi-processing.

### Data

The sanitized PTB WSJ datasets (taken from [Berkeley Neural Parser](https://github.com/nikitakit/self-attentive-parser/tree/master/data)) are at `spectral_parser/data/`.

### Cite this Work

```
@Misc{Spectral-Parser,
    author = {Haoran Peng},
    title = {Spectral Learning of Latent-Variable PCFGs: High-Performance Implementation},
    year = {2021},
    url = "https://github.com/GavinPHR/Spectral-Parser"
}
```
