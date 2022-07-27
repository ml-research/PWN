# Predictive Whittle Networks for Time Series

This repository is the official implementation of Predictive Whittle Networks (PWN) introduced in the paper titled [Predictive Whittle Networks for Time Series](https://ml-research.github.io/papers/yu2022whittle.pdf) by Zhongjie Yu, Fabrizio Ventola, Nils Thoma, Devendra Singh Dhami, Martin Mundt, and Kristian Kersting, published at UAI 2022.

It evolved from the implementation of the seminal work named "RECOWNs" [1], presented at the 4th Workshop on Tractable Probabilistic Modeling (TPM 2021) @ UAI 2021. 

The repository depends solely on PyTorch.
It also contains PyTorch implementations of other existing methods, e.g., Whittle SPNs [2], Conditional SPNs [3], and Spectral RNNs [4].

The employed dataset "Retail" and its related information have been removed from the repository due to a confidentiality agreement (NDA). 
Please, do not hesitate to open an issue or to write an e-mail to <nthoma@nilsthoma.de> if this causes issues in reproducing the results.

###### [1] Thoma, N.; Yu, Z.; Ventola, F.; and Kersting, K. RECOWNs: Probabilistic Circuits for Trustworthy Time Series Forecasting. 4th Workshop on Tractable Probabilistic Modeling (TPM 2021) @ UAI 2021.

###### [2] Yu, Z.; Ventola, F.; and Kersting, K. Whittle Networks: A Deep Likelihood Model for Time Series. International Conference on Machine Learning (ICML 2021).

###### [3] Shao, X.; Molina, A.; Vergari, A.; Stelzner, K.; Peharz, R.; Liebig, T.; and Kersting, K. Conditional sum-product networks: Imposing structure on deep probabilistic architectures. International Conference on Probabilistic Graphical Models (PGM 2020).

###### [4] Wolter, M.; Gall, J.; and Yao, A. Sequence Predic-tion Using Spectral RNNs. International Conference on Artificial Neural Networks. 2020.


## Setup

This will clone the repository, install a Python virtual environment (requires Python 3.8), and the required packages.

    git clone https://github.com/ml-research/PWN.git
    cd PWN
    ./setup.sh

## Demos

### Activate the virtual environment:

    source ./venv_pwn/bin/activate

### To run an experiment on the data set Power

    python training.py

### To run a demo of a long-range prediction

First, you need to download "data_cache_pwr_long.pkl" data from [TU datalib](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/3532/):

    wget https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/3532/data_cache_pwr_long.pkl -P res

and then launch the script:

    python long_term_prediction_showcase.py


## Citation
If you find this code useful for your research, please consider citing:


    @inproceedings{yu2022whittle,
      title = {Predictive Whittle Networks for Time Series},
      author = {Zhongjie Yu and Fabrizio Ventola and Nils Thoma and Devendra Singh Dhami and Martin Mundt and Kristian Kersting},
      booktitle = { Proceedings of the 38th Conference on Uncertainty in Artificial Intelligence (UAI)},
      pages = {},
      year = {2022}
    }
    


## Acknowledgments

* This work was supported by the Federal Ministry of Education and Research (BMBF; project "MADESI", FKZ 01IS18043B, and Competence Center for AI and Labour; "kompAKI", FKZ 02L19C150), the ICT-48 Network of AI Research Excellence Center "TAILOR" (EU Horizon 2020, GA No 952215), the project "safeFBDC - Financial Big Data Cluster" (FKZ: 01MK21002K), funded by the German Federal Ministry for Economics Affairs and Energy as part of the GAIA-x initiative and the Collaboration Lab "AI in Construction" (AICO). It benefited from the Hessian Ministry of Higher Education, Research, Science and the Arts (HMWK; projects "The Third Wave of AI" and "The Adaptive Mind") and the Hessian research priority programme LOEWE within the project "WhiteBox". The authors thank German Management Consulting GmbH for supporting this work.
