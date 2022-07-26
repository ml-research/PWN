# Predictive Whittle Networks for Time Series

This repository is the official implementation of Predictive Whittle Networks (PWN) introduced in [Predictive Whittle Networks for Time Series](https://ml-research.github.io/papers/yu2022whittle.pdf) by Zhongjie Yu, Fabrizio Ventola, Nils Thoma, Devendra Singh Dhami, Martin Mundt, and Kristian Kersting, published at UAI 2022.

This repository is developed based on the code for RECOWNs [1].
The repository depends solely on PyTorch.
Therefore, it also contains PyTorch implementations of existing methods, e.g. CSPNs [2] or Spectral RNNs [3].

Due to a confidentiality agreement concerning the involved retail data, this data and every related information has been removed from the repository beforehand.
If this causes any problems when reproducing the results, please do not hesitate and open up an issue or write an E-Mail: <nthoma@nilsthoma.de>.

## Setup

This will clone the repo, install a Python virtual env (requires Python 3.6), and the required packages.

    git clone https://github.com/ml-research/PWN.git
    cd PWN
    ./setup.sh

## Demos

### Activate the virtual environment:

    source ./venv_pwn/bin/activate

### To run an experiment on the data set Power

    python training.py

### To run a showcase of long range prediction

Download "data_cache_pwr_long.pkl" data from [TU datalib](https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/3532/):

    wget https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/3532/data_cache_pwr_long.pkl -P res
    python long_term_prediction_showcase.py


## Citation
If you find this code useful in your research, please consider citing:


    @inproceedings{yu2022whittle,
      title = {Predictive Whittle Networks for Time Series},
      author = {Zhongjie Yu and Fabrizio Ventola and Nils Thoma and Devendra Singh Dhami and Martin Mundt and Kristian Kersting},
      booktitle = { Proceedings of the 38th Conference on Uncertainty in Artificial Intelligence (UAI)},
      pages = {},
      year = {2022}
    }
    

\
[1] Thoma, N.; Yu, Z.; Ventola, F.; and Kersting, K. RECOWNs: Probabilistic Circuits for Trustworthy Time SeriesForecasting. 4th Workshop on Tractable Probabilistic Modeling (TPM 2021).

[2] Shao, X., Molina, A., Vergari, A., Stelzner, K., Peharz, R., Liebig, T., & Kersting, K. Conditional sum-product networks: Imposing structure on deep probabilistic architectures. In International Conference on Probabilistic Graphical Models (PGM 2020).

[3] Wolter, M.; Gall, J.; and Yao, A. Sequence Predic-tion Using Spectral RNNs. In International Conference on Artificial Neural Networks. 2020.

## Acknowledgments

* This work was supported by the Federal Ministry of Education and Research (BMBF; project "MADESI", FKZ 01IS18043B, and Competence Center for AI and Labour; "kompAKI", FKZ 02L19C150), the ICT-48 Network of AI Research Excellence Center "TAILOR" (EU Horizon 2020, GA No 952215), the project "safeFBDC - Financial Big Data Cluster" (FKZ: 01MK21002K), funded by the German Federal Ministry for Economics Affairs and Energy as part of the GAIA-x initiative and the Collaboration Lab "AI in Construction" (AICO). It benefited from the Hessian Ministry of Higher Education, Research, Science and the Arts (HMWK; projects "The Third Wave of AI" and "The Adaptive Mind") and the Hessian research priority programme LOEWE within the project "WhiteBox". The authors thank German Management Consulting GmbH for supporting this work.
