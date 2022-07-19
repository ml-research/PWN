# Predictive Whittle Networks for Time Series

This repository is the official implementation of Predictive Whittle Networks (PWN) introduced in [Predictive Whittle Networks for Time Series](https://ml-research.github.io/papers/yu2022whittle.pdf) by Zhongjie Yu, Fabrizio Ventola, Nils Thoma, Devendra Singh Dhami, Martin Mundt, and Kristian Kersting, published at UAI 2022.

## Setup

This will clone the repo, install a Python virtual env (requires Python 3.6), and the required packages.

    git clone https://github.com/ml-research/PWN.git
    ./setup.sh

## Execution

### Activate the virtual environment:

    source ./venv_pwn/bin/activate

To run an experiment on the data set Power, execute

    training.py

## Citation
If you find this code useful in your research, please consider citing:


    @inproceedings{yu2022whittle,
      title = {Predictive Whittle Networks for Time Series},
      author = {Zhongjie Yu and Fabrizio Ventola and Nils Thoma and Devendra Singh Dhami and Martin Mundt and Kristian Kersting},
      booktitle = { Proceedings of the 38th Conference on Uncertainty in Artificial Intelligence (UAI)},
      pages = {},
      year = {2022}
    }

## Acknowledgments

* This work was supported by the Federal Ministry of Education and Research (BMBF; project "MADESI", FKZ 01IS18043B, and Competence Center for AI and Labour; "kompAKI", FKZ 02L19C150), the ICT-48 Network of AI Research Excellence Center "TAILOR" (EU Horizon 2020, GA No 952215), the project "safeFBDC - Financial Big Data Cluster" (FKZ: 01MK21002K), funded by the German Federal Ministry for Economics Affairs and Energy as part of the GAIA-x initiative and the Collaboration Lab "AI in Construction" (AICO). It benefited from the Hessian Ministry of Higher Education, Research, Science and the Arts (HMWK; projects "The Third Wave of AI" and "The Adaptive Mind") and the Hessian research priority programme LOEWE within the project "WhiteBox". The authors thank German Management Consulting GmbH for supporting this work.
