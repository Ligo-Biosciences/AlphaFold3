
# AlphaFold3 Open-Source Implementation

## Introduction
Welcome to the open-source implementation of AlphaFold3, an ongoing research project aimed at advancing the field of biomolecular structure prediction. This repository is in its early stages and is intended to foster collaboration and innovation within the scientific community.


## Acknowledgments
This project stands on the shoulders of giants. We are deeply indebted to:

- The original AlphaFold team at Google DeepMind for their groundbreaking work.

- The OpenFold project, which laid the foundation for open-source protein structure prediction.

- The ProteinFlow library, especially the contributions of Liza Kozlova (@elkoz), which has been instrumental in our data processing pipeline.


## Project Status

This is an active research project in its early phases. We are working diligently to prepare a stable release for the community. While we're excited about the potential of this work, we want to emphasize that this is not yet a production-ready tool.


## Getting Started


### Prerequisites

- Python 3.9+

- PyTorch 1.12+

- CUDA-compatible GPU (recommended)


### Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-org/alphafold3-open.git
   cd alphafold3-open
   ```


2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```


3. Download the necessary databases:
   ```
   bash scripts/download_alphafold_dbs.sh /path/to/download/directory
   ```


## Usage

For now, the primary use of this repository is for research and development. We will include more user-facing functionality in the future once the ligand-protein and nucleic acid prediction capabilities are ready.


## Contributing

We welcome contributions from the community! If you're interested in contributing, please:

1. Fork the repository

2. Create a new branch for your feature / corrections

3. Submit a pull request with a clear description of your changes


## Citation
@misc{alphafold3_open,
  author = {Your Team},
  title = {Ligo's AlphaFold3 Open-Source Implementation},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Ligo-Biosciences/AlphaFold3}}
}


## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.