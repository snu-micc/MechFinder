# MechFinder

Implementation of MechFinder developed by Prof. Yousung Jung group at Seoul National University (contact: yousung@gmail.com)<br>


## Contents

- [Developer](#developer)
- [Python Dependencies](#python-dependencies)
- [Installation Guide](#installation-guide)
- [Mechanism labeling](#mechanism-labeling)
- [Publication](#publication)
- [License](#license)

## Developer
Shuan Chen (shuan75@snu.ac.kr), Ramil Babazade (ramil_babazade@kaist.ac.kr)<br>

## Python Dependencies
* Python (version >= 3.6)
* Numpy (version >= 1.16.4)
* RDKit (version >= 2019)

## Installation Guide
Create a virtual environment to run the code of MechFinder.<br>
```
git clone https://github.com/kaist-amsg/MechFinder.git
cd MechFinder
conda create -c conda-forge -n rdenv python -y
conda activate rdenv
```


## Mechanism labeling
See `Demo.ipynb` for running instructions and expected output for single and multiple reaction mechanism labeling with MechFinder.

## Publication
```
@article{chen2024large,
  title={A large-scale reaction dataset of mechanistic pathways of organic reactions},
  author={Chen, Shuan and Babazade, Ramil and Kim, Taewan and Han, Sunkyu and Jung, Yousung},
  journal={Scientific Data},
  volume={11},
  number={1},
  pages={863},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```
## License
This project is covered under the **MIT License**.
