# MechFinder

Implementation of MechFinder developed by Prof. Yousung Jung group at Seoul National University (contact: yousung@gmail.com)<br>


## Open-Source Release Notice

This is the open-source release of MechFinder. Some core functions are not available in this version and will raise `NotImplementedError` when called. For access to the full implementation, please contact the developers for commercial licensing.

## Contents

- [Open-Source Release Notice](#open-source-release-notice)
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

**Note:** The following expected results require the full implementation. The open-source release will raise `NotImplementedError` for core functions.

### Single reaction labeling
```python
from MechFinder import MechFinder
finder = MechFinder(collection_dir='collections')

updated_reaction, LRT, MT_class, electron_path = finder.get_electron_path(sampled_rxn)
```

Expected output:
```
Identified mechanistic class: nucleophilic_attack_to_(thio)carbonyl_or_sulfonyl
Generated mechanism: [(26, 1), ([1, 9], 9), (9, [9, 1]), ([1, 101], 101)]
```

### Dataset labeling
```python
for rxn in tqdm(dataset_rxns['reaction'], total=len(dataset_rxns)):
    updated_rxn, LRT, MT_class, electron_path = finder.get_electron_path(rxn)
```

Expected output:
```
100%|█████████████████████████████████████████| 33099/33099 [04:46<00:00, 115.42it/s]
Labeled 31364 reactions.
```

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
This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**.

For commercial licensing inquiries, please contact Shuan Chen (shuan.micc@gmail.com).
