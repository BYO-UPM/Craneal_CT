# Automatic semantic segmentation of the osseous structures of the paranasal sinuses
This repository contains the implementation of the automatic semantic segmentation model for the osseous structures of the paranasal sinuses, as described in our [paper](https://github.com/BYO-UPM/Craneal_CT). The model aims to assist in robotic-assisted surgeries by accurately delimiting critical anatomical structures, using U-Net based architectures enhanced with semi-supervised learning techniques.

<div align="center">
    <img src="https://github.com/BYO-UPM/Craneal_CT/blob/main/Documents/Figures/seg_example.gif" width="80%" alt="Segmentation Example">
</div>

## Abstract
Endoscopic sinus and skull base surgeries require the use of precise neuronavigation techniques, which may take advantage of accurate delimitation of surrounding structures. This delimitation is critical for robotic-assisted surgery procedures to limit volumes of no resection. In this respect, accurate segmentation of the osseous structures surrounding the paranasal sinuses is a relevant issue to protect critical anatomic structures during these surgeries. Currently, manual segmentation of these structures is a labour-intensive task and requires expertise, often leading to inconsistencies. This is due to the lack of publicly available automatic models specifically tailored for the automatic delineation of the complex osseous structures surrounding the paranasal sinuses. To address this gap, we introduce an open-source data/model for the segmentation of these complex structures. The initial model was trained on nine complete ex vivo CT scans of the paranasal region and then improved with semi-supervised learning techniques. When tested on an external data set recorded under different conditions and with various scanners, it achieved a DICE score of 94.82&plusmn;0.9. These results underscore the effectiveness of the model and its potential for broader research applications. By providing both the dataset and the model publicly available, this work aims to catalyse further research that could improve the precision of clinical interventions of endoscopic sinus and skull-based surgeries.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Available models](#available-models)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## Installation

### Development Environment
- Python 3.10+
- For 2D architectures: PyTorch 2.2.1, Segmentation-models-pytorch 0.3.3
  
  Use `conda env create -f segmentation2D.yml` to create the 2D segmentation enviornment.
- For 3D architectures: TensorFlow == 2.8, Keras == 2.8, Keras_applications 1.0.8, Segmentation-models-3D 1.0.8

  Use `conda env create -f segmentation3D.yml` to create the 2D segmentation enviornment.

  (Note: After the updating in Dec. 2023, CUDA 12.2 doesn't support TensorFlow and Keras 2.8 anymore, the codes can run but GPU not working. There are two methods to train the 3D architectures with GPU in this case: (1) Use `apt update && apt install cuda-11-8` to get previous CUDA version; (2) In Google Colab: Use Colab’s fallback runtime version by selecting "Use fallback runtime version" command when connecting to a runtime from the Command Palette.)


For 2D architectures: PyTorch 2.2.1, Segmentation-models-pytorch 0.3.3

For 3D architectures: TensorFlow == 2.8, Keras == 2.8, Keras_applications 1.0.8, Segmentation-models-3D 1.0.8

After the updating in Dec. 2023, CUDA 12.2 doesn't support TensorFlow and Keras 2.8 anymore, the codes can run but GPU not working. There are two methods to train the 3D architectures with GPU in this case:

(1) Use `apt update && apt install cuda-11-8` to get previous CUDA version

(2) In Google Colab: Use Colab’s fallback runtime version by selecting "Use fallback runtime version" command when connecting to a runtime from the Command Palette.

### Setup
To set up the codes, run the following commands:
```bash
git clone https://github.com/BYO-UPM/Craneal_CT.git
cd Craneal_CT
```

## Dataset



## Available models


## Contributing
Contributions are welcome! Please follow the standard fork-and-pull request workflow on GitHub.

If you use our model or dataset in your research, please cite our paper:
```bash
@article{yichun2024automatic,
  title={Automatic Semantic Segmentation of the Osseous Structures of the Paranasal Sinuses},
  author={Yichun Sun, Alejandro Guerrero-López, Julián D. Arias-Londoño, Juan I. Godino-Llorente},
  journal={xxx},
  year={2024},
  volume={XX},
  number={XX},
  pages={xx-xx}
}
```
## Acknowledgments
This research was funded by an agreement between Comunidad de Madrid (Consejería de Educación, Universidades, Ciencia y Portavocía) and Universidad Politécnica de Madrid, to finance research actions on SARS-CoV-2 and COVID-19 disease with the REACT-UE resources of the European Regional Development Funds. This work was also supported by the Ministry of Economy and Competitiveness of Spain under Grants PID2021-128469OB-I00 and TED2021-131688B-I00, and by Comunidad de Madrid, Spain. Universidad Politécnica de Madrid supports J. D. Arias-Londoño through a María Zambrano grant, 2022. The authors also thank the Madrid ELLIS unit (European Laboratory for Learning & Intelligent Systems) for its indirect support.



