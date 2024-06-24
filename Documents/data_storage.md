# Dataset storage structure

The internal and external datasets are stored in the `Dataset` folder according to the following structure. 

- The labeled data from the internal dataset is referred to as **Subset 1**.
- The labeled data from the external dataset is referred to as **Subset 2**.
- The unlabeled data is referred to as **Subset 3**.

Please refer to [data information](https://github.com/BYO-UPM/Craneal_CT/blob/main/Documents/dataset_information.pdf) for more details.

```bash
Dataset/
├── Data DICOM
│   ├── External Dataset
│   │   ├── P13
│   │   │   ├── Corte_Dicom_01.DCM
│   │   │   ├── Corte_Dicom_02.DCM
│   │   │   ├── ...
│   │   ├── P14
│   │   ├── ...
│   ├── Internal Dataset
│   │   ├── P01
│   │   ├── P02
│   │   ├── ...
├── Labeled Data PNG
│   ├── External Dataset
│   │   ├── P13
│   │   │   ├── Manual Mask
│   │   │   │   ├── P13_mask_083.png
│   │   │   │   ├── P13_mask_085.png
│   │   │   │   ├── ...
│   │   │   ├── Original CT
│   │   │   │   ├── P13_original_083.png
│   │   │   │   ├── P13_original_085.png
│   │   │   │   ├── ...
│   │   ├── P16
│   │   ├── ...
│   ├── Internal Dataset
│   │   ├── P01
│   │   │   ├── Manual Mask
│   │   │   │   ├── P01_mask_01.png
│   │   │   │   ├── P01_mask_02.png
│   │   │   │   ├── ...
│   │   │   ├── Original CT
│   │   │   │   ├── P01_original_01.png
│   │   │   │   ├── P01_original_02.png
│   │   │   │   ├── ...
│   │   ├── P02
│   │   ├── ...
├── Unlabeled Data PNG
│   ├── P10
│   │   │   ├── img01.png
│   │   │   ├── img02.png
│   │   │   ├── ...
│   ├── P11
│   ├── P12
│   ├── P14
│   ├── ...
```
