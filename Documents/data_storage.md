# Dataset storage structure

The internal and external datasets are stored in the `Dataset` folder according to the following structure. 
- The labeled data from internal dataset is Subset 1
- The labeled data from external dataset is Subset 2
- The unlabeled data is Subset 3

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
