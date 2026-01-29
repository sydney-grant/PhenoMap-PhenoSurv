PhenoMap-PhenoSurv framework leverages genomic data and explainable deep learning to identify phenotype, pathway, and gene-level prognostic markers for precision oncology.
 
<img width="468" height="456" alt="image" src="https://github.com/user-attachments/assets/0c06b470-508e-4a6b-83a3-fb650dbc3d5b" />


In this repository:

-> PhenoMap: R code used to train PhenoMap models based on TCGA pan-cancer data. Pretrained models can be downloaded from OSF (https://osf.io/94e32/)
-> PhenoMapper: R Software package for downloading and using pretrained PhenoMap models to calculate PhenoMap scores on genomic data. Vignette with executable examples located in folder.
-> PhenoSurv: Python code for training and interpreting PhenoSurv models. Example dataset located in data/ folder for executable example with code.
-> Downstream Analyses & Manuscript Figures: all R code used for downstream analyses with PhenoMap and PhenoSurv models and code for generating manuscript figures

