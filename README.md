
## Biology-Aware Mutation-based Deep Learning for Outcome Prediction of Cancer Immunotherapy with Immune Checkpoint Inhibitors 

Full data can be downloaded from https://www.cbioportal.org/ for the mskcc dataset

In this example, NSCLC patients data are loaded for illustration purposes, which is included in 'data.pt' file. 

Each patient is stored in a 'Data' structure, where Data.x is the mutation array, Data.edges is from the PPI database indicating gene interactions, and Data.y is the ground truth (first column is the survival status and second column is the overall survival months).
