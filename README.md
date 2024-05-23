# ALIFE
device: The device where the model is trained. For the convenience of the experiment, it is best to specify a GPU device.  
device_cpu: Device used during non training stages, default CPU device, generally do not need to be changed.  
expr_train_addr: The specific address where the gene expression data is located. This file is a txt file, with line names representing sample IDs and column names representing gene symbols.  
clin_train_addr: The file address for clinical data. The row name represents the sample ID, with the first column representing the outcome and the second column representing the survival time.  



