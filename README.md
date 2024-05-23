# ALIFE
**You can enter parameters at the beginning of the ALIFE file and run the entire file's code to achieve ALIFE functionality.**  

**device**: The device where the model is trained. For the convenience of the experiment, it is best to specify a GPU device.  
**device_cpu**: Device used during non training stages, default CPU device, generally do not need to be changed.  
**expr_train_addr**: The specific address where the gene expression data is located. This file is a txt file, with line names representing sample IDs and column names representing gene symbols.  
**clin_train_addr**: The file address for clinical data. The row name represents the sample ID, with the first column representing the outcome and the second column representing the survival time.  
**expr_test_addr**: The specific address where the gene expression data is located. This file is a txt file, with line names representing sample IDs and column names representing gene symbols.  
**clin_test_addr**: The file address for clinical data. The row name represents the sample ID, with the first column representing the outcome and the second column representing the survival time.
**mask_addr**: The CSV file where the path information is located. We provide a template for the HALLMARK cancer gene set, which can be directly used as the target object for mask_addr.  
**save_addr**: Save path for model output files.  
**n_pathway_embeding**: The number of pathway embeddings.  
**num_epochs_ae**: The number of epochs required for autoencoder training.  
**batch_size_ae**: The batchsize required for autoencoder training. Too large or too small will result in an error.  
**lr_ae**: Initial learning rate of autoencoder.  
n**um_epochs_sup**: The number of epochs required for supervised module training.  
**batch_size_sup**: The batchsize required for supervised module training. Too large or too small will result in an error.  
**lr_sup**: Initial learning rate of supervised module.  




