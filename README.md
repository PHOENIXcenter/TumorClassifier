## A deep learning-based tumor classifier directly using MS raw data
In summary, we presented a novel deep learning-based classifier for distinguishing between tumor and non-tumor samples, which directly used MS raw data, rather than the identification and quantification results of MS data. It directly extracted all the potential precursors in MS raw data, and then built an accurate deep learning-based classification model using the most representative precursors with the best discrimination power. According to our analysis results on three tumor-associated datasets, the CNN model we proposed performed best on the training data and test data, compared with the other popular machine learning models.

Our work is expected to help researchers to make proper clinical decisions and find potential biomarkers with low abundance and can be used as a complementary strategy compared with the traditional one.

Moreover, our deep learning-based classifier using MS raw data can be applied to other classification problems in proteomics, not merely tumor prediction.
## File structure
#### Data_Preprocessing
Script file for data preprocessing  
1.extract_raw_data  
Extract time, intensity, mz from raw data.  
2.stat_con_keys  
Statistics on the data to get a filtered list.  
3.key_features  
Filter data based on the list obtained in the previous step.  
4.pre_classify
Filtering data with SVM.
#### Data_Classification
Script files for classification model training and testing  
1.data_aug  
Data augmentation.  
2.model  
Model classification.  
3.metrics  
Classification results.

## Run script
The data size and format may not be completely suitable, you should be adjusted according to actual needs before running the code.

You can find the main function to run the submodule.
```
python main.py
```
When you run the code, copy the data to the corresponding folder.
`Placeholder.txt` is a placeholder file, delete it before running code.  
If you need to change different parameters for multiple experiments and tests, please write code as needed.
## Dependencies
sklearn 0.20.0  
keras 2.2.2  
tensorflow 1.11.0  
python 3.5.2  
jupyter 1.0.0  
Ubuntu 16.04 LTS
##  Contact

  If you have any question, please contact [Dr. Cheng Chang](https://orcid.org/0000-0002-0361-2438)![](https://orcid.org/sites/default/files/images/orcid_16x16.png)
(Email: [changchengbio@163.com](mailto:changchengbio@163.com)).
