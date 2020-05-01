## DeepBiomarker
### Deep Mining and Visualization of MS Spectra for Cancer Associated Biomarker Discovery  
The deep learning model we proposed has a good performance compared with other machine learning methods and may help researchers to find the potential biomarkers which are likely to be missed by the traditional strategy.
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
#### Visualization
Script files for classification results visualization  
1.Grad-Cam  
2.SHAP
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
(Email: [changchengbio@gmail.com](mailto:changchengbio@gmail.com)).
