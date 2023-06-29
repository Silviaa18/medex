# Medex Disease Prediction
Welcome to the documentation for our Disease Risk Prediction script. This script forms a core part of a larger system designed to predict the risk of various diseases for patients, based on their medical history, demographic information, and other relevant factors. 

The current script focuses on the prediction of three major health conditions: Diabetes, Coronary Heart Disease (CHD), and Stroke. For diabetes and CHD, it utilizes machine learning models trained on relevant clinical and demographic data. The risk of stroke, on the other hand, is assessed using the CHADSVASc scoring system, a widely accepted clinical approach to estimating stroke risk in patients with atrial fibrillation.

At a high level, the script retrieves patient data from a PostgreSQL database, processes this data in line with the requirements of each disease prediction algorithm, and then applies the machine learning models or scoring system as appropriate. The calculated risk scores are then stored back in the database, available for further analysis and review.

# ICD-10 Mapping:

The following is a mapping of ICD-10 codes to the diseases used in the code's plugin classes. These codes represent specific medical conditions or diagnoses that are relevant for calculating the risk scores. The mapping helps identify patients with these conditions in the database and factor them into the risk calculation process.


1. CHD Prediction Plugin:

    Hypertension: ICD-10 code range I10.
   
    Heart Disease: ICD-10 codes in the range I200 to I589.
  
3. CHADSVAScPlugin:

    Hypertension: ICD-10 code range I10.
  
    Congestive Heart Failure: ICD-10 code I500.
  
    Diabetes: ICD-10 codes in the range E100 to E149.
  
    Previous Stroke/Transient Ischemic Attack/Thrombus: ICD-10 codes in the range I600 to I699, G458, G459, and I800 to I809.
  
    Atrial Fibrillation: ICD-10 codes in the range I480 to I483, and I48.
  
    Vascular Disease: ICD-10 codes in the range I700 to I799.

# Missing features:

- Logging: Current error handling and logging are basic, relying primarily on print statements. In a more production-ready version, a structured logging system that writes to files or sends logs to a monitoring service would be beneficial.

- Modularity and Scalability: The system is modular and extendable thanks to the plugin interface. However, as the number of plugins grows, loading all plugins at startup could become expensive. Consider loading plugins on-demand or implementing a more sophisticated plugin management system.

- Testing: The current codebase lacks testing. Unit tests and integration tests are crucial in maintaining code quality and catching bugs early.

- Error Handling: The error handling is basic and allows for continued execution even if one row fails. However, the cause of the error isn't investigated or corrected. In a production system, more sophisticated error handling and recovery could be introduced.

# Limitations:

- Data Cleaning and Preprocessing: The scripts assume that data in the database is clean and ready to use. In reality, there may be need for more comprehensive data cleaning and preprocessing steps.

- Performance: For large databases and complex queries, performance may become an issue. The script does not currently include any optimizations for dealing with large volumes of data, such as batch processing, parallelization, or the use of more performant data structures.

- Hardcoded Values: Certain values, such as column names and model paths, are hardcoded. This could make the system less flexible and harder to maintain as changes require code modifications.

- Model Management: The script assumes the models are present at certain file paths. A more production-ready application might include a more robust system for model management, versioning, and updating.
