# WP9S2SI_EIM
# Evaluating Machine Learning Imputation Methods Using Employment Earnings Data

In this project, we aim to evaluate ML imputation methods on monthly earnings data from employment.

The data is processed monthly using in-house generic system for data editing and imputations. Erroneous data and item non-response in the monthly earnings data are addressed using a combination of systematic corrections (over 1,000 edit rules) and imputation methods, while unit non-response is also imputed. Currently applied imputation methods include logical, historical, and hot-deck imputations for handling erroneous data, item non-response, and unit non-response. To ensure the suitability of the donor population, generalized linear models (GLMs) with logarithmic transformations of selected variables are applied. 

We believe that ML offers opportunities for developing new imputation techniques, and this project will allow us to explore and learn from them.
Our objectives are to:
-	Evaluate open-source ML imputation methods in terms of accuracy, ad hoc input, and processing time, compared to the existing methods in our generic system.
-	Explore the potential for code parametrization, traceability, and reproducibility of ML-based approaches.
 
The dataset that we work with consists of monthly earnings, their components, and other relevant variables. The observation units are individuals employed by business entities. The dataset provides full population coverage, encompassing approximately 815,000 persons. The two primary data sources, supplemented by several additional administrative and statistical sources, are:
-	Monthly data from the Statistical Register of Employment, maintained by SURS
-	Administrative data from REK-O forms, which are used for calculating withholding tax returns on employee income and are submitted by employers to the Financial Administration of the Republic of Slovenia

In addition to earnings data, the dataset includes contextual variables related to business entities (e.g., activity, institutional sector) and individuals (e.g., age, sex, occupation, educational attainment), which are relevant for both data editing and imputations. The data preparation process included data integration, identification and recalculation of individuals’ main employment, derivation of new variables, and data validation. A training set was derived from input data that remained unchanged during the editing phase for a selected month. Missing values for item and unit non-response are simulated to evaluate performance of methods.  
