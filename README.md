# Project_4_Group_4
Drishti Patel, Priyanshu Rana, Michelle Harris

## Lung Cancer Predictive Modeling

### Objective.
Lung Cancer is the fifth most diagnosed cancer in Australia (cancer.org.au). It is estimated that more than 14, 700 people were diagnosede with lung cancer in 2023.

The purpose of this project is to create an algorithm that can be used to predict lung cancer risk as low, medium or high with a greter than 75% accuracy. This is achived by analysing a lung cancer dataset containing 1000 records sourced from Kaggle "Cancer Patients and air pollution a new link" https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link/data.

#### Questions
1. Can we predict a person's cancer risk with greater than 75% accuracy using a neural network model?

2. What additional risk factors make a person more susceptible to lung cancer? It is hypothesised that smoking (both passive and actual), air pollution exposure and alcohol use may lead to a higher chance of contracting lung cancer. These are explored in an initial tableau analysis. 

#### Models 
1. Logistic Regression
2. Random Forest
3. K-Nearest
4. Decision Tress
5. Neural Network Modeling

#### Dependencies
- Python Pandas
- Python Matplotlib
- Scikit-learn
- Tensorflow
- PySpark
- Tableau
- Google collaboratory 

### Data Model Implementation

The dataset was read into a pandas DataFrame and examined, index and Patient Id columns were removed and the level column identified as the target for our modeling. The low, medium, high level for each patient was then converted from categorical to numerical for analysis.
Tableau was used for an exploratory analysis of the dataset (see tableau_figXXX) where smoking and alcohol use appeared to be contributory factors, the dataset was also displayed using a matplotlib heatmap. 

A series of boxplots was created to determine if there were any outliers in the data, an outlier in the age category was discovered.

The dataset was then parsed to logistic regession, random forest, k-nearest and decision tree modelling for an initial assessment before being scaled using standard scaler and analysed again to compare the results.
