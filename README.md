# Mobile Category Predection: Project Overview
- Created a tool that estimantes to which your mobile belongs to which catigory.
- Data is collected from [Kaggle](https://www.kaggle.com/iabhishekofficial/mobile-price-classification)
- Select the Best features for the predection.
- Engineered Features from the data and handling the missing values and outliers.
  - Logistic Regression,
  - Decision Tree Classifier,
  - K-Neighbors Classifier,
  - Linear Discriminant Analysis,
  - Navi-Baies Classifier,
  - Support Vector Machine.
-Built a client facing API using Flask.
## Code and Resources Used
**Python Version:** 3.7  
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, flask, json, pickle  
**For Web Framework Requirments:** `pip install -r requirements.txt`
## Data Collection
For this project we collected the data from the [Kaggel](https://www.kaggle.com/iabhishekofficial/mobile-price-classification)
- Because of there are many features we use **SelectKBest** algorithem to select the best features.
## Data Cleaning
After the data is collected from the Kaggle, I need to clean it up so that it was usable for our model. I made the the following changes:
- Finding and handling the missing values.
- Selecting the Best Features for the model Building.
- Droping the unusable data.
## EDA
- We had the best data. By using the data visulization we find the relation between the collumns.
## Model Building
First splot the data into train and test data.  
I tried six different models using the KFold from the model selection for selection the best model.  
- LogisticRegression
- DecisionTreeClassifier
- KNeighborsClassifier
- LinearDiscriminantAnalysis
- GaussianNB
- SVC
## Model Performance
Out of all the models **Linear Discriminate Analysis** give the better accuracy when compared with the other models.  
By using the **Linear Discriminate Analysis** model will be built.  
## Deployment
The model is dumped using the pickle file.
## Production
In this step I build a flask API endpoint that hosted on a local webserver. The API endpoint takes in a request with a list of values from patients report and returns an estimated result as which catgory your mobile belongs too (Cheep one, Middle range, Budject range, High range).
Later, I uploaded this project [Heroku](https://mobile-category-predection-api.herokuapp.com/)


