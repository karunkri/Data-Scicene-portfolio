# Movie Rating Prediction Project
This repository contains code for predicting movie ratings based on features such as genre, director, and actors. 
The goal of this project is to analyze historical movie data and develop a model that accurately estimates the rating given to a movie by users or critics.

## Project Overview
The Movie Rating Prediction project focuses on:

- Data analysis: Exploring the key factors influencing movie ratings.
+ Data preprocessing: Handling missing data, encoding categorical features, and scaling numerical data.
* Feature engineering: Working with columns like Genre, Director, and Actors to extract relevant information.
- Modeling: Using regression models to predict the movie ratings.

### Table of Contents
- Project Overview
+ Installation
* Data Preprocessing
- Feature Engineering
+ Training the Model
* Model Evaluation
- Contributors


### Installation
1. Clone the repository:
```
git clone https://github.com/your-username/movie-rating-prediction.git
cd movie-rating-prediction
```
2. Install the required dependencies:
```
pip install -r requirements.txt
```
3. Install additional libraries (if not included in the requirements):
```
pip install category_encoders
```
##### Data Preprocessing
- Handling Missing Values:
  Missing values in the dataset are handled by filling them with the most appropriate values (e.g., using mode or mean).

+ Categorical Encoding:

Actors, Directors, and Genre columns are encoded using one-hot encoding or target encoding to convert categorical data into a numeric format that machine learning algorithms can process.
We also perform binary encoding for actors and directors across different columns.
Example:
```
unique_actors = pd.unique(new_movie[['Actor 1', 'Actor 2', 'Actor 3']].values.ravel('K'))
for actor in unique_actors:
    if pd.notna(actor):
        new_movie[actor] = new_movie[['Actor 1', 'Actor 2', 'Actor 3']].apply(lambda x: 1 if actor in x.values else 0, axis=1)
```
* Memory Optimization: We reduce memory usage by converting numerical columns to lower memory types like int8, and converting categorical columns to the category type.
```
numeric_columns = data_final.select_dtypes(include=['int64', 'float64']).columns
data_final[numeric_columns] = data_final[numeric_columns].astype('int8')

non_numeric_columns = data_final.select_dtypes(include=['object']).columns
data_final[non_numeric_columns] = data_final[non_numeric_columns].astype('category')
```
###### Feature Engineering
We have three actor columns (Actor 1, Actor 2, and Actor 3). These columns are merged and transformed into binary columns for each unique actor in the dataset.

###### Training the Model
The model is trained using various regression techniques such as Linear Regression. The features like actors, directors, and genres are used to predict the movie rating.
```
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
```
###### Model Evaluation
The model is evaluated using metrics like Mean Squared Error (MSE) and R-squared (R²) to measure how well the model predicts movie ratings.
```
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
```
###### Contributors
[https://github.com/karunkri]


