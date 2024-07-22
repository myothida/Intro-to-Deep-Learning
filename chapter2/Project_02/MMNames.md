# Project 02: Name Classification for Regional Identification

## Description
This project aims to classify names of locations within Myanmar into one of the 18 states and regions. The classification model will be built using various machine learning techniques to accurately predict the state or region based on the given location name. The primary goal is to facilitate easy categorization and identification of locations for various applications, including geographic analysis, administrative management, and data organization.

## Table of Contents
- [Background](#background)
- [Data](#data)
- [Methodology](#method)
- [Expected Outcomes](#outcome)
- [Conclusion](#conclusion)


## Background
Myanmar, known for its diverse cultures and ethnic groups, is divided into 7 states and 7 regions, as well as 6 self-administered zones and a union territory, making a total of 18 distinct administrative divisions. Correctly identifying the state or region of a location name is essential for many administrative, logistic, and analytical tasks.

## Data
The datasets used in this project are obtained from the Myanmar Information Management Unit (MIMU) Resource Centre, a reliable source providing extensive geographical data. The datasets include:

1.  Villages Dataset: Contains 14,047 village names along with their corresponding states and regions.
2.  Towns Dataset: Comprises 536 town names and their respective states and regions.
3.  Supplementary Dataset: Collected from various news articles, includes 58,789 records of town names, often with variations in spelling and formatting. This dataset also contains duplicated entries due to multiple mentions in articles.

## Methodology

1. **Data Collection**: Acquire the dataset from the provided URL.
2. **Data Preprocessing**: Clean the data to handle missing values, duplicates, and inconsistencies. Transform categorical variables into numerical formats.
3. **Model Selection**: MLP is used in this project. 
5. **Model Training and Validation**: Split the data into training and validation sets. Train the selected models on the training set and validate their performance using appropriate metrics like accuracy, precision and recall.
6. **Hyperparameter Tuning**: Optimize the chosen model's hyperparameters to improve prediction accuracy.
7. **Model Evaluation**: Assess the final model's performance on a test set to ensure it generalizes well to unseen data.

## Expected Outcomes

The primary outcome of this project is a machine learning model capable of classifying location names into their respective states or regions in Myanmar.


## Conclusion

This project aims to enhance geographical information management in Myanmar by providing an efficient tool for classifying location names. By leveraging comprehensive datasets and advanced data processing techniques, the model will contribute significantly to various humanitarian and development activities in the region.
