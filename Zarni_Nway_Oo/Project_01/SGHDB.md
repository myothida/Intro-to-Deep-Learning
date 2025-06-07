# Project Title: Predicting Resale Prices for HDB Flats in Singapore

## Project Overview

This project aims to develop a regression model to predict the prices for the resale of public housing flats (HDB flats) in Singapore. By analyzing historical data from January 2017 to February 2023, the model will provide accurate price predictions that can help buyers and sellers make informed decisions in the real estate market.

## Dataset

The dataset used for this project is obtained from the Singapore government's open data portal. It includes detailed information on resale transactions of HDB flats between 01-Jan-2017 and 23-May-2024. The dataset contains 128,796 rows and 11 features.

- **Dataset URL**: [HDB Resale Prices](https://beta.data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view)


### Features

| No. | Name                     | Type                   |
|-----|--------------------------|------------------------|
| 1   | Month                    | DateTime (YYYY-MM)     |
| 2   | Town                     | Categorical Text       |
| 3   | Flat Type                | Categorical Text       |
| 4   | Block                    | Text                   |
| 5   | Street Name              | Text                   |
| 6   | Storey Range             | Categorical Text       |
| 7   | Flat Area                | Numeric (sqm)          |
| 8   | Flat Model               | Categorical Text       |
| 9   | Lease Commencement Date  | DateTime (YYYY)        |
| 10  | Remaining Lease          | Text (YY-MM)           |
| 11  | Resale Price             | Numeric (SGD)          |

## Methodology

1. **Data Collection**: Acquire the dataset from the provided URL, ensuring it includes all relevant attributes such as flat type, location, floor area, remaining lease, and transaction date.
2. **Data Preprocessing**: Clean the data to handle missing values, outliers, and inconsistencies. Transform categorical variables into numerical formats where necessary.
3. **Feature Engineering**: Select and engineer features that are most likely to influence resale prices. This may include creating new variables based on existing ones or transforming variables to better capture their relationships.
4. **Model Selection**: MLP is used in this project.
5. **Model Training and Validation**: Split the data into training and validation sets. Train the selected models on the training set and validate their performance using appropriate metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
6. **Hyperparameter Tuning**: Optimize the chosen model's hyperparameters to improve prediction accuracy.
7. **Model Evaluation**: Assess the final model's performance on a test set to ensure it generalizes well to unseen data.

## Comparisons

To benchmark our model's performance, we will compare our results with existing work. One such relevant project is available on GitHub:

- **Comparison Project URL**: [HDB Resale Price Prediction](https://github.com/JosiahMendes/HDB-Resale-Price-Prediction.git)

## Expected Outcomes

The primary outcome of this project is a robust regression model capable of predicting the resale prices of HDB flats with high accuracy. The model will provide insights into the key factors influencing resale prices and offer a reliable tool for prospective buyers, sellers, and real estate professionals in Singapore.

## Conclusion

By leveraging historical data and advanced regression techniques, this project aims to contribute to the transparency and efficiency of the HDB resale market. Accurate price predictions can empower stakeholders to make better-informed decisions, ultimately benefiting the entire real estate ecosystem in Singapore.
