# data_cleaning.py
import pandas as pd
from fuzzywuzzy import process

def joint_data_set(df1, df2):
    merged_df = pd.concat([df1, df2], ignore_index=True)
   
    return merged_df

def remove_duplicates(df):  
    return df.drop_duplicates()

def remove_missing_values(df):
    return df.dropna()

def find_closest_match(row, mimu_SR_Name):
    matches = process.extractOne(row, mimu_SR_Name, score_cutoff=90)
    return matches[0]

def data_cleaning(df_towns, df_villages, df_news):
    df_towns.columns = ['SR_Name', 'name']
    df_villages.columns = ['SR_Name', 'name']
    df_news.columns = ['SR_Name', 'name']
    df_towns['name'] = df_towns['name'].apply(lambda name: name.replace('Town', '').strip())
    df_mimu = joint_data_set(df_towns, df_villages)
    df_mimu = remove_missing_values(df_mimu)
    df_news = remove_duplicates(df_news)
    mimu_SR_Name = df_mimu['SR_Name'].unique()
    df_news.loc[:, 'SR_Name']  = df_news['SR_Name'].apply(lambda x: find_closest_match(x, mimu_SR_Name))
    df_mimu = joint_data_set(df_mimu, df_news)
    
    return df_mimu

def main():
    # Load data from files  
    df_towns = pd.read_excel('./data/MMNames_mimu.xlsx', sheet_name='Towns')
    df_villages = pd.read_excel('./data/MMNames_mimu.xlsx', sheet_name='Villages')
    df_news = pd.read_csv('./data/MMNames_news.csv')
    df_mimu = data_cleaning(df_towns, df_villages, df_news)

    # save the cleaned data
    df_mimu.to_csv('./data/MMNames_clean.csv', index=False)
    
if __name__ == "__main__":
    main()