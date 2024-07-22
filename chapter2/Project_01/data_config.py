# data_config.py
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def save_dataset_info(df, fname):
    dataset_info = pd.DataFrame(columns = ['Num', 'Data Attributes', 'Column Name', 'Data Type', 'Description'])
    dataset_info['Num'] = range(1, len(df.columns)+1)
    dataset_info['Data Attributes'] = [col.capitalize().replace('_', ' ') for col in df.columns]
    dataset_info['Column Name'] = df.columns
    dataset_info['Data Type'] = df.dtypes.values
    data_type_map = {'int64': 'Numeric', 'float64': 'Numeric',  'object': 'Text'}
    dataset_info['Data Type'] = dataset_info['Data Type'].astype(str).map(data_type_map)
    description = ['Month and Year of sale', 'Designated residential area', 
            'Classification of units by room size.', 
            'The Block number where the unit sold located', 
            'Stree name of the unit sold located', 
            'Estimated range of floors the unit sold was located on', 
            'Total interior space within the unit, measured in square meters', 
            'Classification of units by generation', 
            'Starting point of a lease agreement (Year)', 
            'Remaining amount of time left on the lease (Years and Months)', 
            'Resale Price of the flat sold'
        ]
    dataset_info['Description'] = description
    dataset_info.to_csv('./data/'+fname, index=False)

def read_resale_price_index():
    df_RPI = pd.read_csv('./data/HDBRPIMonthly.csv')
    df_RPI = df_RPI.iloc[:-3, :].copy()
    extended_date_range = pd.date_range(start='2023-01', end='2024-03', freq='MS')
    extended_formatted_dates = extended_date_range.strftime('%Y-%m').tolist()
    rpi_data = [183.7, 180.4, 178.5, 176.2, 173.6][::-1] #24 t0 23 first quarter
    monthly_rpi_values = [value for value in rpi_data for _ in range(3)]
    df_RPI_new = pd.DataFrame({
        'month': extended_formatted_dates,
        'index': monthly_rpi_values
    })
    df_RPI_combined = pd.concat([df_RPI, df_RPI_new], ignore_index=True)
    df_RPI_combined['month'] = pd.to_datetime(df_RPI_combined['month'], format='%Y-%m')
    return df_RPI_combined

def adjust_resale_price(df, cut_off_date='2024-04-01'):
    df_RPI = read_resale_price_index()
    df['month'] = pd.to_datetime(df['month'], format='%Y-%m')
    df = df[df['month'] < cut_off_date]
    df = df.drop_duplicates()
    df = df[df['resale_price'] > 0]
    
    df = df.merge(df_RPI, on='month', how='left')
    q4_rpi = df_RPI[df_RPI['month'] == '2024-03']['index'].values[0]
    df['adjusted_price'] = q4_rpi* df['resale_price'] / df['index'] 
    return df

def encode_categorical_features(df, col_name, encoder=None):
    if encoder is None:
        df[col_name] = df[col_name].astype('category').cat.codes
    else:
        encoded = encoder.fit_transform(df[[col_name]])    
        df = pd.DataFrame(encoded.toarray(), columns=encoder.get_feature_names_out([col_name]))
    return df

def get_floor_from_range(storey_range):
    start, end = map(int, storey_range.strip().split(" TO "))
    return (start + end) / 2
       
def get_remaining_lease_months(lease_str):
    numbers = re.findall(r'\d+', lease_str)
    if(len(numbers)>1):
        return int(numbers[0])*12+int(numbers[1])
    else:
        return int(numbers[0])
    
def preprocess_data(df):
    df['month'] = pd.to_datetime(df['month'], format='%Y-%m')    
    df['floor'] = df['storey_range'].apply(get_floor_from_range)
    df['remaining_lease_months'] = df['remaining_lease'].apply(get_remaining_lease_months)    
    onehot_encoder = OneHotEncoder()
    df = encode_categorical_features(df, 'flat_type')
    df_town = encode_categorical_features(df, 'town', encoder = onehot_encoder)
    df_flat_model = encode_categorical_features(df, 'flat_model', encoder = onehot_encoder)
    df = df.drop(columns=['month', 'town', 'storey_range','flat_model', 'lease_commence_date',  'remaining_lease',
                            'street_name', 'block', 'index', 'resale_price'])
    df = pd.concat([df, df_town, df_flat_model], axis=1)
    return df

def visualize_adjusted_price(df, fname):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    df['month'] = pd.to_datetime(df['month'], format='%Y-%m')    
    df = df.resample('1ME', on='month').mean().reset_index()
    ax.plot(df['month'], df['resale_price'], marker='o', linestyle='-', label='Original Resale Price')
    ax.plot(df['month'], df['adjusted_price'], marker='s', linestyle='-', color = 'y',
            label='Adjusted Resale Price')
    ax.set_xlabel('Month')
    ax.set_ylabel('Resale Price')    
    ax.set_title('Price vs Month (Sampled Every Month)')
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('./imgs/' + fname, dpi = 700)

    
