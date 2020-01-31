# install pandas: run pip install pandas before you run this code
import pandas as pd
airline_url = 'https://raw.githubusercontent.com/quankiquanki/skytrax-reviews-dataset/master/data/airline.csv'
airline_data = pd.read_csv(airline_url)
airline_data.to_csv('airline.csv')
