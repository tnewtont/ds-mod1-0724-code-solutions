# -*- coding: utf-8 -*-
"""scraping-formula1_solution

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1NrElKaMCO7YlzM6L8pvnJ-XVbpISZrTS
"""

from bs4 import BeautifulSoup
import pandas as pd
import requests

url = 'https://www.formula1.com/en/results/2022/drivers'

r = requests.get(url)

r.status_code # Success!

data = BeautifulSoup(r.content)
data

data.find_all('table')

table = data.table #table

table_rows = table.find_all('tr') #table rows
header_row = table_rows[0]
data_rows = table_rows[1:]

headers = header_row.find_all('th') #table headers
colnames = [name.text for name in headers]
colnames

# Make the names in Driver column display more nicely (remove 3-letter abbrev)
# I'm not sure if the 3-letter abbrev should be still included (i.e. add a space
# after the last name) or be removed completely. Thus, I wrote a function that
# removes the 3-letter abbrev using regular expressions.

test_string = "Max VerstappenVER"
import re
def make_last_name_nicer(last_name):
  new_last_name = re.sub(r'[A-Z]+$','',last_name)
  return new_last_name
make_last_name_nicer(test_string)

rows = []
for row in data_rows:
  tds = row.find_all('td')
  tds_data = [td.text for td in tds]
  rows.append(tds_data)

drivers = pd.DataFrame(data = rows, columns = colnames)
# Fix the Driver column using the function written earlier
drivers['Driver'] = drivers['Driver'].apply(make_last_name_nicer)
drivers

drivers['Pos'] = drivers['Pos'].astype(int)
drivers['Pts'] = drivers['Pts'].astype(int)

# After you get a dataframe, use apply function, lambda function
# Loop through all the rows
# Map

f1_2022 = drivers.to_csv('f1_2022.csv', index = False)

f1_2022_read = pd.read_csv('f1_2022.csv')
f1_2022_read

# Changing two columns' data types

f1_2022_read.equals(drivers)

# Using the pd.read_html( ) method
drivers_html = pd.read_html(url)[0]
drivers_html['Driver'] = drivers_html['Driver'].apply(make_last_name_nicer)
drivers_html