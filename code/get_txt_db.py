import sqlite3
import pandas as pd
import re

con = sqlite3.connect('../db/products.db')
cur = con.cursor()

data = pd.read_sql_query('SELECT * FROM products', con, index_col='id')
data = data.drop_duplicates()


def data_to_text(serie):
    #text = ' '.join(map(str, serie.iloc[0:]))
    text = serie['name']
    text = re.sub(r'по.*', '', text)
    text = text.lower().strip()
    return text

result = data.apply(data_to_text, axis=1)

for row in result:
    row = row + '\n'
    with open('../db/names.txt', 'a', encoding='utf-8') as f:
        f.write(row)


