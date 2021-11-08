# Parameters
processed_data_file = "Data/data_processed.csv"
original_data_file = "Data/data1.csv"
search_space = 1000
no_of_products = 3
retention  = 0.8

from numpy.core.records import record
from src.model import Retrieval_Model
import pandas as pd

model = Retrieval_Model()
pdf = pd.read_csv(processed_data_file)
odf = pd.read_csv(original_data_file)

request  = input("Which product are you interested in? ")
id_score = model.get_similar_items(request, search_space)
ids = id_score.index.to_list()
cur_score = id_score['ensemble_similarity']

avg_score = dict()
for i in range(search_space):
    avg_score[ids[i]] = float(cur_score[i])

def print_product(product_id : str) -> None:
    product = odf.query('product_id==@product_id')
    print("Product Title", list(product['title'])[0])
    print("Product Brand", list(product['brand'])[0])
    print("Product Price", list(product['price'])[0])

while True:
    details = input("Noted. What else? ")
    if details == "nothing":
        suggested_ids = sorted(avg_score, key=avg_score.get, reverse=True)[:no_of_products]
        for i in range(no_of_products):
            print("Product no ", i+1,":")
            suggested_id = suggested_ids[i]
            product_id = list(pdf.query('title==@suggested_id')['product_id'])[0]
            print_product(product_id)
            print()
        break

    id_score = model.get_similar_items(details, search_space)
    
    ids = id_score.index.to_list()
    cur_score = id_score['ensemble_similarity']

    for i in range(search_space):
        if ids[i] not in cur_score:
            avg_score[ids[i]] = 0
        avg_score[ids[i]] = (1-retention)*float(id_score) + retention*cur_score[ids[i]]
