# Parameters
processed_data_file = "Data/data_processed.csv"
original_data_file = "Data/data1.csv"
search_space = 1000
no_of_products = 3
retention  = 0.8
exit_keywords = ["no", "exit", "finish", "done", "nothing", "no thanks"]
start_sentences = [
    "Which product are you interested in?", 
    "Hi, let me help you find the product you are interested in:",
]
details_sentences = [
    "Noted. What else feature you want? (like digital/high resolution/etc) or type \"no\" if done: ",
    "any more info you would like to add (like 16gb ram/high power/etc)  or type \"no\" if done: "
]


from numpy.core.records import record
from src.model import Retrieval_Model
import pandas as pd
import random


def get_str_to_list(desc):
    desc = desc.strip('[').strip(']').split(',')
    return desc
def print_product(product_id : str) -> None:
    
    product = odf.query('product_id==@product_id')
    
    print("Product Title : ", list(product['title'])[0])
    print("Product Brand : ", list(product['brand'])[0])
    product_desc = get_str_to_list(list(product['feature'])[0])
    for i, feature in enumerate(product_desc):
        print(f"Feature {i} : {feature}")
    
    print("Product Price : ", list(product['price'])[0])


pdf = pd.read_csv(processed_data_file)
odf = pd.read_csv(original_data_file)

def print_final_products(avg_score):
    print("Final Product List According to your interests: \n\n")
    suggested_ids = sorted(avg_score, key=avg_score.get, reverse=True)[:no_of_products]
    for i in range(no_of_products):
        print("Product no ", i+1,":")
        suggested_id = suggested_ids[i]
        product_id = list(pdf.query('title==@suggested_id')['product_id'])[0]
        print_product(product_id)
        print()

def print_intermediate_products(avg_score):
    print("Products you may like: \n\n")
    suggested_ids = sorted(avg_score, key=avg_score.get, reverse=True)[:no_of_products]
    for i in range(no_of_products):
        print("Product no ", i+1,":")
        suggested_id = suggested_ids[i]
        product_id = list(pdf.query('title==@suggested_id')['product_id'])[0]
        print_product(product_id)
        print()

def start_new_conversation():

    print("** Welcome to Appliance Digital Store **\n")
    print("I am Chat Bot to assist you buying a product\n")
    start_sentence = random.choice(start_sentences)
    request  = input(start_sentence+" ")

    max_price = input("Do you have any max price preferences ? : ")

    if(max_price.isnumeric()):
        model = Retrieval_Model(int(max_price))
    else:
        model = Retrieval_Model()

    id_score = model.get_similar_items(request, search_space)
    ids = id_score.index.to_list()
    cur_score = id_score['ensemble_similarity']

    if(len(ids)==0):
        print("Sorry! No desired item exist\nPlease start again\n\n")
        start_new_conversation()

    avg_score = dict()
    for i in range(min(search_space, len(ids))):
        avg_score[ids[i]] = float(cur_score[i])

    if(len(ids)<=no_of_products):
        print_final_products(avg_score)
        start_new_conversation()

    print_intermediate_products(avg_score)

    # continue convo to narrow down products
    while True:
        details_sentence = random.choice(details_sentences)
        details = input(details_sentence+" ")
        if details in exit_keywords:
            print_final_products(avg_score)
            start_new_conversation()

        id_score = model.get_similar_items(details, search_space)
        
        ids = id_score.index.to_list()
        cur_score = list(id_score['ensemble_similarity'])

        for i in range(min(search_space, len(ids))):
            if ids[i] not in avg_score:
                avg_score[ids[i]] = 0
            avg_score[ids[i]] = (retention)*float(avg_score[ids[i]]) + (1 - retention)*cur_score[i]

        print_intermediate_products(avg_score)

if __name__ == "__main__":
    start_new_conversation()