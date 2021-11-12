import gradio as gr
import random
from numpy.core.records import record
from src.model import Retrieval_Model
import pandas as pd

def get_str_to_list(desc):
    desc = desc.strip('[').strip(']').split(',')
    return desc

def bot(request, model, details):
    processed_data_file = "Data/data_processed.csv"
    original_data_file = "Data/data1.csv"
    search_space = 1000
    no_of_products = 3
    retention  = 0.8
    exit_keywords = ["no", "exit", "finish", "done", "nothing", "no thanks"]

    pdf = pd.read_csv(processed_data_file)
    odf = pd.read_csv(original_data_file)

    id_score = model.get_similar_items(request, search_space)
    ids = id_score.index.to_list()
    cur_score = id_score['ensemble_similarity']

    avg_score = dict()
    for i in range(search_space):
        avg_score[ids[i]] = float(cur_score[i])

    while True:
        if details in exit_keywords:
            suggested_ids = sorted(avg_score, key=avg_score.get, reverse=True)[:no_of_products]
            for i in range(no_of_products):
                prod_id = i+1
                suggested_id = suggested_ids[i]
                product_id = list(pdf.query('title==@suggested_id')['product_id'])[0]
            break

        id_score = model.get_similar_items(details, search_space)
    
        ids = id_score.index.to_list()
        cur_score = list(id_score['ensemble_similarity'])

        for i in range(search_space):
            if ids[i] not in avg_score:
                avg_score[ids[i]] = 0
            avg_score[ids[i]] = (retention)*float(avg_score[ids[i]]) + (1 - retention)*cur_score[i]

    product = odf.query('product_id==@product_id')
    product_title = list(product['title'])[0]
    product_brand = list(product['brand'])[0]
    product_desc = get_str_to_list(list(product['feature'])[0])
    for i, feature in enumerate(product_desc):
        feat = {f"Feature {i} : {feature}"}
    product_price = list(product['price'])[0]

    resp = {'Product No: ': prod_id,  'Product Title': product_title, 'Product brand': product_brand, 'Features': feat, 'Product Price': product_price}
    return str(resp)

user_input = []
model = 0
def chat(message):
    start_sentences = [
        "Which product are you interested in?", 
        "Hi, let me help you find the product you are interested in:",
    ]
    details_sentences = [
        "Noted. What else?",
        "Go on:",
        "OK! continue:"
    ]
    start_sentence = random.choice(start_sentences)
    details_sentence = random.choice(details_sentences)
    fixed = [start_sentence, "Do you have any max price preferences ? ", details_sentence]
    global model

    history = gr.get_state() or []
    
    user_input.append(message)
    if message.startswith(user_input[0]):
        response = fixed[0]
    elif message.startswith(user_input[1]):
        response = fixed[1]
    elif message.startswith(user_input[2]):
        if(user_input[2].isnumeric()):
            model = Retrieval_Model(int(user_input[2]))
        else:
            model = Retrieval_Model()
        response = fixed[2]
    else:
        response = bot(user_input[1], model, user_input[2])       
    history.append((message, response))
    gr.set_state(history)

    html = "<div class='chatbot'>"
    for user_msg, resp_msg in history:
        html += f"<div class='user_msg'>{user_msg}</div>"
        html += f"<div class='resp_msg'>{resp_msg}</div>"
    html += "</div>"
    return html

iface = gr.Interface(chat, gr.inputs.Textbox(placeholder="Type here..."), "html", css="""
    .chatbox {display:flex;flex-direction:column}
    .user_msg, .resp_msg {padding:4px;margin-bottom:4px;border-radius:4px;width:80%}
    .user_msg {background-color:cornflowerblue;color:white;align-self:start}
    .resp_msg {background-color:lightgray;align-self:self-end}
""", allow_screenshot=False, allow_flagging=False)
iface.launch()
