from flask import Flask, render_template, request
from main import *

state = 0
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")

# state -1 : crashed, ask for category 
# state 0 : got category, ask for price
# state 1 : got price, ask ques
# state 2 : got more info, repear

def intro(inp):
    return "I am Chat Bot to assist you buying a product"+"</span><br><br><span>"+"Which product are you interested in?"

def ask_price(inp):
    global category
    category=inp
    return "Do you have any max price preferences ?"

def print_product(product_id : str) -> None:
    res = ""
    product = odf.query('product_id==@product_id')
    
    res+="</span><br><span>"+"Product Title : "+str(list(product['title'])[0])
    res+="</span><br><span>"+"Product Brand : "+str(list(product['brand'])[0])
    product_desc = get_str_to_list(list(product['feature'])[0])
    for i, feature in enumerate(product_desc):
        res+="</span><br><span>"+"Feature "+str(i)+" : "+str(feature)
    
    res+="</span><br><span>"+"Product Price : "+str(list(product['price'])[0])
    return res

def print_product_without_feature(product_id : str) -> None:
    res = ""
    product = odf.query('product_id==@product_id')
    res+="</span><br><span>"+"Product Title : "+str(list(product['title'])[0])
    res+="</span><br><span>"+"Product Brand : "+str(list(product['brand'])[0])
    res+="</span><br><span>"+"Product Price : "+str(list(product['price'])[0])
    return res

def print_final_products(avg_score):
    res = ""
    res+="Final Product List According to your interests: "
    suggested_ids = sorted(avg_score, key=avg_score.get, reverse=True)[:no_of_products]
    for i in range(no_of_products):
        res+="</span><br><br><span>"+"Product no "+ str(i+1)+":"
        suggested_id = suggested_ids[i]
        product_id = list(pdf.query('title==@suggested_id')['product_id'])[0]
        res+=print_product(product_id)
    return res+"</span><br><br><span>"+"I am Chat Bot to assist you buying a product"+"</span><br><br><span>"+"Which product are you interested in?"

def print_intermediate_products(avg_score):
    res = ""
    res+="Products you may like: "
    suggested_ids = sorted(avg_score, key=avg_score.get, reverse=True)[:no_of_products]
    for i in range(no_of_products):
        res+="</span><br><br><span>"+"Product no "+ str(i+1)+":"
        suggested_id = suggested_ids[i]
        product_id = list(pdf.query('title==@suggested_id')['product_id'])[0]
        res+=print_product_without_feature(product_id)
    return res+"</span><br><br><span>"+random.choice(details_sentences)

def price_asked(max_price):
    global state, category, search_space, avg_score, model

    print("max_price: ", max_price)
    if(max_price.isnumeric()):
        print("num")
        model = Retrieval_Model(int(max_price))
    else:
        print("non")
        model = Retrieval_Model()

    id_score = model.get_similar_items(category, search_space)
    ids = id_score.index.to_list()
    cur_score = id_score['ensemble_similarity']

    print(category)
    print(search_space)
    print(len(ids))
    if(len(ids)==0):
        state=0
        return "Sorry! No desired item exist"+"</span><br><br><span>"+"Starting again"+"</span><br><br><span>"+"Which product are you interested in?"

    avg_score = dict()
    for i in range(min(search_space, len(ids))):
        avg_score[ids[i]] = float(cur_score[i])

    if(len(ids)<=no_of_products):
        state=0 
        return print_final_products(avg_score)

    details_sentence = random.choice(details_sentences)
    state = 2
    return print_intermediate_products(avg_score)    

def continue_convo(inp):
    global model, avg_score, state

    if inp in exit_keywords:
        state=0
        return print_final_products(avg_score)

    id_score = model.get_similar_items(inp, search_space)
    
    ids = id_score.index.to_list()
    cur_score = list(id_score['ensemble_similarity'])

    for i in range(min(search_space, len(ids))):
        if ids[i] not in avg_score:
            avg_score[ids[i]] = 0
        avg_score[ids[i]] = (retention)*float(avg_score[ids[i]]) + (1 - retention)*cur_score[i]

    state=2
    return print_intermediate_products(avg_score)

@app.route("/get")
def get_bot_response():
    inp = request.args.get('msg')
    global state

    if(state==-1):
        state=0
        return intro(inp) 
    elif(state==0):
        state=1
        return ask_price(inp)
    elif(state==1):
        state=2
        return price_asked(inp)
    elif(state==2):
        return continue_convo(inp)
    else:
        return "invalid"


if __name__ == "__main__":
    model = ""
    category = ""
    avg_score = {}
    app.run(host="127.0.0.1", port=4283)