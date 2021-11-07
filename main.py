# Parameters
data_file = "Data/data_processed.csv"
cnt_items = 3

from src.model import Retrieval_Model
import pandas as pd

model = Retrieval_Model()
df = pd.read_csv(data_file)

request = input("Hi, how may I help you? ")
while True:
    records = model.get_similar_items(request, cnt_items)
    for i in range(cnt_items):
        print("Item ", i)
        title = records['title'][i]
        similarity = records['ensemble_similarity'][i]
        print("Title of product is ", title)
        print("Score of the product is ", similarity)
    details = input("Any further details? ")
    if details == "exit": break
    request += details
