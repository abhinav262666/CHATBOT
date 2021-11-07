# Parameters
data_file = "Data/data_processed.csv"
search_space = 1000
retention  = 0.8

from numpy.core.records import record
from src.model import Retrieval_Model
import pandas as pd

model = Retrieval_Model()
df = pd.read_csv(data_file)

request  = input("Which product are you interested in? ")
id_score = model.get_similar_items(request, search_space)
ids = id_score.index.to_list()
cur_score = id_score['ensemble_similarity']

avg_score = dict()
for i in range(search_space):
    avg_score[ids[i]] = float(cur_score[i])


while True:
    details = input("Noted. What else? ")
    if details == "nothing":
        suggested_id = max(zip(avg_score.values(), avg_score.keys()))[1]
        print(df.query('title==@suggested_id'))    
        print(suggested_id)
        break

    id_score = model.get_similar_items(details, search_space)
    
    ids = id_score.index.to_list()
    cur_score = id_score['ensemble_similarity']

    for i in range(search_space):
        if ids[i] not in cur_score:
            avg_score[ids[i]] = 0
        avg_score[ids[i]] = (1-retention)*float(id_score) + retention*cur_score[ids[i]]
