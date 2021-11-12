from model import Retrieval_Model

model = Retrieval_Model()

recs = model.get_similar_items("a electric heater", 5)
print(recs)
# print(model.view_recommendations(recs))
single_title = "Tupperware Freezer Square Round Container Set of 6"
# print(model.df['title'][0])
print(model.df[model.df.title == single_title])
# print(model.df.query('title==@single_title'))