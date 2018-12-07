import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

data = fetch_movielens(min_rating=4.0)

print(data['train'])
print(data['test'])

model = LightFM(loss='warp')

model.fit(data['train'], epochs=30, num_threads=2)

def sample_recommendation(model, data, u_ids):

    n_users, n_items = data['train'].shape

    for u_id in u_ids:
        known_positives = data['item_labels'][data['train'].tocsr()[u_id].indices]

        scores = model.predict(u_id, np.arange(n_items))

        top_items = data['item_labels'][np.argsort(-scores)]

        print("User %s" % u_id)
        print("     Known positives: ")

        for x in known_positives[:3]:
            print("            %s" % x)
        
        print("     Recommended: ")
        for x in top_items[:3]:
            print("            %s" % x)

sample_recommendation(model, data, [3, 25, 149])