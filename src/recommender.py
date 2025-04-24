from surprise import Dataset, Reader, KNNBasic
from collections import defaultdict

def prepare_dataset(ratings):
    reader = Reader(rating_scale=(1, 5))
    return Dataset.load_from_df(ratings[['user_id', 'item_id', 'rating']], reader)

def build_user_cf(trainset):
    sim_options = {'name': 'cosine', 'user_based': True}
    model = KNNBasic(sim_options=sim_options)
    model.fit(trainset)
    return model

def build_item_cf(trainset):
    sim_options = {'name': 'cosine', 'user_based': False}
    model = KNNBasic(sim_options=sim_options)
    model.fit(trainset)
    return model

def get_top_n(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n
