from surprise import accuracy
from collections import defaultdict
import json

def evaluate_rmse(model, testset):
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    return rmse, predictions

def precision_recall_at_k(predictions, k=5, threshold=3.5):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions, recalls = [], []
    for user_ratings in user_est_true.values():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]

        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in top_k)
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in top_k)

        precision = n_rel_and_rec_k / n_rec_k if n_rec_k else 0
        recall = n_rel_and_rec_k / n_rel if n_rel else 0

        precisions.append(precision)
        recalls.append(recall)

    return sum(precisions) / len(precisions), sum(recalls) / len(recalls)

def save_metrics(rmse, precision, recall, path='results/evaluation_metrics.json'):
    with open(path, 'w') as f:
        json.dump({
            "rmse": rmse,
            "precision@5": precision,
            "recall@5": recall
        }, f)
