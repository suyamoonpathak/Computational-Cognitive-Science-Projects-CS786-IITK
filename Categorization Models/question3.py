import pandas as pd
import numpy as np
from random import shuffle
from sklearn.metrics.pairwise import euclidean_distances
from question2 import dLocalMAP

X = pd.read_csv('X.csv', header=None).values 
y = pd.read_csv('y.csv', header=None).values

def gcm_predict(test_data, training_data):
    predictions = []
    attention_weights = np.array([0.8, 0.2])
    
    for test_point in test_data:
        distances = euclidean_distances(training_data[:, :-1] * attention_weights, 
                                        test_point.reshape(1, -1) * attention_weights).flatten()
        
        
        inv_distances = 1 / (distances + 1e-5)
        weighted_votes = np.zeros(3)
        
        for idx, inv_dist in enumerate(inv_distances):
            category = int(training_data[idx, -1]) - 1
            weighted_votes[category] += inv_dist
        
        predictions.append(np.argmax(weighted_votes) + 1)
    
    return predictions

def predict_categories(training_data, test_data):
    model = dLocalMAP([0.5, np.ones(training_data.shape[1] - 1)])  # c = 0.5, alpha = [1, 1]
    
    for i in range(len(training_data)):
        model.stimulate(training_data[i, :-1])  
    
    predictions = [model.predict(test_data[i]) + 1 for i in range(len(test_data))]
    return predictions

num_shuffles = 5

gcm_predictions_list = []
rmc_predictions_list = []

for _ in range(num_shuffles):
    X_shuffled = np.copy(X)
    np.random.shuffle(X_shuffled)
    
    gcm_predictions = gcm_predict(y, X_shuffled)
    gcm_predictions_list.append(gcm_predictions)
    
    rmc_predictions = predict_categories(X_shuffled, y)
    rmc_predictions_list.append(rmc_predictions)

gcm_consistent = all(np.array_equal(gcm_predictions_list[0], preds) for preds in gcm_predictions_list)
rmc_consistent = all(np.array_equal(rmc_predictions_list[0], preds) for preds in rmc_predictions_list)

print("GCM predictions consistent across shuffles:", gcm_consistent)
print("RMC predictions consistent across shuffles:", rmc_consistent)

pd.DataFrame(gcm_predictions_list).to_csv('question3_gcm_predictions_across_shuffles.csv', index=False)
pd.DataFrame(rmc_predictions_list).to_csv('question3_rmc_predictions_across_shuffles.csv', index=False)
