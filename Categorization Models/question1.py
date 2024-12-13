import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


X = pd.read_csv('X.csv', header=None, names=['Weight', 'Height', 'Category'])
y = pd.read_csv('y.csv', header=None, names=['Weight', 'Height'])

alpha_weight = 0.8
alpha_height = 0.2

def weighted_distance(x, y):
    return np.sqrt(alpha_weight * (x[0] - y[0])**2 + alpha_height * (x[1] - y[1])**2)

def gcm_predict(test_points, train_data):
    predictions = []
    for test_point in test_points:
        similarities = {1: 0, 2: 0, 3: 0} 
        for _, row in train_data.iterrows():
            distance = weighted_distance(test_point, row[['Weight', 'Height']])
            similarity = np.exp(-distance)
            similarities[row['Category']] += similarity
        
        if similarities[3] > similarities[2] and (similarities[3] - similarities[2]) < 0.1:
            predicted_category = 2
        else:
            predicted_category = max(similarities, key=similarities.get)
        
        predictions.append(predicted_category)
    return predictions

test_points = y.values
predictions = gcm_predict(test_points, X)

y['Predicted_Category'] = predictions
y.to_csv('question1_y_with_predictions.csv', index=False)

print("Predictions saved to question1_y_with_predictions.csv")