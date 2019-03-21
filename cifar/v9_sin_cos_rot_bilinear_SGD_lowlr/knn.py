import h5py
import numpy as np
from sklearn import neighbors

n_neighbors = 10

with h5py.File('./features/train_pool2.h5', 'r') as hf:
    train_feats = hf['feats'][:]
    train_labels = hf['labels'][:]
    
with h5py.File('./features/test_pool2.h5', 'r') as hf:
    test_feats = hf['feats'][:]
    test_labels = hf['labels'][:]
    
train_labels = np.reshape(train_labels, -1).astype(np.int64)
test_labels = np.reshape(test_labels, -1).astype(np.int64)

clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform', n_jobs=16)
clf.fit(train_feats, train_labels)

num_correct = 0.
for ii in range(len(test_labels)):
    print(ii)
    predict_label = clf.predict(test_feats[ii:ii+1])
    if predict_label[0] == test_labels[ii]:
        num_correct += 1.
        
accuracy = num_correct / float(len(test_labels)) 
    
#predict_labels = clf.predict(test_feats)
#correct = predict_labels == test_labels
#correct = np.asarray(correct, dtype=np.float32)
#accuracy = np.sum(correct)/len(test_labels)
print(accuracy)




