from sklearn.datasets import fetch_openml 
import matplotlib.pyplot as plt 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.metrics import accuracy_score 
mnist = fetch_openml('mnist_784', version=1, as_frame=False) 
X = mnist["data"] 
y = mnist["target"] 
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:] 
lda = LinearDiscriminantAnalysis(n_components=9) # ncomponent comes from label of data 
X_train_lda=lda.fit_transform(X_train, y_train) 
test_lda = lda.transform(X_test)   
test_predict = lda.predict(X_test)   
plt.figure(figsize=(10, 10)) 
plt.scatter( 
    X_train_lda[:,0], 
    X_train_lda[:,1], 
    c=y_train.astype(int), 
    cmap='tab10' 
) 
plt.colorbar() 
plt.title('MNIST train data set classified in 2D') 
plt.show() 
plt.figure(figsize=(10, 10)) 
plt.scatter( 
    test_lda[:,0], 
    test_lda[:,1], 
    c=y_test.astype(int), 
    cmap='tab10' 
) 
plt.colorbar() 
plt.title('MNIST test data set classified in 2D') 
plt.show() 
acc = accuracy_score(y_test, test_predict) 
print(f'Accuracy: {acc:.4f}')