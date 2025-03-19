# Import libraries 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_openml 
from sklearn.manifold import TSNE 
import seaborn as sns 

# Load the MNIST 784 data set 
mnist = fetch_openml('mnist_784', version=1, as_frame=False) 
X = mnist.data / 255.0  # Normalize the pixel values 
y = mnist.target  # Get the labels 

# Perform TSNE with 2 components 
tsne_model = TSNE(n_components=2, perplexity=30, n_iter=1500) 
tsne_results = tsne_model.fit_transform(X) 

# Convert the results to a pandas data frame 
feat_cols = ['dim1', 'dim2'] 
tsne_df = pd.DataFrame(tsne_results, columns=feat_cols) 
tsne_df['label'] = y 

# Plot the TSNE results with labels 
plt.figure(figsize=(12, 12)) 
sns.scatterplot( 
    data=tsne_df, x="dim1", y="dim2", 
    palette=sns.color_palette("tab10", 10),  # tab10  is color type 
    hue="label", 
    legend="full", 
    hue_order=sorted(tsne_df['label'].unique()) 
) 
plt.title("TSNE For MNIST Data Set") 
plt.show() 