# Ch4-MachineLearningDeepLearning
 ### **TASK 1**
 
I have completed several steps in data preparation and modeling using the SC_HW1_bank_data.csv dataset. Here is a detailed explanation of the tasks I performed:
Certainly, here are the changes to the sentences:

### Data Preparation

1. **Import Library:** I imported two libraries, `pandas` and `numpy`, which are used for data manipulation and numerical calculations.

2. **Input Dataset:** I read the dataset from the given URL and loaded it into a Pandas DataFrame. Then, I selected the relevant columns for modeling and saved them in a new DataFrame.

3. **Preprocessing:** I performed several preprocessing steps, namely:
   - Conducted one-hot encoding to transform categorical data (the "Geography" and "Gender" columns) into a numeric form.
   - Separated the target variable ("Exited") from the features (X).
   - Applied normalization using Min-Max Scaler if needed.

4. **Train-Test Split:** I split the data into a training set (train) and a testing set (test) with a 75% training and 25% testing ratio.

### Modeling Phase

#### Model 1 (K-Nearest Neighbors - KNN)

- I selected the K-Nearest Neighbors (KNN) model for the first model.
- I performed parameter tuning using GridSearchCV to find the best parameters, such as the number of neighbors (n_neighbors), distance metric (metric), and neighbor weighting scheme (weights).
- I trained the model using the best-found parameters.
- I evaluated the model using metrics such as the classification report, confusion matrix, and accuracy.

#### Model 2 (Random Forest)

- The second model I chose is Random Forest.
- Similar to Model 1, I conducted parameter tuning using GridSearchCV to find the best parameters, such as the number of trees (n_estimators), maximum depth (max_depth), maximum features (max_features), and the minimum number of samples required to split a node (min_samples_split).
- I trained the model with the best parameters.
- I evaluated the model using the same metrics.

#### Model 3 (Decision Tree)

- The third model I selected is the Decision Tree.
- I once again performed parameter tuning using GridSearchCV to find the best parameters, such as maximum depth (max_depth), minimum samples in leaf nodes (min_samples_leaf), minimum samples to split a node (min_samples_split), and the splitting criterion (criterion).
- I trained the model with the best parameters.
- I evaluated the model using the same metrics.

Based on the evaluation results, I concluded that Model 3 (Decision Tree) is the best model because it has the highest accuracy (0.8628) compared to Model 1 (KNN) and Model 2 (Random Forest). I also explained that the Decision Tree is a relatively simple and interpretable model, making it suitable for explaining predictions to users who may not have in-depth statistical knowledge.

### **TASK 2**

In the process of preparing the data and conducting clustering analysis on the 'Cluster S1.csv' dataset, I undertook a series of essential steps. These steps involved importing libraries, reading and cleaning the data, performing data visualization, searching for the optimal number of clusters using the K-Means algorithm, evaluating the clusters through the silhouette score, and finally, visualizing the results to gain insights into the underlying data patterns. Here is a detailed explanation of the tasks I performed:

# Preparation
The preparation phase is typically carried out to prepare data before entering the modeling phase. In this case, I am working with the "Cluster S1.csv" dataset, which was randomly generated. The following are the steps I went through before the modeling phase:

1. **Importing Libraries:** I imported several Python libraries necessary for data analysis and clustering, including NumPy, Pandas, Matplotlib, Seaborn, and scikit-learn's clustering and silhouette score functions.

2. **Reading the Dataset:** I read the dataset from a URL using Pandas, loaded it into a DataFrame, and displayed the first few rows to inspect the data.

3. **Data Cleaning:** I dropped a column named "no" that was not needed for clustering analysis.

4. **Data Visualization:** I visualized the data by creating a scatter plot to get a better understanding of its structure.

# Clustering

5. **Cluster Search:** I proceeded to segment the data using a clustering method of my choice. The ultimate goal is to evaluate the best number of clusters using the silhouette score.

6. **Cluster Search Process:** I conducted a search for the best number of clusters using the K-Means clustering algorithm. This process involved iterating over a range of cluster numbers, fitting the K-Means model with different clusters, and collecting the inertia (within-cluster sum of squares) for each cluster number. The inertia was plotted against the number of clusters to help identify an optimal number.

7. **Silhouette Score Evaluation:** I completed the code to evaluate the best number of clusters based on the silhouette score. The silhouette score is a metric that measures how similar an object is to its own cluster compared to other clusters. A higher silhouette score indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters.

8. **Visualization of Cluster Results:** To visualize the cluster results, I assigned cluster labels to the data points and created a scatter plot with different colors indicating the clusters. Seaborn was used for this visualization.

In summary, I started by importing necessary libraries, reading the dataset, and cleaning it. Then, I performed data visualization to gain insights. Following that, I explored different numbers of clusters using the K-Means algorithm and evaluated the clusters' quality using the silhouette score. Finally, I visualized the results by plotting the data points with cluster labels to help understand the underlying patterns in the data.

### **TASK 3**

In this task, I worked on a series of steps related to training a neural network for image classification using the MNIST dataset. Let's break down what I did in each step:

1. **Loading the MNIST Dataset:** I imported the necessary libraries for deep learning, including PyTorch, and used torchvision to load the MNIST dataset. I applied transformations to the dataset, including converting the images to tensors and normalizing them.

2. **Visualizing MNIST Datasets:** I defined a function, `plot_images`, to visualize several MNIST dataset images. I then loaded a batch of images and labels using a data loader and used the `plot_images` function to display the first ten images.

3. **Designing the Neural Network Model:** I created a neural network model class called `NN`. This simple feedforward neural network consists of two linear layers with a ReLU activation function in between. The input to the model is an image with a size of 28x28 pixels, which is flattened to 784 dimensions. The output layer has ten units, corresponding to the ten possible digits (0-9).

4. **Setting Hyperparameters:** I defined hyperparameters such as the loss function (cross-entropy), optimizer (Adam), learning rate (0.001), and the number of training epochs (20).

5. **Training the Model:** I implemented a training loop to train the neural network on the MNIST dataset. In the loop, I iterated over the training data and performed the following:
   - Zeroed out gradients.
   - Performed a forward pass to obtain predictions.
   - Computed the loss.
   - Backpropagated the gradients.
   - Updated the model's parameters using the optimizer.
   - Kept track of the running loss for each epoch.

6. **Evaluating the Model:** I evaluated the trained model using various performance metrics. I calculated accuracy, precision, recall, and F1 score. Additionally, I computed a confusion matrix to visualize the model's performance in classifying the test data.

7. **Describing and Explaining the Results:** Finally, I displayed one of the test images and showed the model's prediction for that image. This is a visual representation of the model's ability to classify a digit based on the input image.

Overall, I have successfully trained a neural network to recognize handwritten digits in the MNIST dataset, and I assessed its performance using multiple evaluation metrics, providing a comprehensive evaluation of the model's classification capabilities.

### **TASK BONUS**

In this bonus task, I conducted a series of experiments to compare the performance of different configurations of neural network models, loss functions, and activation functions. Let's break down what I did in each part:

**1. Compare 3 Different Configurations for Model Depth:**

I experimented with three different configurations of model depth:
- **WideModel**: This model had a wider architecture with three fully connected layers (256, 128, 10).
- **DeepModel**: This model had a deeper architecture with five fully connected layers (128, 128, 128, 128, 10).
- **OriginalModel**: This model had a more standard architecture with two fully connected layers (128, 10).

After training and evaluating these models, I found that the DeepModel had the highest accuracy (0.9635) on the test data. This suggests that increasing the depth of the neural network can improve its performance.

**2. Compare 3 Different Loss Functions:**

I tested three different loss functions on a model with a standard architecture:
- **CrossEntropyLoss**: This is a common choice for classification problems.
- **NLLLoss**: Negative Log-Likelihood Loss, which is suitable for probability distribution-based models.
- **MSELoss**: Mean Squared Error Loss, typically used for regression problems but tested here for classification.

After training and evaluation, I found that the CrossEntropyLoss had the highest accuracy (0.9545) on the test data. This demonstrates that the appropriate choice of loss function significantly impacts model performance.

**3. Compare 3 Different Activation Functions:**

I investigated the impact of three different activation functions on a model with a standard architecture:
- **ReLU (Rectified Linear Unit)**: A common choice known for its simplicity and effectiveness.
- **Sigmoid**: A sigmoid activation function used in earlier neural networks.
- **Tanh (Hyperbolic Tangent)**: Another activation function similar to sigmoid but with a range of -1 to 1.

The results showed that the Sigmoid activation function led to the highest accuracy (0.9774) on the test data. This indicates that the choice of activation function plays a crucial role in the model's performance.

In conclusion, these experiments highlighted the importance of model depth, loss function selection, and activation function choice in deep learning. The results can guide practitioners in making informed decisions when designing and training neural networks for various tasks. My detailed experimentation and analysis provide valuable insights into the impact of these choices on model performance.
