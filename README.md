# GNN-Recommender_system
Graph Neural Network Recommender system
Overview
The project uses PyTorch Geometric, a library built on PyTorch specifically for graph neural networks, to create a recommendation system that models user-item interactions as a graph and predicts ratings using a Graph Convolutional Network (GCN).

# Key Components
# 1. Data Preparation
Dataset: The Amazon Electronics ratings dataset containing user-product interactions with ratings (1-5 scale).

Data Loading: The dataset is loaded from a CSV file with columns representing user IDs, product IDs, ratings, and timestamps.

Preprocessing:

Renaming columns for clarity (userId, productId, Rating, timestamp)

Limiting to the first 5000 entries for manageability

Removing NA values and duplicates

Encoding user and product IDs using LabelEncoder to convert them to numerical indices

Splitting data into training (80%), validation (20%), and test (20%) sets

# 2. Graph Construction
The recommendation problem is modeled as a bipartite graph where:

Nodes represent users and products

Edges represent user-product interactions (ratings)

Edge attributes store the rating values

Key steps:

Creating an edge index tensor that connects users to products they've rated

Using rating values as edge attributes

Adding node features (identity matrices) to represent each node

# 3. Model Architecture
The GCN model consists of:

Two GCNConv layers:

First layer transforms input features (5227 dimensions) to hidden representations (16 dimensions)

Second layer further processes these hidden representations

Both use ReLU activation functions

Final linear layer:

Takes concatenated node features from both ends of an edge

Outputs a single predicted rating value

# 4. Training Process
Optimization: Adam optimizer with learning rate 0.01

Loss Function: Mean Squared Error (MSE) between predicted and actual ratings

Training Loop: 200 epochs showing gradual decrease in loss from ~17 to ~2

# 5. Evaluation
The model is evaluated on both validation and test sets using:

RMSE (Root Mean Squared Error): ~1.60

MAE (Mean Absolute Error): ~1.45

# Technical Details
# Graph Representation:

Users and products are treated as separate node types in a bipartite graph

Each rating creates an edge between a user and product node

Node features are simple one-hot encodings (identity matrices)

# Message Passing:

The GCN layers perform neighborhood aggregation

Information propagates from users to products they've rated and vice versa

The model learns meaningful embeddings for users and products

# Prediction:

For a given user-product pair, the model concatenates their learned embeddings

Passes this through a linear layer to predict the rating

Strengths of the Approach
Graph Structure Utilization: Effectively captures the user-item interaction patterns

Neighborhood Aggregation: Leverages information from similar users/items

Flexibility: Can be extended to incorporate additional node/edge features
