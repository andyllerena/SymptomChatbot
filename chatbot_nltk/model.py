import torch.nn as nn

# Define a neural network class
class NeuralNet(nn.Module):
    # Constructor to initialize the neural network architecture
    def __init__(self, input_size, hidden_size, num_classes):
        # Call the constructor of the parent class (nn.Module)
        super(NeuralNet, self).__init__()

        # Define the first linear layer with input_size input features and hidden_size output features
        self.l1 = nn.Linear(input_size, hidden_size) 
        
        # Define the second linear layer with hidden_size input features and hidden_size output features
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        
        # Define the third linear layer with hidden_size input features and num_classes output features
        self.l3 = nn.Linear(hidden_size, num_classes)
        
        # Define the ReLU activation function
        self.relu = nn.ReLU()

    # Forward pass through the neural network
    def forward(self, x):
        # Pass input through the first linear layer
        out = self.l1(x)
        
        # Apply ReLU activation function to the output of the first linear layer
        out = self.relu(out)
        
        out = self.l2(out)
        
        out = self.relu(out)
        
        out = self.l3(out)
        return out
