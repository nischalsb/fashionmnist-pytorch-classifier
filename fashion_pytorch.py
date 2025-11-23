import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """

    # 1. Define the transform 
    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    if training:
        # 3. Create the training set
        train_set = datasets.FashionMNIST(
            root='./data', 
            train=True, 
            download=True, 
            transform=custom_transform
        )
        
        # 4. Create the training loader
        # For training, we should shuffle the data
        loader = torch.utils.data.DataLoader(
            train_set, 
            batch_size=64, 
            shuffle=True
        )
    else:
        # 3. Create the test set
        test_set = datasets.FashionMNIST(
            root='./data', 
            train=False, 
            download=True, 
            transform=custom_transform
        )
        
        # 4. Create the test loader
        # For testing, we do not shuffle the data
        loader = torch.utils.data.DataLoader(
            test_set, 
            batch_size=64, 
            shuffle=False
        )

    # 5. Return the single loader that was created
    return loader


def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """

    model = nn.Sequential(
        # 1. Flatten the 28x28 image to a 784-pixel array
        nn.Flatten(),
        
        # 2. Dense layer with 256 nodes and ReLU activation
        nn.Linear(28*28, 256),
        nn.BatchNorm1d(256),  # Batch normalization
        nn.ReLU(),
        nn.Dropout(0.2),  # Add dropout for regularization
        
        # 3. Dense layer with 128 nodes and ReLU activation
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),  # Batch normalization
        nn.ReLU(),
        nn.Dropout(0.2),  # Add dropout for regularization
        
        # 4. Dense layer with 10 nodes (the final output)
        nn.Linear(128, 10)
    )
    return model

def build_deeper_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        # 1. Flatten layer
        nn.Flatten(),
        
        # 2. Dense layer with 512 nodes and a ReLU activation.
        nn.Linear(28*28, 512),
        nn.BatchNorm1d(512),  # Batch normalization
        nn.ReLU(),
        nn.Dropout(0.25),  # Add dropout for regularization
        
        # 3. Dense layer with 256 nodes and a ReLU activation.
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),  # Batch normalization
        nn.ReLU(),
        nn.Dropout(0.25),  # Add dropout for regularization
        
        # 4. Dense layer with 128 nodes and a ReLU activation.
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),  # Batch normalization
        nn.ReLU(),
        nn.Dropout(0.2),  # Add dropout for regularization
        
        # 5. Dense layer with 64 nodes and a ReLU activation.
        nn.Linear(128, 64),
        nn.ReLU(),
        
        # 6. A Dense layer with 10 nodes.
        nn.Linear(64, 10)
    )
    return model



def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    
    # 1. Define the "tuner" (Optimizer) - Using Adam for better convergence
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 2. Set the model to "training mode"
    # This tells PyTorch to activate features like dropout (if we had them)
    model.train()
    
    # 3. Loop for T number of epochs
    for epoch in range(T):
        
        # 4. Initialize trackers for this epoch
        running_loss = 0.0
        correct_count = 0
        total_count = 0
        
        # 5. Inner loop: iterate over each batch from the train_loader
        for inputs, labels in train_loader:
            
            # --- The 5-Step Training "Dance" ---
            
            # 1. Clear the gradients
            optimizer.zero_grad()
            
            # 2. Get the model's guess (Forward pass)
            outputs = model(inputs)
            
            # 3. Calculate the error (Loss)
            loss = criterion(outputs, labels)
            
            # 4. Calculate adjustments (Backward pass)
            loss.backward()
            
            # 5. Apply adjustments (Update weights)
            optimizer.step()
            
            
            # 6. Update our trackers for printing
            
            # Add this batch's loss to the total
            running_loss += loss.item()
            
            # Get the model's actual prediction (the class with the highest score)
            _, predicted = torch.max(outputs.data, 1)
            
            # Add the number of items in this batch (usually 64)
            total_count += labels.size(0)
            
            # Count how many were correct
            correct_count += (predicted == labels).sum().item()
            
        # 7. End of Epoch: Calculate and print results
        
        # Calculate average loss and accuracy for the whole epoch
        avg_loss = running_loss / len(train_loader)
        accuracy_percent = 100 * correct_count / total_count
        
        # Print in the required format
        print(f"Train Epoch: {epoch}   Accuracy: {correct_count}/{total_count}({accuracy_percent:.2f}%)   Loss: {avg_loss:.3f}")

    # This function returns None, so we don't add a return statement.
    


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    
    # 1. Set the model to evaluation mode
    model.eval()
    
    # 2. Initialize trackers
    running_loss = 0.0
    correct_count = 0
    total_count = 0
    
    # 3. Disable gradient calculations
    with torch.no_grad():
        
        # 4. Loop through the test data
        for inputs, labels in test_loader:
            
            # 5. Get the model's guess (Forward pass)
            outputs = model(inputs)
            
            # 6. Calculate the error (Loss)
            loss = criterion(outputs, labels)
            
            # 7. Update trackers
            
            # Add this batch's loss to the total
            running_loss += loss.item()
            
            # Get the model's actual prediction (the class with the highest score)
            _, predicted = torch.max(outputs.data, 1)
            
            # Add the number of items in this batch
            total_count += labels.size(0)
            
            # Count how many were correct
            correct_count += (predicted == labels).sum().item()
            
    # 8. End of loop: Calculate final statistics
    
    # Calculate average loss and accuracy for the whole test set
    avg_loss = running_loss / len(test_loader)
    accuracy_percent = 100 * correct_count / total_count
    
    # 9. Print results based on the 'show_loss' flag
    
    if show_loss:
        # Print loss with 4 decimal places
        print(f"Average loss: {avg_loss:.4f}")
    
    # Print accuracy with 2 decimal places and % sign
    print(f"Accuracy: {accuracy_percent:.2f}%")
    


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT:
        model - the trained model
        test_images - a batch of test images (e.g., from test_loader)
        index - the index of the image within the batch to predict

    RETURNS:
        None (prints the top 3 predictions)
    """
    
    # 0. The list of class names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
    
    # 1. Set the model to evaluation mode
    model.eval()
    
    # 2. Select the single image from the batch
    # test_images[index] has shape [1, 28, 28]
    image = test_images[index]
    
    # 3. Add a "batch" dimension
    # The model expects [N, 1, 28, 28] where N is batch size
    # .unsqueeze(0) changes the shape to [1, 1, 28, 28]
    image_batch = image.unsqueeze(0)
    
    # 4. Disable gradients and get the model's prediction
    with torch.no_grad():
        
        # 5. Get the raw model output (logits)
        # The output 'logits' will have a shape of [1, 10]
        logits = model(image_batch)
        
        # 6. Convert logits to probabilities
        # We apply softmax across the 10 classes (dimension 1)
        probabilities = F.softmax(logits, dim=1)
        
        # 7. Get the top 3 probabilities and their indices
        # .squeeze() removes the [1] batch dimension, leaving a [10] tensor
        # torch.topk returns two tensors: (values, indices)
        top_probs, top_indices = torch.topk(probabilities.squeeze(), 3)
        
        # 8. Loop through the top 3 results and print them
        for i in range(3):
            prob = top_probs[i].item() * 100  # Get the probability as a percentage
            label_index = top_indices[i].item() # Get the original index
            class_name = class_names[label_index] # Look up the name
            
            # Print in the required format
            print(f"{class_name}: {prob:.2f}%")


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    
    # --- 1. Get the Data Loaders and define the "Judge" ---
    
    # Get the loader for training data
    train_loader = get_data_loader(training=True)
    
    # Get the loader for test data
    test_loader = get_data_loader(training=False)
    
    # Define the "judge" (Loss Function)
    criterion = nn.CrossEntropyLoss()

    # --- 2. Train and Evaluate the SIMPLE model ---
    
    print("--- Training the simple model ---")
    model_simple = build_model()
    train_model(model_simple, train_loader, criterion, T=40)
    
    print("\n--- Evaluating the simple model ---")
    evaluate_model(model_simple, test_loader, criterion, show_loss=True)

    # --- 3. Train and Evaluate the DEEPER model ---
    
    print("\n\n--- Training the DEEPER model ---")
    model_deeper = build_deeper_model()
    train_model(model_deeper, train_loader, criterion, T=40)
    
    print("\n--- Evaluating the DEEPER model ---")
    evaluate_model(model_deeper, test_loader, criterion, show_loss=True)

    # --- 4. Predict labels using the DEEPER model ---
    
    print("\n\n--- Predicting labels with the DEEPER model ---")
    
    # Get one batch of test images to use for prediction
    # 'iter' makes the loader an iterator, 'next' gets the first item
    test_images, test_labels = next(iter(test_loader))
    
    # Predict the first image in the batch (index 0)
    print("Prediction for image at index 0:")
    predict_label(model_deeper, test_images, 0)
    
    # Predict the second image in the batch (index 1)
    print("\nPrediction for image at index 1:")
    predict_label(model_deeper, test_images, 1)