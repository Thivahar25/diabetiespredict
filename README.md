# diabetiespredict
This code is an implementation of logistic regression, a binary classification algorithm, in Python


    Libraries:
          numpy for numerical operations.
          train_test_split from sklearn.model_selection for splitting the dataset into training and testing sets.
          LogisticRegression from a custom module logisticRegression.ipynb, presumably containing the logistic regression implementation.
           pandas for handling and reading data from a CSV file.
    Sigmoid Function (sigmoid): 
         This function implements the sigmoid activation function, which takes an input 'x' and returns the output of the sigmoid function, which is a value between 0 and 1. It's a crucial component in logistic regression for mapping real numbers to probabilities.


    Initialization: 
         When an instance of this class is created, it can be configured with a learning rate (lr) and the number of iterations (n_iters) for training. By default, lr is set to 0.001, and n_iters is set to 1000. It also initializes the weights and bias to None.

    Fit Method (fit): 
         This method is used to train the logistic regression model. It takes training data X and corresponding target values y as input. Inside the method, it initializes the model's weights and bias, then performs gradient descent for the specified number of iterations to optimize the model's parameters. The goal is to find the parameters that best fit the given data. The code calculates the gradients of the weights and bias and updates them using the learning rate and the difference between the predicted values and the actual target values.

   Predict Method (predict): 
         This method is used for making predictions after the model has been trained. It takes input data X, calculates the linear prediction (dot product of input and weights plus bias), applies the sigmoid function to get a probability, and then converts these probabilities to binary class predictions (0 or 1) based on a threshold of 0.5. The predicted class labels are returned as a list.

         

     Training model
            Reading Data:
            It reads a cleaned dataset from a CSV file called "diabetes.csv" into a Pandas DataFrame (learn_d).
            It splits the dataset into input features (X) and the target variable ("Outcome") (y).

    Train-Test Split:
        It further splits the data into training and testing sets using the train_test_split function. The training set consists of 80% of the data, and the random seed is set to ensure reproducibility.

    Training the Logistic Regression Model:
        An instance of the LogisticRegression class is created with a learning rate of 0.0001 and a maximum of 100,000 iterations.
        The logistic regression model is trained on the training data using the fit method.

    Making Predictions:
        Predictions are made on the test data using the predict method and stored in Y_pred.

    Accuracy Calculation:
        An accuracy function is defined to calculate the accuracy of the model's predictions. It compares the predicted values (Y_pred) to the actual test labels (y_test) and computes the ratio of correct predictions to the total number of predictions.

    User Interaction (Main Loop):
        The code enters a while loop that allows the user to interact with the program. The user can choose from the following options:
            Option 1: Evaluate the accuracy of the model on the test data and print it.
            Option 2: Provide input features for a new data point to predict diabetes outcome (0 or 1).
            Option 3: Exit the program.

    Option 2 - Input Features:
        If the user chooses option 2, they are prompted to enter various health-related input features (e.g., number of pregnancies, glucose level, blood pressure, etc.) for a new data point.
        These input features are used to create an input array (x_input).
        The logistic regression model (LR) is used to predict the diabetes outcome for the new input, and the result is printed.

