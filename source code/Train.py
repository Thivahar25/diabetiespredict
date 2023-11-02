# === libraries === 
import numpy as np
from sklearn.model_selection import train_test_split
from logisticRegression import LogisticRegression
import pandas as pd

# == custom standard ==


# === read the cleaned data === 
learn_d = pd.read_csv("diabetes.csv")

X = learn_d.drop('Outcome', axis=1)  # Assuming 'Outcome' is the diabetes label
y = learn_d['Outcome']
#train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# === calling logistic regression to train itself ===
LR = LogisticRegression(lr = 0.0001, n_iters= 100000)
LR.fit(X_train, y_train)

Y_pred = LR.predict(X_test)

# === Accuracy ===
def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test)/len(y_test)

# result = accuracy(Y_prediction, Y_test)
# print(result)

# === main === 
def new_func():
    option = int(input())
    return option
while(True):
    print("Select an option: \n 1) Evaluation\n 2) Give input\n 3) Exit Program")
    option = new_func()

    if(option == 1):
        acc = accuracy(y_test, Y_pred)
        print("Accuracy is:",acc)

    elif(option == 2):
        pregnancies = float(input("Please enter number of pregnancy you had: "))
        glucose = float(input("Please enter your glucose rate ==> mg/dl: "))
        bloodPressure = float(input("Please enter your blood pressure ==> mm/Hg: "))
        skinThickness = float(input("Please enter thickness of your skin ==> (0,99): "))
        insulin = float(input("Please enter insulin level of your blood ==> mm: "))
        bmi = float(input("Please enter you BMI: "))
        diabetesPedigreeFunction = float(input("Please enter Diabetes pedigree function: "))
        age = float(input("Please enter your age: "))

        x_input = [[pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, diabetesPedigreeFunction, age]]
        prob = LR.predict(x_input)
        print("If outcome is 1 diabetes is Positive")
        print("If outcome is 0 diabetes is Negative")
        print("Outcome: ", prob[0])

    elif(option == 3):
        print("exit")
        break
    
1
