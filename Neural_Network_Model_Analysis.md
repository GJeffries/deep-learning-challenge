# Nueral Network Model Analysis
## Overview
The purpose of this project is to build a binary nural network model to determine whether AlphabetSoup will approve funding for applicants.
This will be undertaken by training the TensorFlow Keras library to learn from the charity_data.cvs dataset basing successfulness of applications from more than 34,000 
applicants dataset features.

The aim is to achieve a model with 75% accuracy whilst limiting model loss.

## Results
### Data Processing
Variables considered:
* **Target** - IS_SUCCESSFUL
* **Features** - APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT
* **Non-Target/Features** - EIN, NAME

Bins were then created for the APPLICATION_TYPE and CLASSIFICATION features in order to reduce noise and improve accuracy by defining the less 
frequent values as 'Other'.

![image](https://user-images.githubusercontent.com/90528880/164300705-946b6e24-b21b-4929-a264-66d01357d3c6.png)
![image](https://user-images.githubusercontent.com/90528880/164300764-deedd105-ab90-49a8-8e6e-6e12d5a74532.png)

The features were then converted to numerical (integer) data using pd.get_dummies.
![image](https://user-images.githubusercontent.com/90528880/164300949-602812b0-d114-4b19-b762-edfe4d474343.png)

This processed data was then converted into features and target arrays.

The data was then split into a training and test dataset, then standardised using StandardScaler.

### Compiling, Training, and Evaluating the Model
**Input Layer**:
* **Neurons**: 51, this is the length of the features in the pre-proccessed dataset.

**Hidden Layers**:
* **Layers** - 2, the dataset wasn't too complex and is a good starting point for a neural network.
* **Neurons** - 20 (1st layer) and 10 (2nd layer), this is 2/3rds of the difference between the input and outer layers.
* **Activation Function** - Relu, fast to compute and simple.

**Ouput Layer**
* **Neurons**: 1, as the model is binary it limits this option to only 1's and 0's
* **Activation Function** - Sigmoid, used in binary classification.



**Model Summary**:

![image](https://user-images.githubusercontent.com/90528880/164308005-944d73e8-ac1b-4d31-9203-72e8471f4eaa.png)

The model was complied using binary_crossentropy for model loss, the 'adam'optimizer, and the accuracy metric.

100 epochs were selected to train the data.


**Model Evaluation**:

![image](https://user-images.githubusercontent.com/90528880/164308412-d2fcdce7-4ea0-42a2-acb8-1995ecc22c99.png)

The results show that the initial model did not reach the desired accuracy.


### Model Optmisation
**Attempt 1**:
An additional bin for the ASK_ATM feature was created in order to further reduce noise by creating an 'Other' value.

![image](https://user-images.githubusercontent.com/90528880/164315367-c72445d4-7462-47c3-bdd0-4a9e6d44050e.png)

**Result**

![image](https://user-images.githubusercontent.com/90528880/164315486-944deba2-d921-44da-afbb-a01a9dd42cb7.png)

Suprisingly, the additional bin slightly hindered the accuracy, this was then removed for the next attempt.


**Attempt 2**:
A third hidden layer was added with additional neurons.

![image](https://user-images.githubusercontent.com/90528880/164316494-e58368d8-1ff8-49f9-be90-5d4854874c56.png)

**Result**:

![image](https://user-images.githubusercontent.com/90528880/164316574-eac850cd-5b5d-4a68-8e55-188c929b44d5.png)

Accuracy remained nearly the exact same compared to the inital model however model loss increased from 0.55 to 0.58


**Attempt 3**:
The activation function for the hidden layers was changed from 'Relu' to 'Tanh'.

**Result**:

![image](https://user-images.githubusercontent.com/90528880/164317073-1aadd83a-2a8b-4b0d-9682-47e48ec4f784.png)

No noticable changes in model accuracy or loss occured from changing the activation function.

## Summary
Each optimisation attempt only included one modification at a time to better pinpoint where improvements or impairments were made.
After all attempts were made, the desired accuracy of 75% was not acheived, the closest model was the original model with an accuracy of 0.7322.

An alternative model to use would be the Random Forest Classifier (RFC) as this model works with our binary dataset and will give a different interpreation of 
the data. As our dataset is small, the simplicity of the RFC model may be more beneficial.











