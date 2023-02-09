# to perform methods/operations on arrays

import numpy as np 



# to clean and processing the data

import pandas as pd 



# reading spam.csv file from local system

df_sms = pd.read_csv(r'C:\Users\HP\Downloads\spam.csv', encoding = 'latin-1')



# removing unnecessary columns from df_sms 

df_sms = df_sms.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)



# renaming column names in df_sms

df_sms = df_sms.rename(columns= {"v1":"label", "v2":"sms"})



# adjusting the length of the dataframe

df_sms['length'] = df_sms['sms'].apply(len)



# replacing 'ham' with 0 and 'spam' with 1

df_sms.loc[:,'label'] = df_sms.label.map({'ham':0, 'spam':1})



# importing CountVectorizer 

from sklearn.feature_extraction.text import CountVectorizer



# Instantiating the CountVectorizer method

count_vector = CountVectorizer ()



# importing train_test_split

from sklearn.model_selection import train_test_split



# splitting the data into x_train, x_test, y_train, y_test and setting the testing data to 20 %

X_train, X_test, y_train, y_test = train_test_split(df_sms['sms'], df_sms['label'], test_size = 0.20, random_state = 1)



# fitting the training data and then returning the matrix

training_data = count_vector.fit_transform(X_train)



# importing MultinomialNB from sklearn

from sklearn.naive_bayes import MultinomialNB



# initializing MultinomialNB object

naive_bayes = MultinomialNB()



# fitting the training data

naive_bayes.fit(training_data, y_train)



# setting the pre-defined arguments of the MultinomialNB object

MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)



# reading the text message as input

msg = input("Enter the message : ")

l = [msg]



# transforming the input data and returning the matrix
 
input_data = count_vector. transform(l)



# predicting the output 

prediction = naive_bayes.predict(input_data)



# printing the predicted output

if prediction[0].item() == 1:
  
  print("Predicted output : SPAM")
  
else:
  
  print("Predicted output : HAM")



# importing the accuracy metrics 

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score




# transforming the testing data and returning the matrix
 
testing_data = count_vector. transform(X_test)




# predicting the outputs for testing_data

predictions = naive_bayes.predict(testing_data)




# printing metrics scores of the model

print('Accuracy score: {}'.format(accuracy_score(y_test, predictions)))

print('Precision score: {}'.format(precision_score(y_test, predictions)))

print('Recall score: {}'.format(recall_score(y_test, predictions)))

print('F1 score: {}'.format(f1_score(y_test, predictions)))