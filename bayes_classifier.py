import numpy as np
import math
import re
from sklearn.model_selection import train_test_split

   

def buildClassifier(training,spam_count,no_spam_count):
    num_features = 334
    feature_true_table = np.zeros((2,num_features),dtype = float)
    feature_false_table = np.zeros((2,num_features),dtype = float)
    # for feature_iterator in range(num_features):
    #     feature_false_spam = 0
    #     feature_true_spam = 0

    #     feature_false_notspam = 0
    #     feature_true_notspam = 0

        
    #     for data_iterator in range(np.shape(training)[0]):
    #         current_value = training[data_iterator][1]
    #         current_vector = training[data_iterator][2]
    #         print("current vector length", len(current_vector))
    #         if((current_vector[feature_iterator] == "0") & (current_value == "1")):
    #             feature_false_spam = feature_false_spam + 1
    #         elif((current_vector[feature_iterator] == "1") & (current_value == "1")):
    #             feature_true_spam = feature_true_spam + 1
    #         elif((current_vector[feature_iterator] == "0") & (current_value == "-1")):
    #             feature_false_notspam = feature_false_notspam +1
    #         elif((current_vector[feature_iterator] == "1") & (current_value == "-1")):
    #             feature_true_notspam = feature_true_notspam + 1
    #     feature_false_table[0][feature_iterator] = feature_false_spam/spam_count
    #     feature_false_table[1][feature_iterator] = feature_false_notspam/no_spam_count

    #     feature_true_table[0][feature_iterator] = feature_true_spam/spam_count
    #     feature_true_table[1][feature_iterator] = feature_true_notspam/no_spam_count
        
                
    # return feature_false_table,feature_true_table
    for data_iterator in range(np.shape(training)[0]):
        current_label = training[data_iterator][1]
        current_vector = training[data_iterator][2]
        current_vector = np.array(list(current_vector), dtype=int)
        if(current_label == "1"):
            feature_true_table[0,:] = feature_true_table[0,:] + current_vector
        elif(current_label == "-1"):
            feature_true_table[1,:] = feature_true_table[1,:] + current_vector
    feature_true_table[0,:] =  feature_true_table[0,:]/spam_count
    feature_true_table[1,:] = feature_true_table[1,:]/no_spam_count
    feature_false_table = np.ones((2,num_features),dtype = float) - feature_true_table

    return feature_true_table,feature_false_table



def Classify(testing,feature_true_table,feature_false_table):
    number_correct = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0

    testing_size = np.shape(testing)[0]
    for data_iterator in range(testing_size):
        probability_spam = (testing[:,1] == "1")/testing_size
        probability_notspam = (testing[:,1] == "-1")/testing_size

        actual_label = testing[data_iterator][1]        
        vector = testing[data_iterator][2]
        vector = np.array(list(vector), dtype=int)

        probability_selector = vector.astype(np.bool)
        conditional_probability_spam = probability_spam * np.prod((feature_true_table[0,:])[probability_selector])
        probability_selector = np.logical_not(probability_selector)
        conditional_probability_spam = conditional_probability_spam * np.prod((feature_false_table[0,:])[probability_selector])

        probability_selector = np.logical_not(probability_selector)
        conditional_probability_notspam = probability_notspam * np.prod((feature_true_table[1,:])[probability_selector])
        probability_selector = np.logical_not(probability_selector)
        conditional_probability_notspam = conditional_probability_notspam * np.prod((feature_false_table[1,:])[probability_selector])

        max_index = np.argmax([conditional_probability_spam,conditional_probability_notspam])
        predicted_label = 100
        if(max_index == 0):
            predicted_label = "1"
        elif(max_index == 1):
            predicted_label = "-1"
        
        if(predicted_label == actual_label):
            number_correct = number_correct + 1
        if((actual_label == "1") & (predicted_label =="1")):
            true_positive = true_positive + 1
        if((actual_label == "-1") & (predicted_label == "1")):
            false_positive = false_positive + 1
        if((actual_label == "1") & (predicted_label == "-1")):
            false_negative = false_negative + 1
    accuracy = number_correct/testing_size        
    precision = true_positive/(true_positive + false_positive)
    recall = true_positive/(true_positive + false_negative)
    return accuracy,precision,recall
    
def start():
    dataset = np.loadtxt("SpamInstances.txt",delimiter = " ",dtype = str)
    spam_dataset = dataset[dataset[:,1] == "1"]
    no_spam_dataset = dataset[dataset[:,1] == "-1"]
    
    spam_training,spam_testing = train_test_split(spam_dataset,test_size = 0.20)
    no_spam_training,no_spam_testing = train_test_split(no_spam_dataset,test_size = 0.20)
    
    entire_training = np.concatenate((spam_training,no_spam_training), axis = 0 )
    entire_testing = np.concatenate((spam_testing,no_spam_testing), axis = 0)

    spam_count= np.shape(spam_dataset)[0]
    no_spam_count = np.shape(no_spam_dataset)[0]
    print("Spam Count = ",spam_count)
    print("Non Spam Count = ",no_spam_count)
    
    print("number in training set" ,np.shape(entire_training)[0])
    print("number in testing set",np.shape(entire_testing)[0])
    true_table,false_table = buildClassifier(entire_training,spam_count,no_spam_count)
    print(Classify(entire_testing,true_table,false_table))
    



start()