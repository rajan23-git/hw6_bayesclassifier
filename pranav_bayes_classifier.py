import numpy as np
import math
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#this function builds bayesian classifer
#returns tables of posterior probabilities
def buildClassifier(training,num_features):
    spam_count = np.sum(training[:,1] == "1")
    no_spam_count = np.sum(training[:,1] == "-1")

    feature_true_table =  np.zeros((2,num_features),dtype = float)
    feature_false_table = np.zeros((2,num_features),dtype = float)
    for feature_iterator in range(num_features):
        feature_false_spam = 0
        feature_true_spam = 0

        feature_false_notspam = 0
        feature_true_notspam = 0

        #loop cycles through all datarows
        #finds feature_false_spam , counts the rows where the given email feature is false and the vector is spam
        #finds feature_true_spam , counts the rows where the given email feature is true and the vector is spam
        #finds feature_false_notspam, counts the rows where the given feature is false and the vector is not spam
        #finds feature_true_notspam, counts the rows where the given feature is true and the vector is not spam
        for data_iterator in range(np.shape(training)[0]):
            current_value = training[data_iterator][1]
            current_vector = training[data_iterator][2]
            print("current vector[7] ", (current_vector[7]))
            if((current_vector[feature_iterator] == "0") and (current_value == "1")):
                feature_false_spam = feature_false_spam + 1
            if((current_vector[feature_iterator] == "1") and (current_value == "1")):
                feature_true_spam = feature_true_spam + 1
            if((current_vector[feature_iterator] == "0") and (current_value == "-1")):
                feature_false_notspam = feature_false_notspam + 1
            if((current_vector[feature_iterator] == "1") and (current_value == "-1")):
                feature_true_notspam = feature_true_notspam + 1
        #creates conditional probabilities and organizes them into tables
        #feature_false_table, contains probabilites that specific email features are false, given either spam or not spam
        #feature_true_table, contains probabilities that specific email features are true, given either spam or not spam
        feature_false_table[0][feature_iterator] = float(feature_false_spam/spam_count)
        feature_false_table[1][feature_iterator] = float(feature_false_notspam/no_spam_count)

        feature_true_table[0][feature_iterator] = float(feature_true_spam/spam_count)
        feature_true_table[1][feature_iterator] = float(feature_true_notspam/no_spam_count)
    #laplace smoothing for zero counts    
    feature_true_table[feature_true_table == 0] = 0.1
    feature_false_table[feature_false_table == 0] = 0.1        
    return feature_true_table,feature_false_table



#this function classifies feature vectors as spam or not spam
#calculates the accuracy, precision, recall, TPR, and FPR for testing data partitions
def Classify(testing,feature_true_table,feature_false_table):
    number_correct = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    testing_size = np.shape(testing)[0]
    num_features = 334
    spam_probability = np.sum(testing[:,1] == "1")/testing_size
    notspam_probability = np.sum(testing[:,1] == "-1")/testing_size
    #this for loop multiplies together posterior probabilites
    #multiplies posterior probabilities to get P(Spam | Vector)
    # and get P(NotSpam | Vector)
    for data_iterator in range(testing_size):
        actual_label = testing[data_iterator][1]
        vector = testing[data_iterator][2]
        #spam_probability_conditional is the conditional probability P(Spam | Vector)
        #it is the probability that it is classified as Spam, given all the values in the feature vector
        spam_probability_conditional = spam_probability
        #notspam_probability_conditional is the conditional probability P(Not Spam| Vector)
        #it is the probability that it is classified as Not Spam, given all the values in the feature vector
        notspam_probability_conditional = notspam_probability
        #multiplies together posterior probabilities depending on whether email feature is true or not
        for feature_iterator in range(num_features):
            if(vector[feature_iterator] == "0"):
                spam_probability_conditional = spam_probability_conditional * feature_false_table[0][feature_iterator]
                notspam_probability_conditional = notspam_probability_conditional * feature_false_table[1][feature_iterator]
            elif(vector[feature_iterator] == "1"):
                spam_probability_conditional = spam_probability_conditional * feature_true_table[0][feature_iterator]
                notspam_probability_conditional = notspam_probability_conditional * feature_true_table[1][feature_iterator]
        #finds the highest conditional probability and classifies data vector as spam or not spam     
        max_index = np.argmax([spam_probability_conditional,notspam_probability_conditional])
        predicted_label = 100
        if(max_index == 0):
            predicted_label = "1"
        elif(max_index == 1):
            predicted_label = "-1"
        print("actual label",actual_label,"predicted_labe1",predicted_label)
        
        if(predicted_label == actual_label):
            number_correct = number_correct + 1
        if((actual_label == "1") and (predicted_label =="1")):
            true_positive = true_positive + 1
        if((actual_label == "-1") and (predicted_label == "-1")):
            true_negative = true_negative + 1
        if((actual_label == "-1") and (predicted_label == "1")):
            false_positive = false_positive + 1
        if((actual_label == "1") and (predicted_label == "-1")):
            false_negative = false_negative + 1
    #calculates accuracy and other measurements based on classification counts
    accuracy = number_correct/testing_size
    #adding 1 to denominator to catch divide by zero error        
    precision = true_positive/(true_positive + false_positive + 1)
    recall = true_positive/(true_positive + false_negative + 1)
    true_positive_rate = true_positive/(true_positive + false_negative + 1)
    false_positive_rate = false_positive/(false_positive + true_negative + 1)
    print("true_positive",true_positive,"false_postive",false_positive,"false negative",false_negative,"true negative",true_negative)
    return accuracy,precision,recall,true_positive_rate,false_positive_rate        

    
def start():
    dataset = np.loadtxt("SpamInstances.txt",delimiter = " ",dtype = str)
    spam_dataset = dataset[np.where(dataset[:,1] == "1")]
    #dataset split into spam and not spam instances
    no_spam_dataset = dataset[np.where(dataset[:,1] == "-1")]
    spam_count= np.shape(spam_dataset)[0]
    no_spam_count = np.shape(no_spam_dataset)[0]
    
    #80-20 split for spam and not spam sets
    eighty_percent_spam = math.floor(spam_count * 0.80)
    eighty_percent_notspam = math.floor(no_spam_count * 0.80)
    spam_training = spam_dataset[np.arange(0,eighty_percent_spam),:]
    spam_testing = spam_dataset[np.arange(eighty_percent_spam,spam_count),:]
    no_spam_training = no_spam_dataset[np.arange(0,eighty_percent_notspam),:]
    no_spam_testing = no_spam_dataset[np.arange(eighty_percent_notspam,no_spam_count),:]
    
    #80 percent spam and not spam combined into training set
    #20 percent spam and not spam combined into testing set
    entire_training = np.concatenate((np.array(spam_training),np.array(no_spam_training)), axis = 0 )
    entire_testing = np.concatenate((np.array(spam_testing),np.array(no_spam_testing)), axis = 0)
    #randomly permute order of training and testing sets
    np.random.shuffle(entire_training)
    np.random.shuffle(entire_testing)
   
    print("Spam Count = ",spam_count)
    print("Non Spam Count = ",no_spam_count)
    
    print("number in training set" ,np.shape(entire_training)[0])
    print("number in testing set",np.shape(entire_testing)[0])
    # true_table,false_table = buildClassifier(entire_training,334)
    # print(true_table)
    # print(false_table)
    # print(true_table + false_table)
    # print(Classify(entire_testing,true_table,false_table))
    accuracy_array = np.array([])
    precision_array = np.array([])
    recall_array = np.array([])
    TPR_array = np.array([])
    FPR_array = np.array([])


    training_partition_sizes = np.arange(100,2100,100)
    testing_partition_sizes = np.arange(25,525,25)
    print("training sizes",training_partition_sizes)
    print("testing sizes",testing_partition_sizes)
    #for loop records the accuracy, precision, recall,TPR, and FPR
    #of different paritions of training and testing data
    #graphs these measurments for each classifier model
    for iterator in range(len(training_partition_sizes)):
        training_partition_size = training_partition_sizes[iterator]
        testing_partition_size = testing_partition_sizes[iterator]
        
        training_partition = entire_training[np.arange(0,training_partition_size),:]
        testing_partition = entire_testing[np.arange(0,testing_partition_size),:]

        feature_true_table,feature_false_table = buildClassifier(training_partition,334)
        accuracy,precision,recall,true_positive_rate,false_positive_rate = Classify(testing_partition,feature_true_table,feature_false_table)
        
        accuracy_array = np.append(accuracy_array,accuracy)
        precision_array = np.append(precision_array,precision)
        recall_array = np.append(recall_array,recall)
        TPR_array = np.append(TPR_array,true_positive_rate)
        FPR_array = np.append(FPR_array,false_positive_rate)
    #graphs Precision vs Recall for different training-testing partitions
    plt.figure(1)
    plt.title("ROC Curve For Models")
    plt.xlabel("Recall")   
    plt.ylabel("Precision")
    plt.plot(recall_array,precision_array)
    plt.scatter(recall_array,precision_array)

    #graphs False Positive Rate vs True Positive Rate for different training-testing partitions
    plt.figure(2)
    plt.title("ROC Curve For Models")
    plt.xlabel("1-Specificity(False Positive Rate)")   
    plt.ylabel("Sensitivity(True Positive Rate")
    plt.plot(FPR_array,TPR_array)
    plt.scatter(FPR_array,TPR_array)

    #graphs Accuracies vs different training-testing partition sizes
    plt.figure(3)
    plt.title("Accuracy For Different Training Partition Sizes")
    plt.xlabel("Training Partition Sizes")
    plt.ylabel("Accuracy")
    plt.plot(training_partition_sizes,accuracy_array)
    plt.scatter(training_partition_sizes,accuracy_array)
    plt.show()


start()