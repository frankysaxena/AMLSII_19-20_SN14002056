from Datasets.data_loader import DataPreprocessor
from A.A_semantic_analysis import SemanticAnalysis
from B.B_topic_semantic_analysis import TopicSemanticAnalysis

import gc

# # ======================================================================================================================
# # Please change this path to the relevant path you're running this code from 

path_to_dir = '/Users/fsaxena/Documents/UCL/amls2/AMLSII_19-20_SN14002056/AMLSII_19-20_SN14002056'

# # ======================================================================================================================
# # Constants

taskA = 'taskA'
taskB = 'taskB'

# ======================================================================================================================
# Data preprocessing

taskA = DataPreprocessor(path_to_dir, taskA)
taskB = DataPreprocessor(path_to_dir, taskB)


A_df = taskA.get_raw_dataframe(path_to_dir, taskA)
B_df = taskB.get_raw_dataframe(path_to_dir, taskB)


# # ======================================================================================================================
# # Task A

A_train, A_test = taskA.split_train_test(A_df)
A_train_features, A_train_labels = taskA.convert_to_arrays(A_train)
A_test_features, A_test_labels = taskA.convert_to_arrays(A_test)

SemanticAnalyzer = SemanticAnalysis(A_train_features, A_train_labels)

words_list, documents = SemanticAnalyzer.featureCreator(A_train_features, A_train_labels)
word_features = SemanticAnalyzer.get_word_features(words_list)
train_set, test_set = SemanticAnalyzer.train_val_generator(documents, word_features)

"""
Bernoulli Naive Bayes was seen to be the best model, however if different classes need to be run, please refer to the 
A_semantic_analysis script where other functions are also defined for the class that run different models/classifiers
"""

acc_A_test = SemanticAnalyzer.BNBClf(train_set, test_set) 

print("-----------------------------------------------")
print("Cleaning memory before next task... ")

gc.collect() # Clean up memory

print("Cleared!")


# # ======================================================================================================================
# # Task B
B_train, B_test = taskB.split_train_test(B_df)
B_train_features, B_train_labels = taskB.convert_to_arrays(B_train)
B_test_features, B_test_labels = taskB.convert_to_arrays(B_test)

Topic_SemanticAnalyzer_B = TopicSemanticAnalysis(B_train_features, B_train_labels)

words_list, documents = SemanticAnalyzer.featureCreator(B_train_features, B_train_labels)
word_features = SemanticAnalyzer.get_word_features(words_list)
train_set, test_set = SemanticAnalyzer.train_val_generator(documents, word_features)


acc_B_test = Topic_SemanticAnalyzer_B.LogClf(train_set, test_set)

print("-----------------------------------------------")
print("Cleaning memory before next task... ")

gc.collect() # Clean up memory

print("Cleared!")

# # ======================================================================================================================
## Print out your results with following format:
print("-----------------------------RESULTS------------------------------------")

data = [['Task', 'Test Accuracy (%)'], 
        ['A', str(acc_A_test)], 
        ['B', str(acc_B_test)]]

col_width = max(len(word) for row in data for word in row) + 2  # padding
for row in data:
    print("".join(word.ljust(col_width) for word in row))