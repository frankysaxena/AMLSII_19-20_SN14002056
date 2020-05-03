from Datasets.data_loader import DataPreprocessor


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

A_data = taskA.get_raw_dataframe(path_to_dir, taskA)
B_data = taskB.get_raw_dataframe(path_to_dir, taskB)

print(B_data)


# data_train, data_val, data_test = data_preprocessing(args...)
# # ======================================================================================================================
# # Task A
# model_A = A(args...)                 # Build model object.
# acc_A_train = model_A.train(args...) # Train model based on the training set (you should fine-tune your model based on validation set.)
# acc_A_test = model_A.test(args...)   # Test model based on the test set.


# # Clean up memory/GPU etc...             # Some code to free memory if necessary.



# # ======================================================================================================================
# # Task B
# model_B = B(args...)
# acc_B_train = model_B.train(args...)
# acc_B_test = model_B.test(args...)




# # ======================================================================================================================
# ## Print out your results with following format:
# print('TA:{},{};TB:{},{};'.format(acc_A_train, acc_A_test,
#                                                         acc_B_train, acc_B_test))

# # If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# # acc_A_train = 'TBD'
# # acc_B_test = 'TBD'