# AMLSII_19-20_SN14002056

## Structure

A and B are the corresponding folders where the classes are stored.
Models have been saved as distinct files which were the last best trained models when I ran through the dataset.

### Files

To run their respective tasks:
* A/A_semantic_analysis.py
* B/B_topic_semantic_analysis.py

For data pre-processing and loading data for the classes:
* Datasets/data_loader.py

For ad-hoc testing and gathering graph data results for both Tasks A and B:
* helper_notebook.ipynb

You can run the Jupyter notebooks as is to display results. These notebooks were essentially my 'working out' while I was going through the different models and tasks.

Main program execution file:
* main.py
    
Total execution time should be about 10-15 minutes.

## Usage

### Requirements

Please ensure you have these dependencies:

vaderSentiment
pandas
nltk
tensorflow
keras
numpy
matplotlib
scikit-learn

These have all been listed in the `requirements.txt` file

### Run

This code has been tested on Python 3.7+. It should work fine on any Python3 flavour, however Python2 is not advised.

Make sure you have set Python 3 as default.
Navigate to your preferred directory

1. ` git clone << this repository >>`
2. `pip install -r requirements.txt`
3. `vi  main.py` 
4. Edit the `path_to_dir` variable accordingly
5. `python main.py`

Or if you already have all the above dependencies installed, then just change the path directory  and run python main.py


