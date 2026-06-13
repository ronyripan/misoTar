# misoTarThis folder has four files.

1. misoTar.py is the main code file.
2. sample_train_df.csv is a sample training dataset.
3. sample_test_df.csv is a sample test dataset.
4. requirements.txt has all the library names and versions that are required to run the misoTar.py file.


Once the train and test files are properly designated in misoTar, the code will run smoothly.

====================================================================================================================================================================================
Dataset Information:

Both the train and test dataset has three columns.

Column name 		Description

1. miRNA/isomiR		miRNA or isomiR sequence.
2. mRNA			mRNA sequence
3. label		1 or 0. 1 if they interact; 0 if they don't.

=====================================================================================================================================================================================
Installation Guide:

1. At first, create a new Anaconda virtual environment and activate it:
	Command: conda create -n misoTar
	Command: conda activate misoTar

   Optional Steps: If you don't have pip in the new virtual environment, install it via:
	Command: conda install pip -y

2. Then install the necessary libraries:
	Command: pip install torch torchvision # PyTorch for CPU. Also if you have a GPU, install PyTorch based on your system from here "https://pytorch.org/get-started/locally/"
	Command: pip install transformers[torch]
	Command: pip install datasets
	Command: pip install scikit-learn
	
	instead of all of this, you can utilize the requirements.txt file to install all the necessary libraries via
	Command: pip freeze > requirements.txt

3. Now misoTar.py should run smoothly in any editor with the misoTar virtual environment.
