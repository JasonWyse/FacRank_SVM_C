
	   
						  |---run.py #4
						  |
						  |---path.py #5
				|---bin---|
				|		  |---mysql.py #6
				|		  |
				|		  |---evaluate.py #7
				|
				|		 		  |---svm_rank_learn #8
				|---source_code---|
				|				  |---svm_rank_classify #9
				|
				|
   svm_struct---|---data #1
				|
				|
				|
				|---localdatasets #2
				|
				|
				|		   |---mq2007 #10
				|---eval---|
				|		   |---mq2008 #11
				|
				|
				|---log #3
				
1.	data
		Intermediate files.

2.	localdatasets
		We convert standard dataset into local data in order to fit the data entry formats of a specific algorithm tool.

3. 	log
		Log files. 
		
4.	**run.py**
		You can use this script to reproduce our experiment, including run the algorithm, tune parameters, 
	output results into files and database. It includes path.py, mysql.py and evaluate.py.

5.	path.py
		This file defines the path's struct of our project. A Class named PathManager is provided.
	
6.	mqsql.py
		In order to manage database processing, we provide some Classes and functions in this file.

7.	evaluate.py
		This file includes our measure functions to calculate MAP and NDCG. 
		
8.	svm_rank_learn
		This is the execute file to train model on the input data.
		
9.	svm_rank_classify
		This is the execute file to create score file with the data and model.

10.	mq2007
		This file shows the best performance on Mq2007, including P@N, MAP, NDCG@N, MeanNdcg on each Fold.
		
11.	mq2008
		This file shows the best performance on Mq2008, including P@N, MAP, NDCG@N, MeanNdcg on each Fold.
