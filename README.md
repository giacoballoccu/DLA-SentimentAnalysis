# DLA-SentimentAnalysis

## How to run using precomputed models
0. Open a terminal in the desiderate path and download the project folder using the command.   
``` git clone https://github.com/giacoballoccu/DLA-SentimentAnalysis ```
1. With the same terminal enter the project folder and install the requirements using this commands. 
```cd DLA-SentimentAnalysis
pip3 install -r requirements.txt
```
2. Wait the installing of all requirements
3. You are ready to evaluate the models, you can run the evalutation using the command:  
```python3 LoadAndEvaluate.py```
## How to run training the model by yourself
0. Open a terminal in the desiderate path and download the project folder using the command.  
``` git clone https://github.com/giacoballoccu/DLA-SentimentAnalysis```
1. With the same terminal enter the project folder and install the requirements using this commands. 
```cd DLA-SentimentAnalysis
pip3 install -r requirements.txt
```
2. Download and extract the dataset in the project folder DLA-SentimentAnalysis/dataset from [here](https://www.kaggle.com/bittlingmayer/amazonreviews) 
3. You should now have a directory called archive in the dataset folder, pick the file "train.ft.txt.bz2" and move it one level down in the "DLA-SentimentAnalysisDataset" folder. 
4. You are ready to train, for starting the training and evaluation paste the following command in the terminal you opened in the step 0 (It must be located in the "DLA-SentimentAnalysis" folder). 
```python3 Train.py```  
