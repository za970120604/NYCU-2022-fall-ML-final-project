# NYCU 2022 fall ML final project
This repository is regarding to a real world ML competition and my proposed answer is shared. 

competition link : https://www.kaggle.com/competitions/tabular-playground-series-aug-2022

# Requirements

To reproduce the result in the notebook , please download the following package with its version : 
| Package Name                |      Version         |
| ----------------------------|----------------------|
| joblib   |      1.2.0         |
| keras        |      2.11.0        |
| matplotlib   |      3.5.1		  (optional)         |
| numpy        |      1.23.5	  (optional)        |
| pandas        |      1.5.2		  (optional)        |
| scikit-learn   |     1.2.0         |
| scipy        |      1.9.3		  (optional)        |
| seaborn   |       0.12.1	  (optional)        |
| tensorflow-gpu        |      2.11.0        |
| tqdm        |     4.64.1        |

# Training

I propose two models for solving this problem , one is based on logistic regression , which have a higher score ; the other one is based on neural network .

To train my logistic regression based model , please clone the [outer most folder](https://github.com/za970120604/NYCU-2022-fall-ML-final-project/tree/main/ML%20final%20project) , find the file named [train_logistic.ipynb](https://github.com/za970120604/NYCU-2022-fall-ML-final-project/blob/main/ML%20final%20project/train_logistic.ipynb) , and simply click run all . 

To train my neural network based model , please clone the [outer most folder](https://github.com/za970120604/NYCU-2022-fall-ML-final-project/tree/main/ML%20final%20project) , find the file named [train_NN.ipynb](https://github.com/za970120604/NYCU-2022-fall-ML-final-project/blob/main/ML%20final%20project/train_NN.ipynb) , and simply click run all .

The final model weight will be generated in the same directory as [train_logistic.ipynb](https://github.com/za970120604/NYCU-2022-fall-ML-final-project/blob/main/ML%20final%20project/train_logistic.ipynb) and [train_NN.ipynb](https://github.com/za970120604/NYCU-2022-fall-ML-final-project/blob/main/ML%20final%20project/train_NN.ipynb) successfully if just clone the [outer most folder](https://github.com/za970120604/NYCU-2022-fall-ML-final-project/tree/main/ML%20final%20project) and do what I mentioned above . The model weight file name will be train_logistic.joblib and train_NN.h5 . 


If you do not want to clone the [outer most folder](https://github.com/za970120604/NYCU-2022-fall-ML-final-project/tree/main/ML%20final%20project) , then please change 2 path in both [train_logistic.ipynb](https://github.com/za970120604/NYCU-2022-fall-ML-final-project/blob/main/ML%20final%20project/train_logistic.ipynb) and [train_NN.ipynb](https://github.com/za970120604/NYCU-2022-fall-ML-final-project/blob/main/ML%20final%20project/train_NN.ipynb) : 

1. The first line in second block : 

>df = pd.read_csv('tabular-playground/train.csv') ==> df = pd.read_csv('your/path/to/training/data')

2. The last line in last block : 

> joblib.dump(model , "train_logistic.joblib") ==>  joblib.dump(model , 'where/you/want/to/store/logistic/model weight')

> model.save('train_NN.h5') ==>  model.save('where/you/want/to/store/NN/model weight')


# Evaluation
To evaluate my logistic regression based model , please clone the [outer most folder](https://github.com/za970120604/NYCU-2022-fall-ML-final-project/tree/main/ML%20final%20project) , find the file named [eval_logistic.ipynb](https://github.com/za970120604/NYCU-2022-fall-ML-final-project/blob/main/ML%20final%20project/eval_logistic.ipynb) , and simply click run all . 

To evaluate my neural network based model , please clone the [outer most folder](https://github.com/za970120604/NYCU-2022-fall-ML-final-project/tree/main/ML%20final%20project) , find the file named [eval_NN.ipynb](https://github.com/za970120604/NYCU-2022-fall-ML-final-project/blob/main/ML%20final%20project/eval_NN.ipynb) , and simply click run all .


The final prediction on [test data] will be generated in the same directory as [eval_logistic.ipynb](https://github.com/za970120604/NYCU-2022-fall-ML-final-project/blob/main/ML%20final%20project/eval_logistic.ipynb) and [eval_NN.ipynb](https://github.com/za970120604/NYCU-2022-fall-ML-final-project/blob/main/ML%20final%20project/eval_NN.ipynb) successfully if just clone the [outer most folder](https://github.com/za970120604/NYCU-2022-fall-ML-final-project/tree/main/ML%20final%20project) and do what I mentioned above . The csv file name will be Logistic_eval.csv and NN_eval.csv . 


If you do not want to clone the [outer most folder](https://github.com/za970120604/NYCU-2022-fall-ML-final-project/tree/main/ML%20final%20project) , then please change 3 + 1 path in both [eval_logistic.ipynb](https://github.com/za970120604/NYCU-2022-fall-ML-final-project/blob/main/ML%20final%20project/eval_logistic.ipynb) and [eval_NN.ipynb](https://github.com/za970120604/NYCU-2022-fall-ML-final-project/blob/main/ML%20final%20project/eval_NN.ipynb) : 

1. The first line in second block : 

>df = pd.read_csv('tabular-playground/train.csv') ==> df = pd.read_csv('your/path/to/training/data')

2. The first line in last block : 

>model = joblib.load('Logistic Regression model weight/logistic.joblib') ==> model = joblib.load('your/path/to/logistic.joblib')

>model = tf.keras.models.load_model('Neural Network model weight/NN.h5') ==> model = tf.keras.models.load_model('your/path/to/NN.h5')

3. The 7th line in last block : 

>test_df = pd.read_csv('tabular-playground/test.csv') ==> test_df = pd.read_csv('your/path/to/testing/data')

4. For the NN model , The third to last line in last block : 
>df_subb = pd.read_csv('tabular-playground/sample_submission.csv') ==> df_subb = pd.read_csv('your/path/to/sample_submission.csv')

Also if you want to try your own trained model weight , please make sure you modify all the required path.
The performance will not be the same as I mentioned because scikit-learn and kears NN models random initialized their weights , but you can find that logistic regression model always pass the baseline(0.58990 private score) and NN model always suffer from 0.58XXX but cannot pass the baseline.

Probably you are luckier than me to find a better performance model weight !


The model weight for logistic regression: [Here](https://github.com/za970120604/NYCU-2022-fall-ML-final-project/blob/main/ML%20final%20project/Logistic%20Regression%20model%20weight/logistic.joblib)

The model weight for NN : [Here](https://github.com/za970120604/NYCU-2022-fall-ML-final-project/blob/main/ML%20final%20project/Neural%20Network%20model%20weight/NN.h5)

# Pre-trained Models
The logistic regression-based model and the neural network-based model do not contain any pretrained model , i.e. these models are hand-crafted from scartch by myself.

# Results
Our model achieves the following performance on Kaggle platform:
| Model name                  |  Top 1 Private Score |
| ----------------------------|----------------------|
| logistic regression model   |      0.59155         |
| neural network model        |      0.58833         |

>![螢幕擷取畫面 2023-01-04 140739](https://user-images.githubusercontent.com/72210437/210575615-05fb60a9-461c-421c-ae5b-94e477f25f1d.png)

>![image](https://user-images.githubusercontent.com/72210437/210576270-b8521cbf-e945-4ffd-a9bd-7c9e49d3a4b5.png)


