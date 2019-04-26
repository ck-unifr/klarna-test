# klarna machine learning case study

The task is to predict the probability of 'default' for the datapionts.

The repository contains:

* **klarna.ipynb**: the notebook contains the steps of my solution.

* **prediction.csv**: the csv file contains the prediction.

* **predict-default.py**: the python script provides the Flask web service of 'default' prediction. 

* **templates/prediction.html**: the html template used to display csv which contains the prediction on a web page.

* **prediction-screenshot.png**: the screen shot of the html which contains the prediction.


Here are the steps of making the prediction:
- Data loading
    First, the data is loaded from csv to a pandas dataframe.

- Data preprocessing. 
    This step contains missing value handling, feature transfomation (categorical features to integer), train and test sets generation.

- Train machine learning model
    This step contains model training, evaluation, and selection.
    In this case we use ROC curve and AUC score to evaluate the performance of the models. 
    This is due to the imbalance of the number of the positive and negative samples.

The details of the above steps can be found in the notebook **klarna.ipynb**.

In order to generate the prediction on the 'dataset.csv' by using Flask web service. 
- Run the script predict-default.py
- Open a web browser and enter
__[link text](http://127.0.0.1:5000/prediction)__


