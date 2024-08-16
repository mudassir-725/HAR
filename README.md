# HAR
This study develops a CNN-LSTM model for Human Activity Recognition using accelerometer and gyroscope data. Achieving over 95% accuracy, it combines LSTM's temporal capabilities with CNN's spatial features.

## Agenda

### 1. Analyzing the Data (EDA)

-  Some Analysis on Data Set below:
-  Here, first I perform EDA on Expert generated Data set. We try to understand the data then create some machine Learning model on top of it.
-  Start with loading the feature.txt file then train data and test data and analysis these data.

-  Total Data point and feature count in train and test data:
   ```python
   train = pd.read_csv('UCI_HAR_dataset/csv_files/train.csv')
   test = pd.read_csv('UCI_HAR_dataset/csv_files/test.csv')
   print(train.shape, test.shape)
   ```
   ```
   Output: (7352, 564) (2947, 564)
   ```
-  __investigate participants activity durations__: Since the dataset has been created in a scientific environment nearly equal preconditions for the participants can be assumed. Let us investigate their activity durations.
   ```python
   sns.set_style('whitegrid')
   plt.rcParams['font.family'] = 'Dejavu Sans'
   plt.figure(figsize=(16,8))
   plt.title('Data provided by each user', fontsize=20)
   sns.countplot(x='subject',hue='ActivityName', data = train)
   plt.show()
   ```
   ![npic5](https://user-images.githubusercontent.com/49862149/91143901-dcc97e00-e6d0-11ea-9a7b-ae5df0b94e82.jpg)
 
   *  Nearly all participants have more data for walking upstairs than downstairs. Assuming an equal number of up- and down-walks the participants need longer walking upstairs. 
   *  We know that we have six class classification so, we have the big problem is to know or check is there any imbalanced in the data. And after plotting above graph we can say data is balanced.
   *  We have got almost same number of reading from all the subject. 
   
-  __Next question is how many data points do I have per class level.__
      ![npic6](https://user-images.githubusercontent.com/49862149/91144223-53ff1200-e6d1-11ea-9c18-06368d397d33.jpg)
      *  Data is almost balanced.
      *  Although there are fluctuations in the label counts, the labels are quite equally distributed.
      *  Assuming the participants had to walk the same number of stairs upwards as well as downwards and knowing the smartphones had a constant sampling rate, there should be the same amount of datapoints for walking upstairs and downstairs.
      *  Disregarding the possibility of flawed data, the participants seem to walk roughly 10% faster downwards.
      
-  Now time is to know __Static and Dynamic Activities of Human__:
   *  In static activities (sit, stand, lie down) motion information will not be very useful.
   *  In the dynamic activities (**Walking, WalkingUpstairs,WalkingDownstairs**) motion info will be significant.
   *  Here we are using “__tBodyAccMagmean__” (*tBody acceleration magnitude feature mean value*) function to plot the graph for better understanding of *Static and Dynamic Activities of Human*.
   ![npic7](https://user-images.githubusercontent.com/49862149/91144681-033be900-e6d2-11ea-93bb-33b5c4137407.jpg)
   
   *  We can see __tbodyAccMagmean__ feature separate very well the *Static and Dynamic Activities of Human*.
   
 - Now we plot the __Static and Dynamic Activities of Human on Box plot__ for understanding:
      ![npic8](https://user-images.githubusercontent.com/49862149/91144988-69c10700-e6d2-11ea-927c-622ceb1d9693.jpg)
   
   *  If __tAccMean__ is __< -0.8__ then the Activities are either *Standing* or *Sitting* or *Laying*.
   *  If __tAccMean__ is __> -0.6__ then the Activities are either *Walking* or *WalkingDownstairs* or *WalkingUpstairs*.
   *  If __tAccMean > 0.0__ then the Activity is __WalkingDownstairs__.
   *  We can classify __75%__ the Acitivity labels with some errors.
 
-  Position of __GravityAccelerationComponants__ also matters:
      
      ![npic9](https://user-images.githubusercontent.com/49862149/91145629-64b08780-e6d3-11ea-8ebc-035929509c09.jpg)

   *  If angleX, __gravityMean > 0__ then Activity is *Laying*.
   *  We can classify all datapoints belonging to Laying activity with just a single if else statement.
   
-  Apply **t-sne** on the data to know __how much the Activities are Separable?__ We know that we have 561-Dimension expert engineered feature now apply TSNE on these features to see how much these features are helpful. 
      ![npic10](https://user-images.githubusercontent.com/49862149/91146092-089a3300-e6d4-11ea-87b8-a325922ed856.jpg)
      *  We can clearly see the *TSNE cluster*, All the Activity are clean separate **except "Standing" and "Sitting"**.
      

### 2. Machine Learning Models:
-  __Important Note as we discussed previous__: I used the __561 expert engineered features__ and we will *apply classical Machine Learning Model* on top of it.
-  The Machine Learning Model which I applied are:
#### a. Logistic Regression
-  Logistic regression is a linear model for classification. In this model, the probabilities describing the possible outcomes of a single trial are modeled using a logistic function. The logistic function is a sigmoid function, which takes any real input and outputs a value between 0 and 1, and hence is ideal for classification.
      
   When a model learns the training data too closely, it fails to fit new data or predict unseen observations reliably. This condition is called overfitting and is countered, in one of many ways, with ridge (L2) regularization. Ridge regularization penalizes model predictors if they are too big, thus enforcing them to be small. This reduces model variance and avoids overfitting.

-  __Hyperparameter Tuning__:
Cross-validation is a good technique to tune model parameters like regularization factor and the tolerance for stopping criteria (for determining when to stop training). Here, a validation set is held out from the training data for each run (called fold) while the model is trained on the remaining training data and then evaluated on the validation set. This is repeated for the total number of folds (say five or 10) and the parameters from the fold with the best evaluation score are used as the optimum parameters for the model.

####  b. Linear SVC
-  The objective of a __Linear SVC__ (Support Vector Classifier) is to fit to the data you provide, returning a "best fit" hyperplane that divides, or categorizes, your data. From there, after getting the hyperplane, you can then feed some features to your classifier to see what the "predicted" class is.

####  c. Kernal SVM
-  __SVM__ algorithms use a set of mathematical functions that are defined as the __kernel__. The function of __kernel__ is to take data as input and transform it into the required form. Different SVM algorithms use different types of kernel functions. These functions can be different types.
####  d.	Decision Tree
-  Decision trees is a hierarchical model also known as classiﬁcation and regression trees. They have the property of predicting response from data. The attributes of the decision trees are mapped into nodes. The edges of the tree represent the possible output values. Each branch of the tree represents a classiﬁcation rule, from the root to the leaf node.
-  This method has been used for several tasks in the ﬁeld of pattern recognition and machine learning as a predictive model. The main goal is to predict the next value given several input variable.
####  e.	Random Forest Classifier
-  Random Forest is an outfit of unpruned demand or descends like bootstrapping algorithm with various decision trees. Each tree depends upon the estimations of the vector picked unpredictably and independently. Random Forest reliably gives an immense improvement than the single tree classifier. Each tree is fabricated using the algorithm.
####  f.	Gradient Boosted
-  Gradient boosting is an AI method for relapse and order issues, which creates an expectation model as a group of powerless forecast models, normally choice trees. The goal of any directed learning algorithm is to characterize a misfortune work and limit it. Gradient boosting machines are in light of a ensemble of choice trees where numerous weak learner trees are utilized in mix as a group to give preferred forecasts over singular trees. Boost has unrivalled regularization and better treatment of missing qualities and also much improved proficiency.
-  ![#FF5733](https://via.placeholder.com/8x24/FF5733/000000?text=+) __NOTE__: I am trying to run the "GradientBoostingClassifier()" with "GridSearchCV", but my system is not supported this pice of code.

### 3. Deep Learning Models:
-  Now, I created __LSTM based Deep learning Model__ on the __Raw time series Data__.
-  HAR is one of the time series classification problem. In this project various machine learning and deep learning models have been worked out to get the best final result. In the same sequence, we can use LSTM (long short-term memory) model of the Recurrent Neural Network (RNN) to recognize various activities of humans like standing, climbing upstairs and downstairs etc.
-  __LSTM model__ is a type of recurrent neural network capable of learning order dependence in sequence prediction problems. This model is used as this helps in remembering values over arbitrary intervals.
-  I applied LSTM as follows: 
   *  __1-Layer of LSTM__
   *  __2-Layer of LSTM with more hyperparameter tuning__
   
###   4.	Results & Conclusion
-  ![#86b300](https://via.placeholder.com/5x27/86b300/000000?text=+) For below table we are comparing all the **ML model Accuracy score**.

| Model Name | Features | Hyperparameter Tuning | Accuracy Score |
| ---------- | -------- | ------ | -------- |
| Logistic Regression | `Expert generated Feature` | Done | **95.83%** |
| Linear SVC | `Expert generated Feature` | Done | **96.47%** |
| RBF SVM classifier | `Expert generated Feature` | Done | **96.27%** |
| Decision Tree | `Expert generated Feature` | Done | **86.46%** |
| Random Forest | `Expert generated Feature` | Done | **92.06%** |

-  We can choose __Linear SVC__ or __rbf SVM classifier__ or __Logistic Regression__ as our best model while applying ML Classical Model.

-  ![#86b300](https://via.placeholder.com/5x27/86b300/000000?text=+) For Below table we are comparing **Deep Learning LSTM Model**.

| Model Name | Features | Hyperparameter Tuning | crossentropy | Accuracy Value |
|---------- | ---------- | -------- | ------ | -------- |
| LSTM With 1_Layer(neurons:32) | `Raw time series Data` | Done | **0.47** | **0.90%** |
| LSTM With 2_Layer(neurons:48, neurons:32) | `Raw time series Data` | Done | **0.39** | **0.90%** |
| LSTM With 2_Layer(neurons:64, neurons:48) | `Raw time series Data` | Done | **0.27** | **0.91%** |

-  When we talking about LSTM Model, here with LSTM we are using simple RAW data(in ML model we are using Single engineered data made by an expert), but we can see the result without any FE data, LSTM perform very-very well and got highest 91% accuracy with 2_layer LSTM with hyperparameter Tuning and also when we are increasing LSTM layer and Hyperparameter Tuning the cross-entropy value is decreasing and Accuracy is increasing.

## Technical Aspect
This project is divided into four part:
   1.	I have done EDA in __first part__.
   2.	Created Classical Machine Learning Prediction Models on top of expert generated features in __second part__.
   3.	Created __LSTM based Deep learning Model__ on top of __Raw time series Data__ in __third part__.
   4.	Machine Learning and Deep Learning Model Comparison and conclusion in __fourth part__.


## Installation
The Code is written in Python 3.7. If you don't have Python installed you can find it [**here**](https://www.python.org/downloads/ "Install Python 3
.7"). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip.

- all the code for the article is available with this __REPOSITORIES__..

  *How To*
  
    * Install Required Libraries
    
      ```python
      pip3 install pandas
      pip3 install numpy
      pip3 install scikit-learn
      pip3 install matplotlib
      pip3 install keras
      ```

## Quick overview of the dataset

-  Accelerometer and Gyroscope readings are taken from 30 volunteers (referred as subjects) while performing the following 6 Activities.
   1.	Walking
   2.	WalkingUpstairs
   3.	WalkingDownstairs
   4.	Standing
   5.	Sitting
   6.	Lying.
 
-  Readings are divided into a window of 2.56 seconds with 50% overlapping.
-  Accelerometer readings are divided into gravity acceleration and body acceleration readings, which has x,y and z components each.
-  Gyroscope readings are the measure of angular velocities which has x,y and z components.
-  Jerk signals are calculated for BodyAcceleration readings.
-  Fourier Transforms are made on the above time readings to obtain frequency readings.
-  Now, on all the base signal readings., mean, max, mad, sma, arcoefficient, engerybands,entropy etc., are calculated for each window.
-  We get a feature vector of 561 features and these features are given in the dataset.
-  Each window of readings is a datapoint of 561 features.

