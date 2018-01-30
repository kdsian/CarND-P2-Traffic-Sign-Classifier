# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

본 프로젝트의 목표 및 수행한 step은 아래와 같습니다.:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/dataset_train.png "Visualization"
[image1_1]: ./examples/dataset_test.png "Visualization"
[image1_2]: ./examples/dataset_histogram_train.png "Visualization-hist"
[image2]: ./examples/dataset_train_gray.png "Grayscaling"
[image3]: ./examples/dataset_train_nom.png "Normalization"
[image4]: ./examples/lenet.png "lenet"
[image5]: ./examples/dataset_new.png "Traffic Sign"
[image6]: ./examples/dataset_new_result.png "prediction result"
[image7]: ./examples/dataset_new_ana.png "probability result"




## Rubric Points
프로젝트 구현 과정에서 아래 사항에 대해 고려하면서 작업을 수행했습니다. 
[rubric points](https://review.udacity.com/#!/rubrics/481/view) 
각각의 항목이 어떻게 구현되었는지는 아래 설명을 통해 보여드리겠습니다.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/kdsian/CarND-P2-Traffic-Sign-Classifier)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

summary statistics of the traffic 을 위해서 numpy library를 주로 사용하였습니다.

sign data의 구조는 아래와 같습니다.:

* The size of training set : 34799
* The size of the validation set : 4410
* The size of test set : 12630
* The shape of a traffic sign image : (32,32,3)
* The number of unique classes/labels in the data set : 43

#### 2. Include an exploratory visualization of the dataset.

`plot_images` 라는 함수를 만들어서 dataset에 대해 임의의 sample 들을 출력해서 확인 할 수 있도록 구성하였습니다.

동시에 label을 출력하도록하여 영상 sample과 label 의 매칭이 제대로 되어 있는지도 확인 하였습니다.

아래 예시는 train sample에 대한 예시 입니다.

![alt text][image1]

아래 예시는 test sample에 대한 예시 입니다.

![alt text][image1_1]

label별 train sample의 분포를 살펴보면 아래와 같습니다.
x축이 label 값이며 y축이 갯수입니다.

![alt text][image1_2]

| ClassId         		|     SignName	        					| 
|:---------------------:|:---------------------------------------------:| 
|0|	Speed limit (20km/h)|
|1|	Speed limit (30km/h)
|2|	Speed limit (50km/h)
|3|	Speed limit (60km/h)
|4|	Speed limit (70km/h)
|5|	Speed limit (80km/h)
|6|	End of speed limit (80km/h)
|7|	Speed limit (100km/h)
|8|	Speed limit (120km/h)
|9|	No passing
|10|	No passing for vehicles over 3.5 metric tons
|11	|Right-of-way at the next intersection
|12|	Priority road
|13|	Yield
|14|	Stop
|15|	No vehicles
|16|	Vehicles over 3.5 metric tons prohibited
|17|	No entry
|18|	General caution
|19|	Dangerous curve to the left
|20|	Dangerous curve to the right
|21|	Double curve
|22|	Bumpy road
|23|	Slippery road
|24|	Road narrows on the right
|25|	Road work
|26|	Traffic signals
|27|	Pedestrians
|28|	Children crossing
|29|	Bicycles crossing
|30|	Beware of ice/snow
|31|	Wild animals crossing
|32|	End of all speed and passing limits
|33|	Turn right ahead
|34|	Turn left ahead
|35|	Ahead only
|36|	Go straight or right
|37|	Go straight or left
|38|	Keep right
|39|	Keep left
|40|	Roundabout mandatory
|41|	End of no passing
|42|	End of no passing by vehicles over 3.5 metric tons


### Design and Test a Model Architecture

#### 1. Describe how you *preprocessed* the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

##### 1) Gray 변환
첫 번째 과정으로 grayscale로 변환하는 작업을 수행했습니다.

수행 예시는 아래와 같습니다.

기존 line detection 과제의 연장선으로 수행하다보니 gray scale로 변환 하는 작업을 수행하였으나, 표지판의 경우 색의 정보도 중요하다고 생각합니다.

추후에 기회가 된다면 R,G,B 입력을 받는 모델로 수정해보도록 하겠습니다.

![alt text][image2]

##### 2) 정규화 과정

두 번째 과정으로는 normalize 작업을 수행했습니다.

입력 데이터의 정규화는 중요한 요소로 알려져있고 실제로

수행 유무에 대한 성능 차이가 심하게 나타나여 필수적으로 구현했습니다.

수행 후 예제는 아래와 같습니다. 



![alt text][image3]

데이터의 범위를 0~255에서 -1~1로 변환한 것이기 때문에

영상에서는 크게 차이를 느낄 수 없습니다.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

기존 강의에서 배운 LeNet 모델을 그대로 가져와서 사용했습니다.

LeNet 모델 구조는 아래와 같습니다.

![alt text][image4]

자세한 각각의 layer들에 대한 정보는 아래 표와 같습니다.


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3	    | 1x1 stride, Valid padding, outputs 10x10x16      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten       		| input 5x5x16, outputs 400        									|
| Fully connected		| input 400, output 120        									|
| RELU  				|         									|
| Dropout  				|         									|
| Fully connected		| input 120, output 84        									|
| RELU  				|         									|
| Dropout  				|         									|
| Fully connected		| input 84, output 42        									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

사용한 hyperparameter는 아래와 같습니다.
* Epochs : 60
* Batch size : 64
* learning rate : 0.0005
* Dropout probability : 0.5

hyperparameter 고려를 줄이기 위해 optimizer의 경우 ADAM optimizer를 사용하였습니다.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

최종적으로 모델 학습 및 평가 결과는 아래와 같이 나타났습니다.
* validation set accuracy : 0.974
* test set accuracy : 0.944

If an iterative approach was chosen:
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
 
 dropout 을 추가하여 accuracy 를 상승시켰습니다. dropout 적용시 validation accuracy 도 상승하고 test set 에 대한 accuracy 역시 상승 하였습니다. 특히 해당 과정은 뒤에서 언급할 새로운 이미지를 이용해 테스트 하는 과정에서 성능 향상에 많은 도움이 되었습니다.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

최종적으로 가장 정확도를 높힌 것은 dropout layer를 추가하는 것이 가장 효과적이었습니다.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] 



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image6] 

위 결과를 보면 대부분의 표지판은 제대로 인식하는 것을 알 수 있습니다.

다만 train data set에서 40km/h 에 대한 데이터 및 U턴에 데한 데이터 set이 없어서 

학습이 진행되지 않았기 때문에 잘못 측정된 부분은 data set이 보충된다면 해결 될 문제로 보이나

20km/h 표지판을 30km/h로 결과가 나온 것은 측정 오류로 보완해야 할 점으로 보입니다.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in  the Ipython notebook.

각각의 sofrmax의 확률을 보면 제대로 측정한 표지판의 경구 거의 100% 확률로 결과를 예측하는 것을 볼 수 있습니다.

다만 앞서 말한 20km/h 표지판을 잘못 인지한 부분에서는 20km/h와 30km/h의 label에서 각각 0.4와 0.6 의 확률로 결과가 나온 부분을 볼 수 있습니다.

![alt text][image7] 




