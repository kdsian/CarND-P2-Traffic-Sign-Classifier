# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
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
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/kdsian/CarND-P2-Traffic-Sign-Classifier)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

`plot_images` 라는 함수를 만들어서 dataset에 대해 임의의 sample 들을 출력해서 확인 할 수 있도록 구성하였습니다.

동시에 label을 출력하도록하여 영상 sample과 label 의 매칭이 제대로 되어 있는지도 확인 하였습니다.

아래 예시는 train sample에 대한 예시 입니다.

![alt text][image1]

아래 예시는 test sample에 대한 예시 입니다.

![alt text][image1_1]

train sample의 분포를 살펴보면 아래와 같습니다.

![alt text][image1_2]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

첫 번째 과정으로 grayscale로 변환하는 작업을 수행했습니다.

수행 예시는 아래와 같습니다.

![alt text][image2]

두 번째 과정으로는 normalize 작업을 수행했습니다.

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

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.974
* test set accuracy of 0.944

If an iterative approach was chosen:
* What were some problems with the initial architecture?

accuracy 가 0.93 이하로 측정되어서 hyperparameter 를 수정하는 작업을 반복하여도 일정 accuracy 이상 상승 하지 않는 문제가 있었습니다.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
 
 dropout 을 추가하여 accuracy 를 상승시켰습니다. dropout 적용시 validation accuracy 도 상승하고 test set 에 대한 accuracy 역시 상승 하였습니다.

* Which parameters were tuned? How were they adjusted and why?

컴퓨팅 파워 상에서는 데이터의 크기도 작고 숫자도 많지 않았기 떄문에 Epoch 크기는 증가시키고 batch 크기는 감소, learning rate역시 감소 시켜 일정 부분 학습이 늦게 되어도 높은 정확도를 갖도록 iteration 작업을 수행 했습니다.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

최종적으로 가장 정확도를 높힌 것은 dropout layer를 추가하는 것이 가장 효과적이었습니다.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] 

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image6] 

위 결과를 보면 대부분의 표지판은 제대로 인식하는 것을 알 수 있습니다.

다만 train data set에서 40km/h 에 대한 데이터 set이 없어서 학습이 진행되지 않았기 때문에 30km/h로 결과가 측정 된 부분은

data set이 보충된다면 해결 될 문제로 보이나

20km/h 표지판을 30km/h로 결과가 나온 것은 측정 오류로 보완해야 할 점으로 보입니다.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in  the Ipython notebook.

각각의 sofrmax의 확률을 보면 제대로 측정한 표지판의 경구 거의 100% 확률로 결과를 예측하는 것을 볼 수 있습니다.

다만 앞서 말한 20km/h 표지판을 잘못 인지한 부분에서는 20km/h와 30km/h의 label에서 각각 0.4와 0.6 의 확률로 결과가 나온 부분을 볼 수 있습니다.

![alt text][image7] 




