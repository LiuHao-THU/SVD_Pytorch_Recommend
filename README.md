# Pytorch Version SVD Algorithms

## Reference:
## https://medium.com/@rinabuoy13/explicit-recommender-system-matrix-factorization-in-pytorch-f3779bb55d74


## Train details

### DataSets: Movie Lens 1M 

Load Datasets Started...
Load Datasets Ended...
1000209it [00:01, 575861.92it/s]
len item = 900188, len user = 900188, len rate = 900188
len item = 100021, len user = 100021, len rate = 100021
len train loader = 880 len test loader = 391
MSE:0.9499 ; RMSE:0.9746; MAE:0.7809
MSE:0.8173 ; RMSE:0.9040; MAE:0.7136
MSE:0.7812 ; RMSE:0.8838; MAE:0.6945
MSE:0.7608 ; RMSE:0.8723; MAE:0.6838
MSE:0.7461 ; RMSE:0.8638; MAE:0.6765
MSE:0.7349 ; RMSE:0.8573; MAE:0.6708
MSE:0.7266 ; RMSE:0.8524; MAE:0.6666
MSE:0.7210 ; RMSE:0.8491; MAE:0.6638
MSE:0.7183 ; RMSE:0.8475; MAE:0.6626
MSE:0.7185 ; RMSE:0.8477; MAE:0.6627
MSE:0.7211 ; RMSE:0.8492; MAE:0.6639
MSE:0.7263 ; RMSE:0.8522; MAE:0.6664
MSE:0.7336 ; RMSE:0.8565; MAE:0.6698
MSE:0.7426 ; RMSE:0.8618; MAE:0.6741
Early stopping!
MSE:0.7183 ; RMSE:0.8475; MAE:0.6626(Results for Validation) 

### Train Time and Machine Matric
Less Than 3.5s Time to Train 1 Epoch

AMD 3600 + 1080 Ti

Less Than 1s to Predict The Top10 Results For All User(6040 Users)

The Results is The Same as discribled in Paper:
https://arxiv.org/abs/1802.04606

## Some Problem In Re-Implementation The Paper

### 1. How to Init The weights in Embedding Layer
		Failed Method: norm, uniform, kaiming_init
		Succuss Method:
### 2. Overfit Problem
		Failed Method:
		1. dropout in embedding product
		2. dropout in embedding layer
		3. too large or small weight decay
		Succuss Method:
		approporiate weight decay: 1e-5(try from 1e-3 to 1e-6)
		early stop
### 3. How to Fast Fill The User-Item Matrix
		Generate a dataLoader for all unseen(Negative) \
		User-Item Samples will Cost Much Longer Time Than \
		Matrix Operation Using Gpus.


## To do
### 1. Collaborative Metric Learning (How to Fast Predict All The Negative Samples For Very Large User-Item Matrix: eg 60w+ Users 3000+ Items)

		When we use SVD to predict all the negative sample to fill User-Item Matrix,
		Gpu can be Used to accelerate Calculation by Matrix Multiply. (Can be Finished In 1 minute)

		If we Generate A DataLoader For all Negative Sample:
		eg:
			num_samples = num_users * num_items
			u = num_sample // num_users
			v = num_sample %  num_items

		This Need More Than 10h to Fill The User-Item Matrix

### 2. Avoid Overfit, Replace The early stop

		Now We Random split Part of the Training Data for Evaluation to Avoid Overfit. However, 
		The Part of Evaluation Data Can never Used In Train

