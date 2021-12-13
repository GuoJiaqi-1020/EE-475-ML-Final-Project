# EE-475-ML-Final-Project: Face Masking Recognition
A monitoring strategy based on machine learning algorithm in the COVID-19 era


<div  align="center">  
<img src=https://www.insightintodiversity.com/wp-content/uploads/2021/05/northwestern-696x392.png width = "80%">
</div>

>As the COVID-19 has brought great disaster to human beings, personal protection has becomeparticularly important. For the purpose of controlling the spread of the epidemic, every one of ushas the obligation and responsibility to wear masks. The objective of our project is to proposed asystem that can monitor people’s mask wearing status **(Correct, Incorrect and No mask)**. In ourproject, we reduce the dimension of input data space by using pre-processing methods:  convertinto gray image, Histogram of Oriented Gradients algorithm and Canny edge detector algorithm.We firstly implement linear model from scratch, then implement SVM, Decision Tree and RandomForest using scikit learn library.

<div  align="center">  
<img src=https://github.com/GuoJiaqi-1020/EE-475-ML-Final-Project/blob/main/img/Others/illustration.png width = "80%">
</div>

##### Looking for the dataset?

https://www.kaggle.com/hadjadjib/facemask-recognition-dataset

In order to run our code, you need to download the dataset and resize the image into size **100×100×3**. Also, all images of the same category should be packaged into a **".npy"** format file and placed in directory **"../Data/Pixel100"**.

(We've already given two examples under this path **"Pixel20"** and **"Pixel50"**)