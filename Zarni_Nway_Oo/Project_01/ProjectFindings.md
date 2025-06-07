# Project Findings

## Before Experiments

First I read the outlines written in the project about the data and its compsitions. I tried looking into how many inputs does this model take and how many features does it take for training and surprising I found out that there are 51 features which are different from what is outlined in the project which is 11 features. So, I tried to analzye the data and found out that the flat_model and town has been one-hot encoded originally and other columns are also used scaler for scaling the values between 0 and 1.

Before watching the video explanation, I tried to understand the code first and why there are only 2 hidden layers with ReLu activation. After understanding that ReLu is better for using in hidden layers because of its properties to be able to mitigate vanishing gradient problem. I tried to search up about why we use 'Adam' as optimizer and 'MSE' as loss function. Even though I do not fully understand, like ReLu, Adam is commonly used for deep learning tasks is what I currently understand and the same goes for MSE which can help reduce the large errors problems.

## During Experiments

After looking at the project and reading the instructions on the LMS and watching the videos, I thought of experimenting with various model architectures(adding more layers) and to experiment with various epoch values.

At first, I was adding layers and training on different layers manually. During the training, I tried to separate the initial model structure, training model name, and different evaluation method. However, during the process, I copied to same model architecture function thinking it was the modified version. After trying to find the total parameters through model.summary() I realized that I made a huge mistake.

So, I used Claude to help me write automated script, that will let me experiment with different model layers with 5 different epoch instances. The script works pretty well and it saved a lot of time for me. 

# Experiment Summary

If we look at this summary graph, we can realized that adding layers did improve the model's performance and if we are on the same architecture, increasing training epoch also improve the performance. However, what I noticed is that increasing epoch alone doesn't improved much for the most hidden layers in this experiments and most of the improvements by epoch happens in initial model architecture.

I have also added the learning curve (loss curve) graphs for each training instances to see if there is overfitting or underfitting happening. From the results, I conclude that each training are within the good-fit range as they still relatively on the same line for both training and validation loss over each epoch. Even though in this experiment, I was able to improve the performance from 0.82 to 0.93 the prediction scatter plot does not change drastically as the difference range is still significant for both the first experiment and last experiment. However, I see the results are closer to the diagonal line in the last interation so, it is a good sign. For more details diagram, please refere to chapter2/Project_01/results/ in that folder each model architecture and each epoch iterations are save individually in each folder.

For training time, we can roughtly assume that as epoch grows it will take more time. And as evidence in the table, more layers and more epoch will increase the training time. 

As we have discussed, I think deeper neural network improve performance because it has much more layers to work through and learn about all the complex relationship between features and outcome more than shallow neural network. However, as per this experiment, if we keep increasing the iteration or epoch allowing the model to train back and forth and adjust their weights according to the loss function, we can see that the performance can also increase even though the network is quite shallow at that stage.

From this observation I can conclude that, for shallow networks we can find out how many times we need to iterate in order to increase the performance and with deeper network we can push pass the initial walls that the shallow network is facing. This is based on this experiement only and it could be totally wrong as I have much more knowledge to acquire in order to make proper conclusions.

| Model | Architecture | Epochs | Train_R2 | Test_R2 | Train_RMSE | Test_RMSE | Training_Time |
|:--------|:---------------------------------------|---------:|-----------:|----------:|-------------:|------------:|----------------:|
| model1 | 32 -> 16 -> 1 | 10 | 0.8188 | 0.8194 | 85982.9 | 85864.9 | 15.1129 |
| model1 | 32 -> 16 -> 1 | 20 | 0.8488 | 0.8487 | 78551.4 | 78577.8 | 30.6848 |
| model1 | 32 -> 16 -> 1 | 30 | 0.8907 | 0.8912 | 66791.5 | 66640.8 | 43.8717 |
| model1 | 32 -> 16 -> 1 | 40 | 0.8915 | 0.8922 | 66530.1 | 66319.5 | 56.7951 |
| model1 | 32 -> 16 -> 1 | 50 | 0.8994 | 0.9 | 64065.1 | 63888 | 71.3689 |
| model2 | 32 -> 16 -> 8 -> 1 | 10 | 0.8684 | 0.8684 | 73280.9 | 73298.5 | 16.3057 |
| model2 | 32 -> 16 -> 8 -> 1 | 20 | 0.9032 | 0.9039 | 62848 | 62638 | 32.0963 |
| model2 | 32 -> 16 -> 8 -> 1 | 30 | 0.9026 | 0.903 | 63043.1 | 62936.2 | 47.5518 |
| model2 | 32 -> 16 -> 8 -> 1 | 40 | 0.9105 | 0.9109 | 60417.6 | 60321.4 | 64.8382 |
| model2 | 32 -> 16 -> 8 -> 1 | 50 | 0.9077 | 0.9084 | 61362 | 61152.6 | 77.6093 |
| model3 | 64 -> 32 -> 16 -> 8 -> 1 | 10 | 0.9064 | 0.907 | 61791.4 | 61605.3 | 18.5435 |
| model3 | 64 -> 32 -> 16 -> 8 -> 1 | 20 | 0.9175 | 0.9179 | 58023.6 | 57885 | 35.8878 |
| model3 | 64 -> 32 -> 16 -> 8 -> 1 | 30 | 0.9176 | 0.9175 | 57969.3 | 58021.7 | 53.8723 |
| model3 | 64 -> 32 -> 16 -> 8 -> 1 | 40 | 0.9228 | 0.9224 | 56127.4 | 56265.2 | 71.7005 |
| model3 | 64 -> 32 -> 16 -> 8 -> 1 | 50 | 0.9208 | 0.9205 | 56832 | 56959.5 | 88.893 |
| model4 | 128 -> 64 -> 32 -> 16 -> 8 -> 1 | 10 | 0.9163 | 0.9168 | 58424.8 | 58259.8 | 21.7189 |
| model4 | 128 -> 64 -> 32 -> 16 -> 8 -> 1 | 20 | 0.9237 | 0.9233 | 55807.3 | 55961.9 | 42.7896 |
| model4 | 128 -> 64 -> 32 -> 16 -> 8 -> 1 | 30 | 0.9225 | 0.9225 | 56217.8 | 56249.9 | 64.155 |
| model4 | 128 -> 64 -> 32 -> 16 -> 8 -> 1 | 40 | 0.9264 | 0.9257 | 54793.9 | 55082 | 85.6268 |
| model4 | 128 -> 64 -> 32 -> 16 -> 8 -> 1 | 50 | 0.9262 | 0.9254 | 54892.4 | 55165.4 | 107.907 |
| model5 | 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 1 | 10 | 0.9201 | 0.92 | 57098.8 | 57152.3 | 32.0035 |
| model5 | 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 1 | 20 | 0.9258 | 0.925 | 55026.6 | 55316.7 | 62.7032 |
| model5 | 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 1 | 30 | 0.9294 | 0.9279 | 53654.6 | 54231 | 99.1184 |
| model5 | 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 1 | 40 | 0.9314 | 0.9296 | 52892.2 | 53594.1 | 124.861 |
| model5 | 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 1 | 50 | 0.9336 | 0.9313 | 52049.7 | 52953.1 | 146.796 |

## Summary Statistics

- **Total experiments**: 25
- **Models tested**: 5 different architectures
- **Epochs range**: 10-50 epochs per model
- **Best Test R²**: 0.9313 (model5, 50 epochs)
- **Best Test RMSE**: 52953.1 (model5, 50 epochs)
- **Training time range**: 15.113s - 146.796s