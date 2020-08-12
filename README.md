# Linear Regression
Linera Regression is a supervised learning quantitative regression. It's create a realtion between explanatory variable to so a specific problem. If there is only one explanatory variable, then it's call simple linear regression. But if there have more than one explanatory variables. At this time, it's call Multi Linear Regression. We need to calculate Gradient Descent, cost function and so on. Out Linear regrassion equestion is,  
![Linear Equation](https://github.com/jacknayem/MachineLearning/blob/Linear-Regression/images/Equations.png)  


**File:** main.m
In my repository, there are two type of Datasets: Data.txt and Salary_data.csv. I whoud practice on Data.txt. First I loaded the data Data.txt and Devide into input value X and output value y. Then I ploted all of the value.
```
data = load("Data.txt");
x = data(:,1);
y = data(:,2);
plot(x,y,'xr')
```
**Output:**  
![Data Plotiing](https://github.com/jacknayem/MachineLearning/blob/Linear-Regression/images/MainValuePlotting.png)


Second, I define some required varibale to calculate **Gradient Descent**. Cause I need to knwo the minimum value of **Theta**. This why I called the gradientDescent().
```
m = length(y);
alpha = 0.01;
theta = zeros(2,1);
x = [ones(length(y),1),data(:,1)];
iterations = 1500;
%[theta,cost_history] = gradientDescent(x, y, theta, alpha, iterations);
```
**Output:** 
```
theta =
   -3.8958
    1.1930
```
We can do the dame thisng by **Normal Rquation**
```
theta = NormalEqu(x,y);
```
**Output:** 
```
theta =
   -3.8958
    1.1930
```
Finally, Predict the value and draw a line graph of predicted value display with ploted main value. 
```
predvalue = x * theta;
plot(x,y,'xr')
hold on
plot(x,predvalue)
```
**Output:**  
![Plotting value](https://github.com/jacknayem/MachineLearning/blob/Linear-Regression/images/MainVsPredicted%20value.png)

Normalize all of the value.
```
%[x_norm, mu, sigma] = featuresNormalization(x)
```
**Function:** _gradientDescent.m_
 ```
function [theta,cost_history] = gradientDescent(x, y, theta, alpha, num_iters)
m = length(y)
cost_history = zeros(num_iters,1)
for iter = 1:num_iters
    error = (x * theta)-y;
    theta = theta - ((alpha/m) .* (x' * error));
    cost_history(iter) = CostFunction(x,y,theta,m)
end
end
 ```
  **Function:** featuresNormalization.m_
```
function [x_norm, mu, sigma] = featuresNormalization(x)
x_norm = x;
mu = zeros(1, size(x, 2));
sigma = zeros(1, size(x, 2));
mu = mean(x)
sigma = std(x)
x_norm = (x - mu)/sigma
end
```
 **Function:** CostFunction.m_
 ```
function cstval = CostFunction(x,y,theta,m)
YPred = x * theta;
cstval = (1/2*m)*sum(YPred-y)^2;
end
 ```
 **Function:** NormalEqu.m_
 ```
function theta = NormalEqu(x,y)
theta = zeros(size(x,2),1);
theta = pinv(x'*x)*x'*y;
end
 ```
