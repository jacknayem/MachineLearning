# Linear Regression
The Linera Regression is a supervised learning quantitative regression. It's create a realtion between explanatory variable to solve a specific problem. If there is only one explanatory variable, it's call simple linear regression. But if there have more than one explanatory variables. At this time, it's call Multi Linear Regression. The Linear Regression uses for predictive analysis. Linear regrassion equestion are,  
![Linear Equation](https://github.com/jacknayem/MachineLearning/blob/Linear-Regression/images/Equations.png)  
Here is x variable is the features of Datasets or Independent variabale. **Theta** Subscript 0 is a y intercept, it also call constant variable. **Theta** Subscript 1 is called solpe and the final result y is our estemeted outcome. Some time our Hypothesis function need nor to be linear, if the that's does not fill well. We can use plynomial equation in this situation. The equation can be  quadratic, cubic or square root function or any other formula. However, there have only features x Dataset. But we need the value of theta to estimate the result.  In this case, how can I get this value. The Gradient Descent or Normal Equation is ones of the best solution for it.
### Gradient Descent Equation:  
![](https://github.com/jacknayem/MachineLearning/blob/Linear-Regression/images/Gradient%20Descent.png)
### Normal Equation:
![](https://github.com/jacknayem/MachineLearning/blob/Linear-Regression/images/NormalEquation.PNG)

Both equation will give you same result. But there have a issue in particular case. The Gradient Decent work well, when the number of feature is large. on the other hand, the Normal Equation might be slower, when n (Number of features) is very large. We don't need to find out the value of **alpha** in Normal equation.

When we get the value of **theta**, the linear equation is rady to calculate the outcome. The Gradient Descent uses the Cost Function to calculate the **theta**. The Cost Function estimate the accuracy of outcome variable.
![](https://github.com/jacknayem/MachineLearning/blob/Linear-Regression/images/Cost%20funtion.PNG)
we can get the actual value of theta by minimizing the value of the Cost Function.

Let's play with the MATLAB to understand the Linear Regression. 
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
