
Imagine, you are a data analyst and you would try to classify an email, which is spam or not spam. Then what you have to do? Maybe already you are thinking about classification models. Yes, the Logistic Regression is a classification model, which helps you to classify our mention problem. There are two types of outcome for binary classification, one is negative(0) and the other one is positive(1). In multi-class case has lots of labels. The logistic regression was developed for population growth and name logistic by Verhulst in 1830 and 1840. Let's extend how it is work step by step through a little project.

First, introduce with logistic regression,
![Logistic Equation](https://github.com/jacknayem/MachineLearning/blob/Logistic-Regression/images/LogisticRegression.PNG)

Sigmoid function looking in graaph,
![Logistic Grap](https://github.com/jacknayem/MachineLearning/blob/Logistic-Regression/images/sigmoidFunction.png)

I made a main.m file is use to load our targeted data and dispay all of the outcome. First I have plot the value to check out the condition of this.
_**main.m**_
```
data = load("Data.txt");
x = data(:,1:2);    %explanatory value
y = data(:,2);      %scalar Response
plotData(X,y);        % Plot the features for visualizing
%Add a new feature
X = [ones(m,1),X];
[m,n] = size(X);      %Define the length and initial theta value
labda = 1
iteration = 400;      %Interating values for minimizing Gradient Descent
theta = zeros(n,1);
```

I have ploted the data using _plotData_ function. Here I send independent variables as x and depended variable as y.
_**plotData.m**_
```
function value = plotData(x,y)
positive = find(y == 1);
negative = find(y ~= 1);
figure;
plot(x(positive,1),x(positive,2), 'k+');
hold on;
plot(x(negative,1),x(negative,2), 'ko');
hold off;
end
```
Output:
![Plot Data](https://github.com/jacknayem/MachineLearning/blob/Logistic-Regression/images/potData.png)

Then we need know global minimum for getting theta values. Here I use function minimization unconstrained (fminunc) function. Using this function I have called the CostFunction() function to get error rate and theta value. Our maximum iterations is 400.
Part code of _**main.m**_
```
option = optimset('GradObj','on','MaxIter',400);
[theta, cost] = fminunc(@(t)(CostFunction(X,y,t,m)),theta,option);
fprintf('The cost function by fminunc is: %f\n',J)
fprintf('Theta is:\n')
fprintf('%f\n',theta)
```
Output:
```
The cost function by fminunc is: 4.794753
Theta is:
-25.161356
0.206232
0.201472
```
Here is the cost function to calculate error rate. X and y go as a parameter. And m is the length of our data.
![](https://github.com/jacknayem/MachineLearning/blob/Logistic-Regression/images/CostFunction.PNG)
**_CostFunction.m_**
```
function [J,grad] = CostFunction(X,y,theta,m)
J = 0;
z = X * theta;
hpyX = Sigmoid(z);
J = (1/m) .* ((-y' * log(hpyX)) - ((1-y)' * log(1-hpyX)));
grad = (1/m) * (X' * (hpyX - y));
end
```
_**Sigmoid.m**_
```
function sigmoid = Sigmoid(z)
sigmoid = zeros(size(z));
sigmoid = 1 ./ (1 + exp(-z));
end
```
Finally we classify the data using predict() function
part of _**main.m**_
```
pred = predict(X,theta);
Ypred = (pred >= 0.5);
fprintf('The training Accuracy: %f\n', mean(double(Ypred == y)) * 100);
```
Output:
```
The training Accuracy: 89.000000
```
_**predict.m**_
```
function pred = predict(theta, X)
z = theta * X;
pred = Sigmoid(z);
end
```
I think 89% accuracy result is not bad.
