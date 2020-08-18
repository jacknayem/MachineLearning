logistic regression

Imagine, you are a data analyst and you would try to classofy an email, which is spam or not spam. what you have to do? I think, already you know you have to move on classification models. The ligistic Regression is a calssification model. it has specific labels. There are two type of data(1,2) for binary classification. In multi-class case has lots of label.The logictic regression was develop for population grouth and name logistic by Verhulst in 1830 and 1840. Let extend how it is work step by step.

First, introduce with logistic regression,
![Logistic Equation](https://github.com/jacknayem/MachineLearning/blob/Logistic-Regression/images/LogisticRegression.PNG)

Sigmoid function looking in graaph,
![Logistic Grap](https://github.com/jacknayem/MachineLearning/blob/Logistic-Regression/images/sigmoidFunction.png)

I made a main.m file to load our targeted data. and I have plot the value to check out the condition of this.
main.m
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

I have plot the data using plotData function
plotData.m
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

Then we need golobal minimul for getting theta value. Here I use function minimization unconstrained (fminunc) function. Using this function I have called the CostFunction() function to get error rate and theta value
Part code of main.m
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
Here is the cost function to calculate error rate.
![](https://github.com/jacknayem/MachineLearning/blob/Logistic-Regression/images/CostFunction.PNG)
CostFunction.m
```
function [J,grad] = CostFunction(X,y,theta,m)
J = 0;
z = X * theta;
hpyX = Sigmoid(z);
J = (1/m) .* ((-y' * log(hpyX)) - ((1-y)' * log(1-hpyX)));
grad = (1/m) * (X' * (hpyX - y));
end
```
Sigmoid.m
```
function sigmoid = Sigmoid(z)
sigmoid = zeros(size(z));
sigmoid = 1 ./ (1 + exp(-z));
end
```
Finally we classify the data using predict() function
part of main.m
```
pred = predict(X,theta);
Ypred = (pred >= 0.5);
fprintf('Train Accuracy: %f\n', mean(double(Ypred == y)) * 100);
```
Output:
```
Train Accuracy: 89.000000
```
predict.m
```
function pred = predict(theta, X)
z = theta * X;
pred = Sigmoid(z);
end
```
I think the prediction result is not bad.
