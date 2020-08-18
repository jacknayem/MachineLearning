%Load and split the features and label 
data = load('Data.txt');
X = data(:,1:2);
y = data(:,3);

%Add a new feature
X = [ones(m,1),X];

%Define the length and initial theta value
[m,n] = size(X);
alpha = 0.01; %Learning rate
labda = 1
iteration = 400; %Interating values for minimizing Gradient Descent
theta = zeros(n,1);

%plotData(x,y); % Plot the features for visualizing

[J,grad] = CostFunction(X,y,theta,m);
fprintf('The cost for inital value zero is: %f\n',J);
fprintf('The Gradient Descent for initala value zero is:\n');
fprintf('%f\n',grad);

[theta,hist] = GradientDescent(X,y,m, alpha, theta, iteration);
J = CostFunction(X,y,theta,m);
fprintf('The cost for inital value zero is: %f\n',J);
fprintf('Minimum Gradient Descent is:\n')
fprintf('%f\n',theta)


%Set for option function minimaization unconstrained
option = optimset('GradObj','on','MaxIter',400);
[theta, cost] = fminunc(@(t)(CostFunction(X,y,t,m)),theta,option);
fprintf('The cost function by fminunc is: %f\n',J)
fprintf('Theta is:\n')
fprintf('%f\n',theta)

%[J,grad] = Regularization(X,y,theta,labda,m)

pred = predict(X,theta);

Ypred = (pred >= 0.5);

fprintf('Train Accuracy: %f\n', mean(double(Ypred == y)) * 100);




