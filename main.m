load('Data.mat');
m = size(X,1);
random_indices = randperm(m);
sel = X(random_indices(1:100),:);
number_labels = 10;

%[h,display_array] = DisplayData(sel);
theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
y_t = ([1;0;1;0;1] >= 0.5);
lambda_t = 3;
%{
[J, grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);
fprintf('The Cost function is: %f\n', J)
fprintf('Theta values\n')
fprintf('%f\n',grad)
%}
lambda = .01;
all_theta = OneVsAll(X, y, number_labels, lambda);
pred = predOneVsAll(all_theta, X);
pred = mean(double(pred == y) * 100);
fprintf('Accuracy is %f',pred)