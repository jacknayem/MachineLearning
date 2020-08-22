load('Data.mat');
load('weights.mat');

pred = predict(Theta1, Theta2, X);
fprintf('Training Set Accuracy: %f\n', mean(double(pred == y)) * 100);