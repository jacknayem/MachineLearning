%This is a hypothesis of AND operation
x1 = [0 0 1 1]
x2 = [0 1 0 1];

features = [x1;x2]';
AND = [-30 20 20];
NOT = [10 -20 -20];
weights = [AND; NOT];
OR = [-10 20 20];
x = [ones(length(features),1) features];

z = x * weights'
[m,n] = size(z);
pred = zeros(m,n);
pred = 1 ./ (1 + exp(-z));
a2 = (pred >= 0.5);
a2 = [ones(length(a2),1) a2]
z = a2 * OR';
[m,n] = size(z);
pred = zeros(m,n);
pred = 1 ./ (1 + exp(-z));
output = (pred >= 0.5)