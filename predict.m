function pred = predict(Theta1, Theta2, X)
m = size(X,1);
pred = zeros(m,1);

a1 = [ones(m,1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);

a2 = [ones(size(a2,1),1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

[prob, pred] = max(a3, [], 2);
end