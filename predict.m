function pred = predict(theta, X)
z = theta * X;
pred = Sigmoid(z);
end