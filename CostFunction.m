function cstval = CostFunction(x,y,theta,m)
YPred = x * theta;
cstval = (1/2*m)*sum(YPred-y)^2;
end