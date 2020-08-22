# Neural Network

Cells into the nervous system is call neurons. The brain is a information processing system, where every information process and transmit within neuron cell. The neuron has three main part. First One is Dendrite for receiving infomation or data form another cell, Second one is nucleus for data process and last one is axon use to transmit the process data to another neuron. The neuron network approch is not differnt from a nervous system. The vision of the neural network is to try to get an appropriation of the nervous system. The neural network are also use three step. Those are input, process(hidden Layer) and Output.

![Neuron](https://github.com/jacknayem/MachineLearning/blob/Neural-Network/images/Brain_Nuron.png)
![Neural Network](https://github.com/jacknayem/MachineLearning/blob/Neural-Network/images/Neural_Network.png)  

In our model, The input is work alike dendrite and Out hypthesis output is like axon. There are three time of layer.  
- input Layer or Initial Layer
- Hidden Layer (Data processing units)
- Output Layer.

In the neural network, there has a biasn unit, whihc value is one. It's defnine as a first unit of column. 
![Units](https://github.com/jacknayem/MachineLearning/blob/Neural-Network/images/UnitLayers.PNG)
i assign for identification of unit and J use for layer number. The diemention of theta count from
![Theta Dimension](https://github.com/jacknayem/MachineLearning/blob/Neural-Network/images/DimentionOfTheta.PNG)  
Let take an example to undestand smoothly. Suppose I have a neural network, where the input layer is four units and also three units in hidden layer and our final output is one. First we calculate,  
![Activation Node](https://github.com/jacknayem/MachineLearning/blob/Neural-Network/images/a2Calculation.PNG)  

Then  
![Hoypthesis](https://github.com/jacknayem/MachineLearning/blob/Neural-Network/images/h(x)Calculation.PNG)  
Finally, we will get our estimeted result. Let's start with a simple project to understand how it work. We have a data set and weights file (The value of theta. Sometimes it's call weights). Firstly we will load the data and plot 100 examples(rows).  

**_main.m_**
```
load('Data.mat'); % 5000 x 400 (Five Thousand example)
m = size(X,1);
random_indices = randperm(m);
sel = X(random_indices(1:100),:); %Selected 100 random sample from Data.mat
[h,display_array] = DisplayData(sel);
```
**Output:**  
![plot Display](https://github.com/jacknayem/MachineLearning/blob/Neural-Network/images/DisplaySampleData.png)  
**_DisplayData.m_**
```
function [h,display_array] =  DisplayData(X,example_width)
if ~exist('example_width', 'var') || isempty(example_width)
    example_width = round(sqrt(size(X,2))); %sqrt(400) = 20
end

colormap(gray); %All of the images will show in gray scale

[m,n] = size(X); % m = 10 and n = 400
example_height = (n/example_width); % example_height = 400/20 = 20

display_rows = floor(sqrt(m)); % display_rows = sqrt(100) = 10
display_cols = ceil(m / display_rows); % display_cols = 100/10 = 10

pad = 1; %Padding for single image

display_array = ones(pad + display_rows * (example_height + pad),...
    pad + display_cols * (example_width + pad)); % Initialize the display_array with one values, display_array( 1 + 10 * (20 + 1), 1 + 10 * (20 + 1)) = display_array(211,211)

curr_ex = 1;
for j = 1:display_rows
    for i = 1:display_cols
        if curr_ex > m
            break;
        end
        
        max_val = max(abs(X(curr_ex, :)));
        display_array(pad + (j-1) * (example_height + pad) + ( 1:example_height),...
            pad + (i-1) * (example_width + pad) + (1:example_width)) = ...
            reshape(X(curr_ex, :), example_height,example_width) /max_val; %Assign 100 images into display_array.
        curr_ex = curr_ex + 1;
    end
    if curr_ex > m
        break;
    end
end

h = imagesc(display_array,[-1,1]);
axis image off;
drawnow;
end
```
We saw an example in previous how we can figure out the error rate and the value of theta in logistic regression. We will do it again to understand the ueural network.
**_main.m_**
```
theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
y_t = ([1;0;1;0;1] >= 0.5);
lambda_t = 3;
[J, grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);
```
**Output:**
```
The Cost function is: 2.534819
Theta values
0.146561
-0.548558
0.724722
1.398003
```
**_lrCostFunction_**
```
function [J, grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t)
J = 0;
grad = zeros(length(theta_t),1);
m = length(y_t);
z = X_t * theta_t;
hyp = sigmoid(z);
J = (1/m) .* (((-y_t)' * log(hyp))-((1 - y_t)' * log(1 - hyp))) + (lambda_t/(2 * m)) .* sum(theta_t(2:end).^2); %Calculating the cost function
grad(1) = (1/m) .* (X_t(:,1)' * (hyp - y_t)); %Calculating the theta fo the bias unit
grad(2:end) = (1/m) .* (X_t(:,2:end)' * (hyp - y_t)) + (lambda_t/m) * theta_t(2:end); %Calculating other theta values
end
```
**_sigmoid_**
```
function hyp = sigmoid(z)
hyp = 1 ./ (1 + exp(-z));
end
```
