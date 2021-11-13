clear ; close all; clc
input_layer_size  = 400;  
hidden_layer_size = 25;   
num_labels = 10;          
load('data1.mat');
m = size(X, 1);
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));
fprintf('\nLoading Saved Neural Network Parameters ...\n')
load('weights1.mat');
nn_params = [Theta1(:) ; Theta2(:)];
fprintf('\nFeedforward Using Neural Network ...\n')
lambda = 0;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf('\nChecking Cost Function (w/ Regularization) ... \n')
lambda = 1;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf('\nEvaluating sigmoid gradient...\n')

g = sigmoidGradient([-1 -0.5 0 0.5 1]);
fprintf('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ');
fprintf('%f ', g);
fprintf('\n\n');

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
fprintf('\nChecking Backpropagation... \n');
checkNNGradients;
fprintf('\nChecking Backpropagation (w/ Regularization) ... \n')
lambda = 3;
checkNNGradients(lambda);
debug_J  = nnCostFunction(nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X, y, lambda);

fprintf('\nTraining Neural Network... \n')
options = optimset('MaxIter', 50);
lambda = 1;
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


