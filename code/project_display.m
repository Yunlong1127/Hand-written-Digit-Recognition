clear ; close all; clc
input_layer_size  = 400;  
hidden_layer_size = 25;   
num_labels = 10;            
fprintf('Loading and Visualizing Data ...\n')

load('data1.mat');
m = size(X, 1);
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nLoading Saved Neural Network Parameters ...\n')
load('weights.mat');

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

fprintf('Program paused. Press enter to continue.\n');
pause;
rp = randperm(m);

for i = 1:m
    fprintf('\nDisplaying Example Image\n');
    displayData(X(rp(i), :));

    pred = predict(Theta1, Theta2, X(rp(i),:));
    fprintf('\nNeural Network Prediction: %d (digit %d)\n', mod(pred, 10), mod(pred, 10));
    s = input('Paused - press enter to continue, q to exit:','s');
    if s == 'q'
      break
    end
end

