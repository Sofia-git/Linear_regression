function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialization
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
   
    for iter = 1:num_iters
        hypothesis = X * theta;
        theta = theta - alpha * ((hypothesis - y)' * X)' / m;
        J_history(iter) = computeCost(X, y, theta); % Save the cost J in every iteration

    end
end
