function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialization
m = length(y); % number of training examples
J = 0;

[m, ~] = size(X);
hypothesis = X*theta;
J = 1/2/m * ((hypothesis - y)' * (hypothesis - y)); % cost function 
%J = (-y' *log10(hypothesis)) - ((1-y')*log10(1-hypothesis));
end
