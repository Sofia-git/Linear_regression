function J = computeCostMulti(X, y, theta,noise_var)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize
m = length(y); 
% You need to return the following variables correctly 
%J = 0;
[m, ~] = size(X);
hypothesis = X * theta;
J = 1/2/m * ((hypothesis - y)' * (hypothesis - y)) - (noise_var * (theta'*theta));

end
