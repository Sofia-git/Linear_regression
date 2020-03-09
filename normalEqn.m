function [theta] = normalEqn(X, y)

%   regression using the normal equations.
%theta = zeros(size(X, 2), 1);
theta = pinv(X'*X)*X'* y;

end
