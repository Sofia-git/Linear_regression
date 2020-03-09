%% Linear Regression
% x refers to the population size in 10,000s
% y refers to the profit in $10,000s
%% Initialization
clear ; close all; clc

%% ======================= Plotting the data =======================
fprintf('Data ...\n')
data = load('ex1data1.txt');
X = data(1:78, 1); y = data(1:78, 2);
m = length(y); % number of training examples
%plotData.m
plotData(X, y);
N = wgn(m,1,0);
var(N)
X = X+N;
plotData(N, y);
%% =================== Gradient descent ===================
fprintf('Running Gradient Descent ...\n')

X = [ones(m, 1), data(:,1)]; % Add a column of ones to , x0
theta = zeros(2, 1); % initialize fitting parameters (theta1 & theta2)
% gradient descent settings
iterations = 1500;
alpha = 0.01;
% compute and display initial cost
computeCost(X, y, theta)

% run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations);

% print theta to screen
fprintf('Theta found by gradient descent: ');
fprintf('%f %f \n', theta(1), theta(2));

% Plot the linear fit
hold on; 
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off 
%% ============== Trying out the learned network ==============
% Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] *theta; %[x0,x1]
fprintf('For population = 35,000, we predict a profit of %f\n',...
    predict1*10000);
predict2 = [1, 7] * theta;
fprintf('For population = 70,000, we predict a profit of %f\n',...
    predict2*10000);

%% ============= Visualizing J(theta_0, theta_1) =============
fprintf('Visualizing J(theta_0, theta_1) ...\n')

% Grid over which J will be calculate
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];    
	  J_vals(i,j) = computeCost(X, y, t);
    end
end

% Surface plot: how J varies when theta0 & theta1 changes
J_vals = J_vals';
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
