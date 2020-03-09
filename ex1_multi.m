%% Multi-Feature

%% Clear and Close Figures
clear ; close all; clc
fprintf('Loading data ...\n');

%% ================ Part 1: Feature Normalization ================
%loading data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);
% N = wgn(m,1,0);
% % N = [N zeros(m,1)];
% noise_var = var(N);
% X = X+N;
noise_var = 0.02; 
W = sqrt(noise_var).*randn(m,2); %Gaussian white noise W
X = X + W; %Add the noise
% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');
[X,mu,sigma] = featureNormalize(X);
% Add X0
X = [ones(m, 1) X];
%% ================ Gradient Descent ================
%At prediction, make sure you do the same feature normalization.
fprintf('Running gradient descent ...\n');
% Choose some alpha value
alpha = 0.1;
num_iters = 60;
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters,noise_var);

%% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

%% Estimate the price of a 1650 sq-ft, 3 br house
X_house = [1 1650 3];
X_house = X_house(:,2:end);
X_house = ((X_house-mu)./sigma);
X_house = [ones(size(X_house,1),1) X_house];
price = X_house*theta;
fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

%% ================ Part 2:Normal Equations ================
fprintf('Solving with normal equations...\n');
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta_nor = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta_nor);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
X_house = [1 1650 3];
price_norm = X_house * theta_nor;

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price_norm);

