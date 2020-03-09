function plotData(x, y)
%   PLOTDATA(x,y) plots the data points: population vs profit
figure; 
data = load('ex1data1.txt');       
X = data(1:78, 1); y = data(1:78, 2);
m = length(y);                      % number of training examples
plot(x, y, 'rx', 'MarkerSize', 5);  % Plot the data
ylabel('Profit in $10,000s');       % Set the y-axis label
xlabel('Population of City in 10,000s'); % Set the x-axis label

end
