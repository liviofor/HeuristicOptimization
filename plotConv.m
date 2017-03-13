function [  ] = plotConv( functionOptions, funChoice, save )
%plotConv Plots Convergence of a Genetic Algorithm
% ************************************************************************
%   To be used with MATLAB script, HW2, application of a 
%   simple binary genetic algorithm (SBGA)
% ************************************************************************
%   Inputs:
%       functionOptions
%           String file with names of functions to call
%       funChoice
%           Number corresponding to the desired function
%       save        - Do you want to save the plot?
%                       1 == Yes, 2 == No
%                       Row vector of size [ 1 x m ]
%

if nargin < 3
    save = zeros(1,length(functionOptions));
end

% Load Function Evalulation Data;
% Each row is the best function evaluation for the generation
data  = importdata(strcat(functionOptions{funChoice},'_Iteration.mat'));

% Randomly select a GA Run
r = randi(size(data,2));

% Plot Function Evaluation vs. Iteration
% Find non-zero indices
nonZeroIndex = find(data(:,r));
% Max non-zero indice
maxNonZero = max(nonZeroIndex);
% Create iteration Matrix
iterMatrix = linspace(1,maxNonZero,maxNonZero);

figure(funChoice);
hold on
grid on
h = plot(iterMatrix,data(nonZeroIndex,r),'k.');
xlabel('Iterations');
ylabel('Function Evaluation Value');
titleText = 'Best Function Value at Iteration vs. Iteration for ';
title(strcat(titleText,{' '},functionOptions{funChoice}));

if save(1,funChoice) == 1
    % Save image as name of function
    saveas(h,strcat(functionOptions{funChoice},'.png'))
end

end
