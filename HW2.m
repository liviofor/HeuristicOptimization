% Livio Forte
% MAE 552 Heuristic Optimization
% Homework 2
% Question 2
clear all; close all; clc;
% Add Test Problems to Directory
addpath(genpath('SingleObjUnconstrTestProblems'));
functionOptions = {'ackley';'griewank';'michal';'rastr';'rosen';...
    'schwef';'dejong';'lisun'};

% ************************************************************************
% Solve Genetic Algorithm for 8 Single Obj Unconstrained Test Functions
% ************************************************************************
% Genetic Algorithm Inputs:
% funcInputs is a matrix that stores variables to be used in the SBGA
%   Function. So that different inputs can be tested for each function.
%   Row of funcInputs correspond to the function in functionOptions
%   Inputs are the same as the SBGA function
%   [ N, Res, Pc, Pm, Dim, Niter, nGA ]
funcInputs = ...
[ 100, 1E-4, 0.8, 0.05, 10, 10000, 25;    % 1 - Ackley's Function
  100, 1E-4, 0.8, 0.05, 2, 10000, 25;    % 2 - Griewangk's Function
  100, 1E-4, 0.8, 0.05, 2, 10000, 25;    % 3 - Michalewicz's Function
  100, 1E-4, 0.8, 0.05, 2, 10000, 25;    % 4 - Rastrigin Function
  100, 1E-4, 0.8, 0.05, 2, 10000, 25;    % 5 - Rosenbrock Function
  100, 1E-4, 0.8, 0.05, 2, 10000, 25;    % 6 - Schwefel Sine Function
  100, 1E-4, 0.8, 0.05, 2, 10000, 25;   % 7 - De Jong's (Sphere) Function
  100, 1E-4, 0.8, 0.05, 25, 10000, 25];   % 8 - Li and Sun's Function

% Prompt User for start and end function
fprintf('**************************************************************\n')
fprintf('List of Functions:\n')
fprintf('1 - Ackley''s Function\n2 - Griewangk''s Function\n')
fprintf('3 - Michalewicz''s Function\n4 - Rastrigin Function\n')
fprintf('5 - Rosenbrock Function\n6 - Schwefel Sine Function\n')
fprintf('7 - De Jong''s (Sphere) Function\n8 - Li and Sun''s Function\n')
fprintf('**************************************************************\n')
fprintf('Enter the number corresponding to the function\n')
    % Record Values for each iteration of GA Code for each function
    startFunc   = input('What Function do you want to Start at?\n');
    endFunc     = input('What Function do you want to End at?\n');

% Save Data for all GA iterations
parfor fun = startFunc:endFunc;
    % Load Variables Specified for each Function
        N     = funcInputs(fun,1); % Population Size
        Res   = funcInputs(fun,2); % Resolution of binary string
        Pc    = funcInputs(fun,3); % Probability of Crossover
        Pm    = funcInputs(fun,4); % Probability of mutation rate
        Dim   = funcInputs(fun,5); % Number of Dimensions
        Niter = funcInputs(fun,6); % Maximum allowed number of iterations
        nGA   = funcInputs(fun,7); % Number of iterations for GA Code
        
    % Run GA nGA times
    for jj = 1:nGA
        SBGA( funcInputs, fun, jj );
        fprintf('Finished run %u of %s \n',jj,functionOptions{fun})
    end
    
     % Save optimum values to mat file
     %   First column contains data about run
      optValSave( functionOptions, funcInputs, fun );
end

%% Question 2 Part B
% Generate funcInputs for every single design variable change
%   [ N, Res, Pc, Pm, Dim, Niter, nGA ]
clear all; clc; close all;
% RUN Genetic algorithm = 1
% RUN Surface Plot / min values      = 2
whatTask = 2;


functionOptions = {'ackley';'griewank';'michal';'rastr';'rosen';...
    'schwef';'dejong';'lisun'};

Res = 1E-4;
Dim = 2;
Niter = 10000;
nGA = 25;
iter = 1;

for Ni = 5*Dim:5*Dim:50*Dim 
    for Pci = 0.8:0.05:1.0
        for Pmi = 0.0:0.025:0.1
            funcInputs(iter,:) = [Ni,Res,Pci,Pmi,Dim,Niter,nGA];
            iter = iter + 1;
        end
    end
end
% Record Values for each iteration of GA Code for each function
    startFunc   = 1;        % Number of function to start at
    endFunc     = size(funcInputs,1);        % Number of function to end at
    fun = 7;
switch whatTask
    case 1
% Save Data for all GA iterations
parfor run = startFunc:endFunc;
    % Load Variables Specified for each Function
        N     = funcInputs(run,1); % Population Size
        Res   = funcInputs(run,2); % Resolution of binary string
        Pc    = funcInputs(run,3); % Probability of Crossover
        Pm    = funcInputs(run,4); % Probability of mutation rate
        Dim   = funcInputs(run,5); % Number of Dimensions
        Niter = funcInputs(run,6); % Maximum allowed number of iterations
        nGA   = funcInputs(run,7); % Number of iterations for GA Code
        
    % Run GA nGA times
    for jj = 1:nGA
        SBGA( funcInputs, fun, jj, run );
        fprintf('Finished iteration %u for run %u of %s \n',jj,run,functionOptions{fun})
    end
    
     % Save optimum values to mat file
     %   First column contains data about run
      optValSave( functionOptions, funcInputs, fun, run );
end

    case 2
% Surface Plot of De Jongs Function
%       Plot 4 Surfaces for varying mutation probability
for iterpm = 1:5
    for iter = startFunc:endFunc/5
        trueIndexValue = iterpm+5*(iter-1);
        runNum = num2str(trueIndexValue); % Run Number for one Pm
        rawData = importdata(strcat(runNum,'dejong','.mat'));
        
        % Remove useless data from rawData
        cols = size(rawData,2);
        cpi = (cols-1)/nGA;
        for nGAi = 1:nGA
            dataTemp(:,nGAi) = rawData(:,3+(nGAi-1)*(cpi-1)); % Not sure if correct
            dataDVTemp1(:,1+2*(nGAi-1):2+2*(nGAi-1))   = rawData(:,4+(nGAi-1)*(cpi-1):5+(nGAi-1)*(cpi-1));
        end
        [x,y] = find(dataTemp);
        for ixy = 1:length(x)
            data(x(ixy),y(ixy)) = dataTemp(x(ixy),y(ixy));
        end

        % Mean Value for entire run
        rows = (cols-1)/cpi;
        for iii = 1:rows
            x = find(data(:,iii));
            avgData(1,iii) = mean(data(x,iii));
        end
        % avgEvalData = mean(avgData(1,:));
        avgEvalData(iterpm,iter) = mean(avgData(1,:));
        
        % Find Average Design Variable for nGA iterations
        for dv = 1:size(dataDVTemp1,2)
            nzCol = find(dataDVTemp1(:,dv));
           dataDVTemp2(1,dv) = mean(dataDVTemp1(nzCol,dv));
        end
        dataDV(iterpm,iter) = min(abs(dataDVTemp2));
        
        X(iterpm,iter) = funcInputs(trueIndexValue,1); % Population Size
        Y(iterpm,iter) = funcInputs(trueIndexValue,3); % Prob of Crossover
        Z(iterpm,iter) = funcInputs(trueIndexValue,4); % Prob of Mutation
    end
%     [minValCol,minColIndex] = min(avgEvalData);
%     [minValCol,minRowIndex] = min(minValCol); 
end

% Find Min Value
[minValCol,minColIndex] = min(avgEvalData); % Finds min for each Coloumn
[minValRow,minRowIndex] = min(minValCol); % Find minimum column


%     fprintf('The optimum solution occured with the following parameters\n')
%     fprintf('Population size: %u \n',X(minRowIndex,minColIndex))
%     fprintf('Prob of Crossover size: %u \n',Y(minRowIndex,minColIndex))
%     fprintf('Prob of Mutation: %u \n',Z(minRowIndex,minColIndex))

end

%% Problem 3
clear all; clc;

functionOptions = {'ackley';'griewank';'michal';'rastr';'rosen';...
    'schwef';'dejong';'lisun'};

funcInputs = ...
[ 100, 1E-4, 0.8, 0.05, 2, 10000, 25;    % 1 - Ackley's Function
  100, 1E-4, 0.8, 0.05, 2, 10000, 25;    % 2 - Griewangk's Function
  100, 1E-4, 0.8, 0.05, 2, 10000, 25;    % 3 - Michalewicz's Function
  100, 1E-4, 0.8, 0.05, 2, 10000, 25;    % 4 - Rastrigin Function
  100, 1E-4, 0.8, 0.05, 2, 10000, 25;    % 5 - Rosenbrock Function
  100, 1E-4, 0.8, 0.05, 2, 10000, 25;    % 6 - Schwefel Sine Function
  100, 1E-4, 0.8, 0.05, 2, 10000, 25;   % 7 - De Jong's (Sphere) Function
  100, 1E-4, 0.8, 0.05, 2, 10000, 25];   % 8 - Li and Sun's Function

% Functions to Loop
startFunc = 8;
endFunc   = 8;
nGA = 25;


for iter = startFunc:endFunc
    for nGAiter = 1:nGA
        Dim   = funcInputs(iter,5); % Number of Dimensions
        Niter = funcInputs(iter,6); % Number of Iterations
        options = gaoptimset('Generations',Niter);
        [x(nGAiter,:),fval(nGAiter,:)] = ga(str2func(functionOptions{iter}),Dim);
    end
    data = [x,fval];
    save(functionOptions{iter},'data')    
end




%%
clear all; clc;close all;
startFunc = 1;
endFunc   = 1;

functionOptions = {'ackley';'griewank';'michal';'rastr';'rosen';...
    'schwef';'dejong';'lisun'};

% Find mean and standard deviation across all runs for the Objective Func
for fun = startFunc:endFunc
    optVal  = importdata(functionOptions{fun});
    cols    = size(optVal,2);            % Total columns in optVal
    numPar  = length(find(optVal(:,1))); % Number of Saved Parameters
    dim     = optVal(numPar-2,1);        % Number of Dimensions
    nGA     = optVal(numPar,1);          % Number of iterations for GA Code
    cpi     = (cols-1)/nGA;              % Columns per Iteration
    for i = 1:nGA
        % ***** Changed ****** data(:,i) = optVal(:,3+i*(cpi-1));
        data(:,i) = optVal(:,3+(i-1)*(cpi-1));  % Objective Function Value
        % mstd - Mean and Variance for each iteration
        %   Rows: [ Min; Mean; Var ]
        %   Columns: Iterations
        % **** Changed ***** [ minVal, indx ] = min(abs(data(:,i)));
        dataTemp = data(find(data(:,i)),i);
        [ minVal, indx ] = min(abs(dataTemp));
        avgIter(1,i) = minVal;             % Min of OFV per iteration
        avgIter(2,i) = mean(dataTemp);    % Mean of OFV per iteration
        avgIter(3,i) = var(dataTemp);     % Variance of OFV per iteration
        % Find Best Design Variable for each iteration
        % minDV = [ DV1, DV2, ... DVn]
        %   Rows are iterations
        dvStart = 4;
        dvEnd   = dvStart+dim;
        minDV(i,:)   = optVal(indx,dvStart+(i-1)*(cpi-1):dvEnd+(i-1)*(cpi-1));
    end
    % Average over entire nGA
     [ minVal, indx ] = min(avgIter(1,:)); % Best Func Eval
    bestFunc = [ minVal; mean(avgIter(2,:)); ...
        sqrt(mean(avgIter(3,:))) ];
    bestDV   = minDV(indx,:);               % Best Design Variable 
    % bestFunc: [ Best Value; Mean Value; Standard Deviation ]
    
    % Save Averages for each Function
    save(strcat(functionOptions{fun},'_BEST_FuncEval'),'bestFunc');
    save(strcat(functionOptions{fun},'_BEST_DesignVar'),'bestDV');
    save(strcat(functionOptions{fun},'_BEST_Iteration'),'avgIter');
    save(strcat(functionOptions{fun},'_Iteration'),'data');
end

% Plot Convergence History for each Function

saveYN = ones(1,8);
% saveYN = zeros(1,8);
for fun = startFunc:endFunc
    plotConv(functionOptions,fun,saveYN)
end







    
    
    
    


% % Change Director to matlab file Folder
% filePath1 = 'D:\Google Drive\My Computer\Livio\Documents\School\';
% filePath2 = 'First Years Masters\Spring 2017\';
% filePath3 = 'MAE 552 - Heuristic Optimization\Assignments\HW2';
% filePath = strcat(filePath1,filePath2,filePath3);
% cd(filePath);