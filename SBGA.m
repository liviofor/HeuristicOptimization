function [  ] = SBGA( funcInputs, funcChoice, nGAiter, run  )
%SBGA - Simple Binary Genetic Algorith (GA)
%   Using Stochastic Uniform Sampling Selection (SUSS)
%   Two point Cross-Over and Mutation
%   
%   Inputs:
%   N       - Population Size
%   Res     - Resolution of binary string (Resolution of atleast 1E-3)
%   Pc      - Probability of Crossover; Use values in range [0.7,1.0]
%   Pm      - Probability of mutation rate; Use values in range [0.0,0.1]
%   Niter   - Maximum allowed number of iterations
%   funcChoice Options:
%       1 = ackley
%       2 = griewank
%       3 = Michalewicz's function
%       4 = rastr
%       5 = rosen
%       6 = schwef
%   Dim     - Dimension size to solve
%
%   Output:
%   optVal  - Optimum Value for each iteration
%               - Rows represent iteration
%           [ Time, minFV , CDV  ]
%               Time   - Time to complete 1 Iteration in Seconds
%               minFV  - Minimum Fitness Value for Iteration
%               CDV    - Corresponding Design Variable for FV
%                      - Will contain as many columns as Design Variables 
%

% Initialize Inputs
fun = funcChoice;
        N     = funcInputs(fun,1); % Population Size
        Res   = funcInputs(fun,2); % Resolution of binary string
        Pc    = funcInputs(fun,3); % Probability of Crossover
        Pm    = funcInputs(fun,4); % Probability of mutation rate
        Dim   = funcInputs(fun,5); % Number of Dimensions
        Niter = funcInputs(fun,6); % Maximum allowed number of iterations
        nGA   = funcInputs(fun,7); % Number of iterations for GA Code

if nargin < 4
    run = '';
end
delRunIter = 0;

functionOptions = {'ackley';'griewank';'michal';'rastr';'rosen';...
    'schwef';'dejong';'lisun'};

% Set Upper and Lower Bounds for each Function
switch funcChoice
    case 1 % Ackley's Function
            XU = 30;
            XL = -30;
    case 2 % Griewangk's Function
            XU = 600;
            XL = -600;
    case 3 % Michalewicz's Function
            XU = pi;
            XL = 0;
    case 4 % Rastrigin Function
            XU = 5.12;
            XL = -5.12;
    case 5 % Rosenbrock Function
            XU = 2.048;
            XL = -2.048;
    case 6 % Schwefel Sine Function
            XU = 500;
            XL = -500;
    case 7 % De Jong's (Sphere) Function
            XU = 5.12;
            XL = -5.12;
    case 8 % Li and Sun's Function
            XU = 5.0;
            XL = -5.0;    
end

% Set Length of Binary String, L, Based on Input Resolution, Res
L = ceil(log((Res-XL+XU)/Res)/log(2)); 

% Initialize:
iter   = 0;                     % Number of generations
P = lhsdesign(N,Dim);           % Initial Population, 2D
% Scale Population
P = XL + P.*(XU-XL);

% Select Function for Fitness Evaluation
func = str2func(functionOptions{funcChoice});

% Encode:
Z = Encode( P, L, XL, XU );

% Terminate Initialize
delRun = 1;
minValPrev = 0;

while iter < Niter && delRunIter < 25
    % Initialize Timer for Each Iteration
    tic
    
    % Fitness Evaluation
    Zfit = Decode( Z, L, XL, XU );
    for i = 1:N
        fz(i,1) = func(Zfit(i,:)); % Fitness Value
    end
    % Fitness Evaluation Reference Value for finding Minimum
    if iter == 0
        fzRef = max(fz);
    end
    % Replace NaN value in Fitness Evaluation with 10*max(fz)
        %fz(find(isnan(fz)),1) = max(fz); % or min(fz)*0.001;
    
    % Selection
    % Should Probably rank values from high to low
    R(1,1) = rand * (1/N); % Random Number between equi space
    %p = fz/sum(fz); % Normalized Maximum Fitness Value
    p = (fzRef - fz)./fzRef; % Norm Minimum Fitness Value   
    ps = p./sum(p); % Sum of Normalized Fitness Values (Easier for Selection)
    for i = 2:N
        ps(i,1) = ps(i,1) + ps(i-1,1);
        R(i,1)  = R(1,1)  + (i-1)/N; % Propogate the random number equally
        R(i,1)  = R(i,1) - floor(R(i,1));
    end
    % Run SUSS Code
    PT = zeros(size(ps));
    j = 1;
    i = 1;
    while j < N+1
        if ps(i,1) >= R(j,1)
            PT(i,1) = PT(i,1) + 1; % Number of Copies to be made
            j = j + 1;
            i = 0;
        end
        if i == N %size(ps,1)
            i = 1;
        else
            i = i + 1;
        end
    end
    % Put Mates Next to each other in Zs
    % Also doesn't put the same parent next to each other
    pop = 1;
    while pop < N+1;
       for i = 1:N
          if PT(i,1) > 0
            Zs(pop,:) = Z(i,:);
            PT(i,1) = PT(i,1) - 1;
            pop = pop + 1;
          end
       end
    end
    Z = Zs;
%    clear Zs;
    
    % Two-Point Crossover
    R = rand(N/2,1); % Check if going to crossover
    pop = 1;
    for i = 1:N/2
        if R(i,1) < Pc % Procede to Crossover
            cpoints = sort(randperm(size(Z,2)-1,2)); % Select locations
            cp1 = cpoints(1,1); cp2 = cpoints(1,2);
            if cp1 == 1; % Two sections
                Zc(pop,:) = strcat(Z(pop+1,cp1:cp2),Z(pop,cp2+1:end));
                Zc(pop+1,:) = strcat(Z(pop,cp1:cp2),Z(pop+1,cp2+1:end));
            else % Three sections
                Zc(pop,:) = strcat(Z(pop,1:cp1-1),Z(pop+1,cp1:cp2),...
                    Z(pop,cp2+1:end));
                Zc(pop+1,:) = strcat(Z(pop+1,1:cp1-1),Z(pop,cp1:cp2),...
                    Z(pop+1,cp2+1:end));
            end
        else % Parents go Straight through
            Zc(pop,:) = Z(pop,:);
            Zc(pop+1,:) = Z(pop+1,:);
        end
        pop = pop + 2;
    end
    Z = Zc; 
%    clear Zc;
    
    % Mutation
%    clear i j;
    R = rand(size(Z)); % Check to see if each bit is going to have mutation
    for i=1:size(Z,1) % Row
        for j = 1:size(Z,2) % Column
            if R(i,j) <= Pm
                if Z(i,j) == '0'
                   Z(i,j) = '1';
                else
                    Z(i,j) = '1';
                end
            end
        end
    end

    % Stop Timer for Each Iteration
    Time = toc;
    
    % Save Optimal Values
        % ONLY WORKS FOR MIN SOLUTION
    [ minVal, indx ] = min(fz);
    optVal(iter+1,:) = [ Time, minVal, Zfit(indx,:) ];
    
    % Compute Convergence from previous run
    delRun = abs(minVal - minValPrev);
    minValPrev = minVal;
    if delRun > Res
        delRunIter = 0;
    else 
        delRunIter = delRunIter +1;
    end
    
iter = iter + 1;
end

% Save optVal for this run only
save(strcat(num2str(run),functionOptions{fun},num2str(nGAiter)),'optVal')
%         cols = 2 + Dim; % Columns in optVal
%         jj = nGAiter;
%         optVal(:,(2+cols*(jj-1)):(cols*jj)+1) = optValTemp;


end

% NOTES:
%
% 1) Should add the capability to switch between min and max evaluation
% 2) Remove the capability to add bounds
% 3) Number of dimensions input
% 4) Add Parfor
% 5) Add Run info Parameters in SBGA
% 6) Remove tic-toc in SBGA




















