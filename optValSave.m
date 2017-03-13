function [  ] = optValSave( functionOptions, funcInputs, fun, run )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

if nargin < 4
    run = '';
end

runInfo = funcInputs(fun,:)';
Dim   = funcInputs(fun,5); % Number of Dimensions
Niter = funcInputs(fun,6); % Maximum allowed number of iterations
nGA = funcInputs(fun,7); % Number of iterations for GA Code

cols = 2 + Dim; % Columns in optVal | 2 = [ Time, minFE ]
optValStore = zeros(Niter, 1+nGA*(2+Dim));
optValStore(1:size(runInfo,1),1) = runInfo;

for jsave = 1:nGA
    optValTemp = importdata(strcat(num2str(run),functionOptions{fun},num2str(jsave),'.mat'));
    row = size(optValTemp,1);
    optValStore(1:row,(2+cols*(jsave-1)):(cols*jsave)+1) = optValTemp; 
end
save(strcat(num2str(run),functionOptions{fun}),'optValStore');

end

