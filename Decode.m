function [ D ] = Decode( B, L, XL, XU )
%Decode - Scaled Binary to Decimal
%   For use with Encode function, in Genetic Algorithm
%   Converts Binary value to Decimal on bounds set by XL and XU.
%   Such that in any string:
%       '00000000' - Represents XL
%       '11111111' - Represents XU
%
%   Inputs:
%       B  - Binary String
%       L  - Length of 1 Variable Binary String
%       XL - Lower bound in 'double'
%       XU - Upper bound in 'double'
%   Output:
%       D  - Scaled Decimal representation
%
%

% Number of Rows in B
rows = size(B,1);
% Number of Columns in B
cols = size(B,2);
% Number of Variables in Binary String
vars = cols/L;

for i = 1:rows
    for j = 1:vars 
        Bij = bin2dec(B(i,(1+L*(j-1)):(L*j))); % Decimal for Row/Col
        D(i,j)= XL + (XU - XL)/(2^L - 1)*Bij;
    end
end

end
