function [ B ] = Encode( D, L, XL, XU )
%Encode - Scaled Decimal to Binary
%   For use with Decode function, in Genetic Algorithm
%   Converts Decimal value to Binary on bounds set by XL and XU.
%   Such that in any string:
%       '00000000' - Represents XL
%       '11111111' - Represents XU
%
%   Inputs:
%       D  - Scaled Decimal representation
%       L  - Length of 1 Variable Binary String
%       XL - Lower bound in 'double'
%       XU - Upper bound in 'double'
%   Output:
%       B  - Binary String
%
%

% Number of Rows in D
rows = size(D,1);
% Number of Columns in D
cols = size(D,2);

for i = 1:rows
    for j = 1:cols 
        Dij = ( (D(i,j) - XL)*(2^L - 1) ) / (XU - XL); % Downscaled Decimal
        B(i,(1+L*(j-1)):(L*j)) = dec2bin(Dij,L);
    end
end

end
