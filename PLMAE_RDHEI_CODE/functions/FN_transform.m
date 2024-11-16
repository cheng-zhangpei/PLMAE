function x=FN_transform(d)
% 
% USAGE:
%       x=FN_transform(d)
% 
%   Given F=[1 1;0 1]
%   F^{x n} or n-fold-Kronecker product of F is taken and is multiplied
%   with the vector "d" to obtain "x" as: 
%               x = F^{x n} * d, in binary-field efficiently with just O(NlogN) complexity.
%
%       It may be considered for encoding, but the encoder repeats the
%       logic.

N= length(d);
n=log2(N);

if(nextpow2(N) ~= n)
    fprintf('\nError: FN_transform(d) -- size of the input vector %d is not a power-of-2',N);
    fprintf('\n Exiting the routine immediately.');
else
    
    for i=1:n
        B = 2^(n-i+1);
        nB = 2^(i-1);
        for j=1:nB
            base = (j-1)*B;
            for l=1:B/2
                d(base+l) = mod( d(base+l)+d(base+B/2+l), 2 );
            end
        end
    end
end

x=d;
end