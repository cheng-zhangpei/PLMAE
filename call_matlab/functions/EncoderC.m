function [y,x,r]=EncoderC(i,j,y,x,r)
% EncoderC of the paper:
%    Harish Vangala, Yi Hong, and Emanuele Viterbo,
%   "Efficient Algorithms for Systematic Polar Encoding",
%    IEEE Communication Letters, 2015.
%

global PCparams;

if(i==j)
    if PCparams.FZlookup(i)==-1
        y(i) = mod(x(i)+r(i), 2);
    else
        x(i) = mod(y(i)+r(i), 2);
    end
else
    m=floor((i+j)/2);

    [y,x,r] = EncoderC(m+1,j,y,x,r);
    r(i:m) = mod(r(i:m)+r(m+1:j)+x(m+1:j),2);
    [y,x,r] = EncoderC(i,m,y,x,r);
    r(i:m) = mod(r(i:m)+r(m+1:j)+x(m+1:j),2);
end
end