function [y,x]=EncoderA(y,x)
% EncoderA of the paper:
%    Harish Vangala, Yi Hong, and Emanuele Viterbo,
%   "Efficient Algorithms for Systematic Polar Encoding",
%    IEEE Communication Letters, 2015.
%

global PCparams;
n=PCparams.n;
N=PCparams.N;

X = zeros(PCparams.N,1+PCparams.n); %initialize
X(:,1) = y;
X(:,n+1) = x;

for i=N:-1:1
    if(PCparams.FZlookup(i)==-1)
        s=n+1;
        delta=-1;
    else
        s=1;
        delta=1;
    end

    str=dec2bin(i-1,n);
    
    for j=1:n
        t = s+j*delta;
        l=min([t,t-delta]);
        kappa=2^(n-l);
        if(str(l)=='0')
            X(i,t) = mod(X(i,t-delta)+X(i+kappa,t-delta), 2);
        else
            X(i,t) = X(i,t-delta);
        end
    end
end
y=X(:,1);
x=X(:,n+1);
% X
end