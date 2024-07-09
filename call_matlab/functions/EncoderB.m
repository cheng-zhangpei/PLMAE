function [v,y,x]=EncoderB(i,j,y,x,r)
% EncoderB of the paper:
%    Harish Vangala, Yi Hong, and Emanuele Viterbo,
%   "Efficient Algorithms for Systematic Polar Encoding",
%    IEEE Communication Letters, 2015.
%

global PCparams;
n=PCparams.n;
N=PCparams.N;

m=floor((i+j)/2);
L=j-i+1;
v=zeros(L,1);

if(L==1)
    if PCparams.FZlookup(i)==-1
        y(i) = mod(x(i)+r(1), 2);
        v(1)=y(i);
    else
        x(i) = mod(y(i)+r(1), 2);
        v(1)=y(i);
    end
else
    v2=zeros(1,L/2);
    v1=v2;
    r2=r(1:L/2);
    r1=r((L/2 +1):L);

    [v1,y,x] = EncoderB(m+1,j,y,x,r1);
    v2 = mod(r2+v1,2);
    [v2,y,x]=EncoderB(i,m,y,x,v2);
    v2 = mod(v2+v1,2);
    v=[v2;v1];
end
end