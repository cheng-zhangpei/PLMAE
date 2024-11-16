function [x,y]=systematic_pencode(u,myfrozenlookup,algoname) 
% 
%       USAGE:
%            x = systematic_pencode(u,myfrozenlookup,algoname)
% 
%                        u     - A Kx1 binary message
% 
%    myfrozenlookup (optional) - A lookup vector of Nx1 size, with elements {-1, 0, 1}
%                                where i-th element = :
%                                     -1 :  if i-th bit is information
%                                      0 :  if i-th bit is frozen to bit '0'
%                                      1 :  if i-th bit is frozen to bit '1'
%                           When not supplied, all frozenbits will be what
%                           "PCparams" structure defines them by default.
% 
%         algoname (optional)  - Systematic Encoding Algorithm Name from [1]
%                                     One of:  'A' or 'B' or 'C'
% 
%                        x     - A Nx1 binary code (systematic-encoded form of "u")
% 
% Note that, The last two arguments "myfrozenlookup" and "algoname" both are optional.
% 
% PCparams structure is an implicit parameter
%
% Encodes 'K' message bits in 'u' and
% Returns 'N' encoded bits in 'x'
%       - Optionally, using "myfrozenlookup", the user-defined frozen-bits
%       supplied as a lookup-vector 
%       - Optionally, using one of the three possible
%       systematic polar encoding algorithms from [1]: 'A' or 'B' or 'C'
%
% The order of outputs is chosen so that when assigned to ONE
% variable directly, we get the (default) output as the desired codeword x.
%
% 'algoname' is a character that must be 'A' or 'B' or 'C', as in [1]
% Its default value is 'A', where each of these letters corresponds
% to a specific algorithm
%
% PCparams structure is implicit parameter
%
% Polar coding parameters (N,K,FZlookup,Ec,N0,LLR,BITS) are taken
% from "PCparams" structure FZlookup is a vector, to lookup each integer
% index 1:N and check if it is a message-bit location or frozen-bit location.
%     FZlookup(i)==0 or 1 ==> bit-i is a frozenbit
%     FZlookup(i)==-1 ==> bit-i is a messagebit
% 
% [1] Harish Vangala, Yi Hong, and Emanuele Viterbo,
%      "Efficient Algorithms for Systematic Polar Encoding",
%     IEEE Communication Letters, 2016.

if(nargin==1 || nargin==2)
    algoname='A';
end

if(nargin==1)
y = PCparams.FZlookup; %loads all frozenbits, incl. -1
x = PCparams.FZlookup;
else
y = myfrozenlookup; %loads all frozenbits, incl. -1
x = myfrozenlookup;
end

global PCparams;
N=PCparams.N;
n=PCparams.n;

x(x == -1) = u; 
x(x ~= -1) = -1;

if(algoname=='A')
    [y,x]=EncoderA(y,x);
elseif(algoname=='B')
    r=zeros(N,1);
    [~,y,x] = EncoderB(1,N,y,x,r);
elseif(algoname=='C')
    r=zeros(N,1);
    [y,x,~] = EncoderC(1,N,y,x,r);
else
    fprintf('\n Invalid Encoder Algorithm %c Supplied! (should be one of A B C)\n',algoname);
end

end