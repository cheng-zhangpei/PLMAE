function u=pdecode_BEC(bec_y,myfrozenlookup)
% 
%   Usage:
%         pdecode_BEC(bec_y,myfrozenlookup)
%
% bec_y           : The ternary output of a BEC, with some channels erased,
%                   denoted with non-{0,1} symbols.
% 
% u               : Decoded message bits
% 
% myfrozenlookup (optional) - A lookup vector of Nx1 size, with elements {-1, 0, 1}
%                   where i-th element = :
%                           -1 :  if i-th bit is information
%                            0 :  if i-th bit is frozen to bit '0'
%                            1 :  if i-th bit is frozen to bit '1'
%                   When not supplied, all frozenbits will be what
%                   "PCparams" structure defines them by default.
% 
% "PCparams" structure is implicit parameter
% 
% Polar coding parameters (N,K,FZlookup etc) are taken
% from "PCparams" structure FZlookup is a vector, to lookup each integer
% index 1:N and check if it is a message-bit location or frozen-bit location.
%
%        FZlookup(i)==0 or 1 ==> bit-i is a frozenbit
%        FZlookup(i)==  -1   ==> bit-i is a messagebit
%
%    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
%    WRITTEN BY: Harish Vangala, Emanuele Viterbo, and Yi Hong,
%                Dept of ECSE, Monash University, Australia.
% 
%    - Latest as on 2016-March-03
%    - Available ONLINE for free: is.gd/polarcodes
%    - Freely distributed for educational and research purposes
%    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

global PCparams;
if(nargin==1) % When no special frozen-bits supplied
    myfrozenlookup=PCparams.FZlookup;
end

N=PCparams.N;

% Initializing the likelihoods to the right end of the butterfly ckt
PCparams.LLR = 0; %PCparams.BITS=-1;

PCparams.LLR(N:2*N-1) = bec_y;

d_hat = zeros(N,1);
% finalLRs = zeros(N,1); %DEBUG
for j=1:N

%         i = bitreversed(j-1,PCparams.n) +1 ; 
        i = PCparams.bitreversedindices(j) +1; % "+1" is to compensate matlab indexing (starting with 1 instead of 0)
        
        updateLLR_BEC(i); %BEC optimized version
%         finalLRs(i) = PCparams.LLR(1); %DEBUG
        
    if myfrozenlookup(i) == -1
        if (PCparams.LLR(1)~=0 && PCparams.LLR(1)~=1) %Erasure
            d_hat(i) = (rand > 0.5);  %Toss a coin
        else
            d_hat(i) = PCparams.LLR(1);
        end
    else
        d_hat(i) = myfrozenlookup(i);
    end
    
        updateBITS(d_hat(i),i); %Same as non-BEC case. No need of a special BEC version
end

u = d_hat ( myfrozenlookup == -1);

end