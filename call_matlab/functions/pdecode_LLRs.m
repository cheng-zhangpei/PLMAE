function u=pdecode_LLRs(initial_LLRs,myfrozenlookup)
% 
%    Usage :
%       pdecode_LLRs(initial_LLRs,myfrozenlookup)
%
% initial_LLRs    : The channel-independent, initial LLRs, which is
%                   sufficient for a SC decoder.
% 
% myfrozenlookup (optional)  : 
%                   A lookup vector of Nx1 size, with elements {-1, 0, 1}
%                   where i-th element = 
%                           -1 :  if i-th index has an information bit
%                            0 :  if i-th index is frozen to bit '0'
%                            1 :  if i-th index is frozen to bit '1'
%                   When not supplied, all frozenbits will be what
%                   "PCparams" structure defines them by default.
% 
% u               : Decoded message bits
% 
% "PCparams" structure is an implicit parameter
%
% Polar coding parameters (N,K, other memory elements) are taken
% from "PCparams" structure
% 
%    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
%    WRITTEN BY: Harish Vangala, Emanuele Viterbo, and Yi Hong,
%                Dept of ECSE, Monash University, Australia.
% 
%    - Latest as on 2016-March-03
%    - Available ONLINE for free: is.gd/polarcodes
%    - Freely distributed for educational and research purposes
%    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
%

global PCparams;

if(nargin==1) % When no special frozen-bits supplied
    myfrozenlookup=PCparams.FZlookup;
end

N=PCparams.N;

% Initializing the likelihoods (i.e. the right end of the butterfly str)
PCparams.LLR = 0; %PCparams.BITS=-1;

PCparams.LLR(N:2*N-1) = initial_LLRs;

d_hat = zeros(N,1);
% finalLRs = zeros(N,1); %DEBUG
for j=1:N
    
%         i = bitreversed(j-1,PCparams.n) +1 ; 
        i = PCparams.bitreversedindices(j) +1 ; % "+1" is to compensate matlab indexing (starting with 1 instead of 0)
        
        updateLLR(i);
%         finalLRs(i) = PCparams.LLR(1); %DEBUG
        
    if myfrozenlookup(i) == -1
        if PCparams.LLR(1) > 0
            d_hat(i) = 0;
        else
            d_hat(i) = 1;
        end
    else
        d_hat(i) = myfrozenlookup(i);
    end
    
        updateBITS(d_hat(i),i);
end

u = d_hat ( myfrozenlookup == -1);

end