function u=pdecode(y,channel_string,channel_state,myfrozenlookup)
% 
%     Usage:
%        u=pdecode(y,channel_string,channel_state,myfrozenlookup)
% 
%  Where,
%           y - Received bits in the channel specified by other two
%               arguments
% 
% channel_string - 'AWGN' or 'BEC' or 'BSC'
% 
% channel_state  - The 'SNR:=Eb/N0' or 'epsilon' or 'p'
% 
% myfrozenlookup (optional) - A lookup vector of Nx1 size, with elements {-1, 0, 1}
%                   where i-th element = :
%                           -1 :  if i-th bit is information
%                            0 :  if i-th bit is frozen to bit '0'
%                            1 :  if i-th bit is frozen to bit '1'
%                   When not supplied, all frozenbits will be what
%                   "PCparams" structure defines them by default.
% 
%      u         - Decoded message bits (OUTPUT)
% 
% Polar coding parameters (N,K,FZlookup,LLR,BITS etc) are implicitly taken
% from "PCparams" structure.
% 
% NOTES:
% 1. 
%   The main parameters used from "PCparams" structure is "FZlookup"
% 
%       FZlookup is a vector, to lookup each integer index 1:N
%       and check if it is a message-bit location or frozen-bit location.
%
%        FZlookup(i)==0 or 1 ==> bit-i is a frozenbit
%        FZlookup(i)==  -1   ==> bit-i is a messagebit
%
%   The other two are allocated memory resources.
% 
%   PCparams.LLR  : Log-Likelihood Ratios data structure for SC
%                   decoding vector of 1 x 2N-1
%   PCparams.BITS : Intermediate bit decisions for SC decoding
%                    matrix of 2 x N-1
% 
% 2.
%    SNR vs. Eb/N0 :
%           By definition, SNR==Eb/N0.
% 
%   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
%    WRITTEN BY: Harish Vangala, Emanuele Viterbo, and Yi Hong,
%                Dept of ECSE, Monash University, Australia.
% 
%    - Latest as on 2016-March-03
%    - Available ONLINE for free: is.gd/polarcodes
%    - Freely distributed for educational and research purposes
%   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

global PCparams;

if(nargin==3) % When no special frozen-bits supplied
    myfrozenlookup=PCparams.FZlookup;
end

% Initializing the likelihoods at the right end of the butterfly ckt
if(strcmpi(channel_string,'AWGN'))
    EbN0 = 10^(channel_state/10); %dB to linear of SNR:=Eb/N0
    initialLLRs = - 2 * sqrt(2*(PCparams.K/PCparams.N)*EbN0) * y; 
    % Explanation:
    % ------------
    %   y(i) = x(i) + n; 
    %           x(i) \in {0==-sqrt(2R*Eb/N0),1==+sqrt(2R*Eb/N0)}; n ~ Gaussian(0,1)
    %
    %                  Pr{y(i) | x(i) = 0} 
    %   LLR(i) = log  ---------------------
    %                  Pr{y(i) | x(i) = 1}
    
    u = pdecode_LLRs(initialLLRs,myfrozenlookup);

elseif(strcmpi(channel_string,'BSC'))
    p=channel_state;
    llr1 = log(p) - log(1-p); %LLR of y=1, in BSC(p)
    initialLLRs = (2*y - 1) * llr1;  %y is binary, 
    u = pdecode_LLRs(initialLLRs, myfrozenlookup);
    
elseif(strcmpi(channel_string,'BEC'))
    u = pdecode_BEC(y,myfrozenlookup);
end

end