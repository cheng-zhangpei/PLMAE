function u=systematic_pdecode(y,channel_string,channel_state,myfrozenlookup) 
% The same Successive Cancellation Decoding but when encoder is systematic.
% 
%        'myfrozenlookup' is optional argument, to allow using userdefined
%        frozen bits look-up, which may be created using build_a_lookup()
%        routine. (See help build_a_lookup)
% 
%   NOTE: 
% 1. Systematic variant of any linear-block-code is simply a permuted
%       mapping of 2^K message-words to corresponding code-words,
%       such that, messageword appears as a part of the codeword
%       itself (like a simpler concatenation of given k-bit message
%       and some (n-k)-bit redundancy)
% 
% 2. Under systematic variation, any linear block code would have a
%       similar BlockErrorRate/FrameErrorRate but a different BitErrorRate.
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

if(nargin==3) % When no special frozen-bits supplied
    myfrozenlookup=PCparams.FZlookup;
end

uu = pdecode(y,channel_string,channel_state,myfrozenlookup);
xx = pencode(uu);
u = xx(PCparams.FZlookup==-1);

end