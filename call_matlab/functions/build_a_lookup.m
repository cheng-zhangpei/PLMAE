function lookup=build_a_lookup(frozenbits,frozenbitindices) %last argumennt "frozenbitindices" is optional
% 
%    USAGE:
%           lookup=build_a_lookup(frozenbits)
%   
%       frozenbits -- (N-K) frozenbits
% 
%       frozenbitindices (optional) --
%                     (N-K) integer indices among 1,2,...N
%                       When not supplied, these are read from PCparams
%                       structure.
% 
%       lookup     -- Nx1 vector of elements {-1,0,+1} only
%                      with "frozenbits" filled in the natural order, at
%                      their N-K frozen-bit-indices. Rest all are -1's.
% 
%       PCparams structure is inherently assumed to know "N" "FZlookup" etc
%
%    RECALL:
%       Polar coding critically assumes K information bits
%       and N-K frozen-bits for both encoding and decoding. They
%       are to be padded together intelligently to form a vector
%       of Nx1 dimension, according to an order. This order is fully defined by 
%               1. A set of frozen-bit-indices
%               2. Equal number of frozen-bits (usually are all zeros)
% 
%    WHY THIS ROUTINE?
% 
%    For faster access, all the polar coding routines use a lookup-vector
%    of Nx1 size, to quickly determine which index is frozen to what bit,
%    and which index is not frozen:
%       frozenlookup(i) = -1 if "i" is not frozen
%                       =  0 if "i" is frozen to bit '0'
%                       =  1 if "i" is frozen to bit '1'
% 
%   By default, all frozen bits are all zeros. But when arbitrary
%   frozen-bits are used, the routines simply require a lookup vector
%   that includes these non-zero frozen-bits.
%       
%   The current routine precisely helps you build such a lookup vector,
%   just by passing (N-K) binary bits.
%   

global PCparams;
N=PCparams.N;
lookup=-1*ones(N,1);
    
if(nargin==1)
    lookup(PCparams.FZlookup ~= -1) = frozenbits;
else
    lookup(frozenbitindices) = frozenbits;
end

end

