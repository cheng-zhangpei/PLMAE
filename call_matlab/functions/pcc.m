function frozenlookup = pcc(N,K,channelstring,channelstate,frozenbits)%'frozenbits' is optional
% 
%   Perform the polar code construction aka, setting the frozen-parameters
%   of the polar codes.
% 
%   We use the simplest of all construction algorithms [1].
% 
%   Currently supported channels: 
%           1. AWGN(designSNR)
%           2. BSC(p)
%           3. BEC(eps)
% 1. 
% Perform the PCC at "design-SNR" := (Eb/N0) = channelparam,
% given in dB. The initial Bhattacharyya parameter = exp(-(K/N)*(Eb/N0))
% 
% 2.
% Perform the PCC at the given "design-p"= channelparam.
% The initial Bhattacharyya parameter = 2*sqrt(p(1-p)), where p is the channelparam
% 
% 3.
% Perform the PCC at the given "design-epsilon"= channelparam.
% The initial Bhattacharyya parameter = epsilon, where epsilon=channelparam
% 
% [1] Vangala, H.; Viterbo, E. & Hong, Y.
%   "A Comparative Study of Polar Code Constructions for the AWGN Channel",
%          arXiv:1501.02473 [cs.IT], 2015.
%
% USAGE:
% 
%       frozenlookup = pcc(N,K,channelparam,ChannelString)
% 
%           N         -   Blocklength
%           K         -   Message length
% 
%       channelstring -  Must be one of (case insensitive)
%                            'AWGN' (default)
%                          or 'BSC'
%                          or 'BEC'
% 
%       channelstate  -  Channel's state to be assumed during the PCC algorithm.
%                        Usually as an initial. It is one of:
%                         design-SNR (Eb/N0)
%                     or  design-p
%                     or  design-eps
%                *** Must match the ChannelString parameter (above) ***
% 
%       frozenbits (optional) -  User-defined (N-K)x1 frozenbits (ideally
%                   the FER/BER performance is identical for any choice,
%                   but are critical to be known at receiver. May be used
%                   in cryptographical ideas, for e.g.)
% 
%       frozenlookup  -  An easy Nx1 look-up table for knowing whether a
%                        bit-index is frozen or not. Its definition:
%            -----------------------------------------------------------
%            frozenlookup(i) = 0 or 1 --- if "i" is frozen index
%                            = -1   ---  if "i" is non-frozen/information index 
%            -----------------------------------------------------------
%    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
%    WRITTEN BY: Harish Vangala, Emanuele Viterbo, and Yi Hong,
%                Dept of ECSE, Monash University, Australia.
% 
%    - Latest as on 2016-March-03
%    - Available ONLINE for free: is.gd/polarcodes
%    - Freely distributed for educational and research purposes
%    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

if nargin==4  %default frozenbits are zeros
    frozenbits=zeros(N-K,1);
end
    
% fprintf(' ... '); %DEBUG

z = zeros(N,1);
%%%%%%%%%%%%%%%%%% INITIALIZATION %%%%%%%%%%%%%%%%%%%
if(strcmpi(channelstring,'AWGN'))
    designSNRdB=channelstate;
    designSNR = 10^(designSNRdB/10);
    z(1) = -(K/N)*designSNR; % In logdomain. Actual initial Bh.Param = exp(-(K/N) * (Eb/N0))
elseif(strcmpi(channelstring,'BSC'))
    designp=channelstate;
    z(1) = log(2)+0.5*log(designp)+0.5*log(1-designp);
           % In logdomain. Actual initial Bh.Param = 2*sqrt(p(1-p))
elseif(strcmpi(channelstring,'BEC'))
    designeps=channelstate;
    z(1) = log(designeps);
           % In logdomain. Actual initial Bh.Param = epsilon (erasure-prob)
else
    fprintf('\n\n     ERROR: invalid channel string "%s" supplied. It must be one of "AWGN" or "BSC" or "BEC"\n\n',channelstring);
    return;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%DEBUG
% fprintf('\n\n   PolarCodeConstruction -- %s %f \n\n',channelstring,channelstate);

for lev=1:log2(N)
    B=2^lev;
    for j=1:B/2
        T = z(j);
        z(j) = logdomain_diff( log(2)+T, 2*T );  %  2z - z^2
        z(B/2 + j) = 2*T;                        %  z^2
    end
end

[~,indices] = sort(z,1,'ascend');  %default sort is "Ascending"; 
                            % 1 refers to sort "columns" (not rows)
global PCparams;                            
PCparams.FZlookup2 = indices;

% %DEBUG-START
% fprintf('\n\n\tEstimated BLER of %s channel at param=%.2f is: %f\n\n',channelstring,channelstate,1-prod(1-exp(z(indices(1:K)))));
% %DEBUG-END

frozenlookup(sort(indices(K+1:end))) = frozenbits; % Could as well be userdefined, esp when called by initPC().
% Sorting is important, as we assume frozenbits are given as a subvector with natural order.

for j=1:K 
    frozenlookup(indices(j)) = -1; %locations in z containing least K values
end

end