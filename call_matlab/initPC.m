function initPC(N,K,design_channelstring,design_channelstate,silentflag,frozenbits) %Optional: N0, designSNRdB and silentflag (last three arguments)
%
% This prepares the collection of all implicit parameters related to
% polar coding & SC decoding; to be used by all subsequent routines later.
%   (Including the memory resources to be used by the polar SC decoding)
%
%       USAGE:
%            initPC(N,K,design_channelstring,design_state,silentflag)
% 
%            N  -  Blocklength; (*immediately adjusted to the least power-of-2 >=N*)
% 
%            K  -  Message length (Rate = K/N); 
% 
%       design_channelstring -  Must be one of (case insensitive)
%                            'AWGN' (default)
%                          or 'BSC'
%                          or 'BEC'
% 
%       design_channelstate  -  Channel's state to be assumed during the
%            PCC algorithm. Usually as an initial. It must be one of:
%                         design-SNR (Default: 0dB;  := Eb/N0,  where (K*Eb/N) is the energy used during BPSK modulation of coded-bits)
%                     or  design-p
%                     or  design-eps
%                *** Must match the ChannelString parameter (above) ***
% 
%       silentflag (optional) -  Whether to print the last result or not
%                                        ** defaults to 0 **
%                   (useful when automated for multiple runs in a Monte-Carlo simulation)
% 
%       frozenbits (optional) -  User-defined (N-K)x1 frozenbits (ideally
%                   the FER/BER performance is identical for any choice,
%                   but are critical to be known at receiver. May be used
%                   in cryptographical ideas, for e.g.)
% 
%        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%        Notes: 1. This routine/equivalent must be called (as many times
%                   if needed to adjust everything right), before we use
%                   any other utility around this package, except the routines of the name "plotPCxxx()"
% 
%               2. For AWGN channels, we assume SNR=Eb/N0 defines the channel-state,
%                   where (K*Eb/N) is the energy used by BPSK modulation of encoded-bits.
%        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% [1]   Vangala, H.; Viterbo, E. & Hong, Y.
%   "A Comparative Study of Polar Code Constructions for the AWGN Channel",
%          arXiv:1501.02473 [cs.IT], 2015.
% 
%    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
%    WRITTEN BY: Harish Vangala, Emanuele Viterbo, and Yi Hong,
%                Dept of ECSE, Monash University, Australia.
% 
%    - Latest as on 2016-March-03
%    - Available ONLINE for free: is.gd/polarcodes
%    - Freely distributed for educational and research purposes
%    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

if nargin==2
    design_channelstring='AWGN';
    design_channelstate=0; %dB, the designSNR for default channel = AWGN
    silentflag=0;
    frozenbits=zeros(N-K,1);
elseif nargin==4 
    silentflag=0;
    frozenbits=zeros(N-K,1);
elseif nargin==5
    frozenbits=zeros(N-K,1);
elseif nargin==6
    % do nothing
else
    fprintf('\n   Usage: initPC(N,K,design_channelstring,design_channelstate,silentflag,frozenbits)\n');
    fprintf('\n       N  -  Blocklength (*will be immediately adjusted to the least power-of-2 >=N*)');
    fprintf('\n       K  -  Message length (Rate = K/N)');
    fprintf('\n       design_channelstring (default:AWGN) -  One of "AWGN" (default) or "BSC" or "BEC" ');
    fprintf('\n       design_channelstate  (default: 0dB) -  Parameter of the channel (design-SNR/design-p/design-epsilon) for PCC (default: design-SNR=0dB)');
    fprintf('\n       silentflag (optional) - whether to print a (detailed) message showing the results');
    fprintf('\n       frozenbits (optional) - store & use these non-zero frozenbits at frozen indices during default versions of encoding/decoding');
    fprintf('\n\n   Note: This routine must be called, before we use any other utility around here.\n\n');
    return;
end

addpath(genpath('./functions')); %All helping routines. (V.IMP)

global PCparams;

% Adjust N & n to the next power of 2, incase supplied N is not a power-of-2.
n = ceil(log2(N)); 
N = 2^n;

% PCparams = struct('N', N, ...
%                   'K', K, ...
%                   'n', n, ...
%                   'FZlookup', zeros(N,1), ...
%                   'design_channelstring', design_channelstring, ...
%                   'design_channelstate', design_channelstate, ...
%                   'LLR', zeros(1,2*N-1), ...
%                   'BITS', zeros(2,N-1),...
%                   'bitreversedindices',zeros(N,1),...
%                   'index_of_first0_from_MSB',zeros(N,1),...
%                   'index_of_first1_from_MSB',zeros(N,1));

PCparams = struct('N', N, ...
                  'K', K, ...
                  'n', n, ...
                  'FZlookup', zeros(N,1), ...
                  'design_channelstring', design_channelstring, ...
                  'design_channelstate', design_channelstate, ...
                  'LLR', zeros(1,2*N-1), ...
                  'BITS', zeros(2,N-1),...
                  'bitreversedindices',zeros(N,1),...
                  'index_of_first0_from_MSB',zeros(N,1),...
                  'index_of_first1_from_MSB',zeros(N,1),...
                  'FZlookup2', zeros(N,1));


%%%%%%%%%%%%%% PRECOMPUTATIONS %%%%%%%%%%%%%%%%%%
for i=1:N
    PCparams.bitreversedindices(i) = bitreversed_slow(i-1,PCparams.n);
    
    %FINDING FIRST INDEX OF '1'
    i_bin = dec2bin(i-1,PCparams.n);
    for lastlevel = 1:PCparams.n
        if i_bin(lastlevel) == '1'
            break;
        end
    end
    PCparams.index_of_first1_from_MSB(i) = lastlevel;
    
    %FINDING FIRST INDEX OF '0'
    i_bin = dec2bin(i-1,PCparams.n);
    for lastlevel = 1:PCparams.n
        if i_bin(lastlevel) == '0'
            break;
        end
    end
    PCparams.index_of_first0_from_MSB(i) = lastlevel;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

PCparams.FZlookup = pcc(N,K,design_channelstring,design_channelstate,frozenbits);

if(~silentflag)
%fprintf('\n All polar coding parameters & resources initialized. (in a structure - "PCparams") \n');
% disp(PCparams);
end

end
