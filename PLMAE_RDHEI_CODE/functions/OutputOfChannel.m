function y=OutputOfChannel(x,channel_string,channel_state)
% 
%   USAGE:
%       y=OutputOfChannel(x,channel_string,channel_state)
% 
% Given a binary "x" (matrix/vector), simulates a channel
% specified by "channel_string" == 'BSC' or 'BEC' or 'AWGN';
% modulates if necessary, adds noise and returns the output.
% Modulation is assumed BPSK, by-default, only for AWGN channel.
%
%   The "channel_state" is :
%       Ec/N0 -aka- SNR when channel_string=='AWGN'
%       p  -aka- the channel transision prob. when channel_string=='BSC'
%       epsilon  -aka- the erasure prob. when channel_string=='BEC'
% 
%    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
%    WRITTEN BY: Harish Vangala, Emanuele Viterbo, and Yi Hong,
%                Dept of ECSE, Monash University, Australia.
% 
%    - Latest as on 2016-March-03
%    - Available ONLINE for free: is.gd/polarcodes
%    - Freely distributed for educational and research purposes
%    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

%No checks performed on arguments to encourage a speedup.

global PCparams;

        if(strcmpi(channel_string,'AWGN'))
            % Assume the NORMALIZATION:
            %     N0/2, the noise-variance is assumed to be unity.
            % BPSK: 0 -> +sqrt(R*Eb/N0);  1 -> -sqrt(R*Eb/N0)
            
            sqrtEcN0 = sqrt(PCparams.K/PCparams.N) * 10^(channel_state/20);
            y = (2*x-1)*sqrtEcN0*sqrt(2) + randn(size(x)); %AWGN with normalization
            
        elseif(strcmpi(channel_string,'BSC'))
            p=channel_state;
            y = mod(x+(rand(size(x))<p),2); %flip with prob "p"
            
        elseif(strcmpi(channel_string,'BEC'))
            eps=channel_state;
            y = x + (rand(size(x))<eps)*2;  %Erase with prob "eps" [(i.e. make some elements >1)]
            
        else %ERROR CASE ignored silently
            strg = sprintf('OutputOfChannel(): Invalid channel_string "%s" supplied. (is not one of the allowed strings BSC/BEC/AWGN)',channel_string);
            warning(strg);
            y = x; %as is
        end
end