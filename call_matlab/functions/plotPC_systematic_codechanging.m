function plotPC_systematic_codechanging(N,K,channelstring,channelstate_range,MaxIterations) % last argument is optional
%   This is exactly the same as plotPC_systematic(), except for a
%   loss-of-one-argument that code is re-constructed -aka- frozenbits
%   are recomputed at every channel state, before the Monte-Carlo is run.
% 
% Plot the performance (FER/BER) curve for (N,K) polar code, by 
% 
%       1. Re-constructing the code at a given design-state of the channel
%       (existing construction is restored at the end)
%       2. Generating as many random messages 
%       3. Performing systematic-encoding on each sample 
%       4. SC-decoding extended to systematic-polar codes
%       5. Repeat at most 'MaxIterations' times, and report the stats
% =======================================================================
%    Usage: plotPC_systematic_codechanging(N,K,channelstring,design_channelstate,channelstate_range,MaxIterations)
%
%            N        -  Block length
%            K        -  Message length
% 
%       channelstring -  'AWGN' or 'BEC' or 'BSC'
% 
%       channelstate_range
%                     -  The range of channel state values at which code
%                        performance (FER & BER) is to be simulated.
%                          AWGN -> it is SNR:=Eb/N0 range in dB scale
%                          BSC  -> it is the range of p values
%                          BEC  -> it is the range of epsilon values
%                       (will be ascending sorted immediately within,
%               to aid avoiding useless MonteCarlo iterations and also for plots)
% 
%       design_channelstate - The design-state of the channel, 
%                             at which the PCC should be run (frozen bit
%                             indices are chosen)
%       MaxIterations - Maximum number of iterations to try, to get a
%               reasonable estimate of the BER/FER. It smartly stops
%               whenever 100 Frames are already received in error, where
%               the estimate is good enough. If we fail to get 100 frames
%               in error even after running MaxIterations, we stop running any higher quality.
%
%      **********************************************************
%       Results of FER, BER and channel_param_range are made available on command line 
%               (so called 'base' workspace)
%      **********************************************************
% 
%   Note: This routine, preserves the global structure PCparams. It changes
%   the structure but restores at the end. So subsequent calls to functions
%   pencode(),pdecode() etc all are not affected.
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

errorflag=0;
if nargin==2 %Quick-test-run
    channelstring='AWGN';
    design_channelstate=0; %dB
    channelstate_range = 0:1:2; %default SNR range to quickly see the o/p
    MaxIterations=10000;
else
    if(nargin==5 || nargin==6)
        if nargin==5
        MaxIterations=10000;
        end
        
        if (strcmpi(channelstring,'AWGN')||strcmpi(channelstring,'BEC')||strcmpi(channelstring,'BSC'))
            errorflag=0;
        else
            fprintf('\n\n   Invalid Channel String "%s" Supplied! (It should be one of "AWGN","BEC" or "BSC" only) \n\n',channelstring);
            errorflag=1;
        end
    else
        errorflag=1;
    end
end

if(errorflag==1)
    fprintf('\n   Usage: plotPC_systematic_codechanging(N,K,channelstring,channelstate_range,MaxIterations)\n');
    fprintf('\n       N  -  Blocklength');
    fprintf('\n       K  -  Message length (Rate = K/N)');
    fprintf('\n       channelstring - "AWGN" or "BEC" or "BSC"');
    fprintf('\n       channelstate_range  - the range of parameters at which Monte-Carlo simulation to be run');
    fprintf('\n       MaxIterations - Maximum Monte-Carlo iterations to be run\n\n');
    return;
end

PCparams1 = PCparams; %store the current

% Save some typing
chstr = channelstring;
chrange = sort(channelstate_range); % *IMP* if they are in "out-of-order"
xlab=[];
if(strcmpi(chstr,'AWGN'))
    xlab='Eb/N0';
    paramstr='SNR (Eb/N0) in dB';
elseif(strcmpi(chstr,'BSC'))
    xlab='Transition probability "p" of BSC';
    paramstr='p';
elseif(strcmpi(chstr,'BEC'))
    xlab='Erasure probability "\epsilon" of BEC';
    paramstr='\epsilon';
end
%%%%%%%%%%%%%%%%%%

MCsize = MaxIterations; %Montecarlo size
MinFrameErrors=100;
MinIters=500;

fprintf('\n * Max. Monte-Carlo Iterations = %d, ensuring %d frame errors, and a guaranteed minimum of %d iterations',MCsize,MinFrameErrors,MinIters);
fprintf('\n * Polar Code *re-constructed* for every channel state of %s-%s (i.e. different code everytime) ',chstr,paramstr);

global BER;
global FER;

BER = zeros(1,length(chrange));
FER = zeros(1,length(chrange));

    fprintf('\n * Channel-states (%s-%s) to be run: \n\t ',chstr,paramstr); disp(chrange);

for jj=1:length(chrange)
    % To ensure increasing quality of channels, 
    %     for a useful MonteCarlo stopping criteria.
    
    if(strcmpi(chstr,'AWGN'))
        j=jj;
    else
        j=length(chrange)-jj+1;
    end
    
    initPC(N,K,channelstring,chrange(j),1); %last optional argument '1' says "be silent" with no output at prompt.
    N=PCparams.N; %adjust in case of being not-a-power-of-2
    K=PCparams.K;
    
    fprintf('\n Now running: %s-%s=%f  [%d of %d]  \n\t Iteration-Counter: %53d',chstr,paramstr,chrange(j),j,length(chrange),0);
    
    tt=tic();
    for l=1:MCsize
        u=randi(2,K,1)-1; %Bernoulli(0.5)
        x=systematic_pencode(u);
        
        %Pass x via a simulated instance of the noisy channel, and grab the o/p
        y=OutputOfChannel(x,chstr,chrange(j));
        
        uhat = systematic_pdecode(y,chstr,chrange(j));
        
        nfails = sum(uhat ~= u);
        FER(j) = FER(j) + (nfails>0);
        BER(j) = BER(j) + nfails;
        
        if mod(l,20)==0
            for iiiii=1:53
                fprintf('\b');
            end
            fprintf(' %7d   ---- %7d FEs, %7d BEs found so far',l,FER(j),BER(j));
        end
        
        if l>=MinIters && FER(j)>=MinFrameErrors  %frame errors, sufficient to stop
            break;
        end
    end
    
    %%%%%%%EARLY STOPPING CRITERIA
    if(FER(j)<MinFrameErrors)
        FER(j) = FER(j)/l;
        BER(j) = BER(j)/(K*l);
        
        if(~strcmpi(chstr,'AWGN')) %For BEC and BSC, points in 'chrange' run in reverse order (increasing quality)
            FER(1:j-1) = 0;
            BER(1:j-1) = 0;
        else
            FER(j+1:end) = 0;
            BER(j+1:end) = 0;
        end
        
        fprintf('\n\t Total time taken: %.2f sec (%d samples), escaping the rest of the points due close to zero FER.',toc(tt),l);
        fprintf('\n\t   (To have more successful points, you must increase the Max. MonteCarlo iterations\n\t     passed as the last argument to plotPC(), and run again)\n');
        break;
    else
        FER(j) = FER(j)/l;
        BER(j) = BER(j)/(K*l);
 
        fprintf('\n\t Total time taken: %.2f sec (%d samples)',toc(tt),l);
    end
     fprintf('\n');
end

 %%%%%%% Set the plot's default settings, including window size
 set(0,'DefaultAxesLineStyleOrder',{'-*','--*','-.*'});
 scrsz = get(0,'ScreenSize');
 figure('Position',[70 scrsz(4)/2-150 scrsz(3)/2 scrsz(4)/2]); 
 %[left,bottom,width,height]
 hold all;
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 subplot(211);
 semilogy(chrange,FER,'LineWidth',2,'MarkerSize',8); grid on;
 titlestr = sprintf('N=%d R=%.2f Systematic Polar code performance (reconstructed everytime)',N,(K/N));
 title(titlestr,'fontsize',12,'fontweight','b');
 xlabel(xlab,'fontsize',12,'fontweight','b');
 ylabel('Frame Error Rate','fontsize',12,'fontweight','b');
 
 subplot(212);
 semilogy(chrange,BER,'LineWidth',2,'MarkerSize',8); grid on;
 titlestr = sprintf('N=%d R=%.2f Systematic Polar code performance (reconstructed everytime)',N,(K/N));
 title(titlestr,'fontsize',12,'fontweight','b');
 xlabel(xlab,'fontsize',12,'fontweight','b');
 ylabel('Bit Error Rate','fontsize',12,'fontweight','b');

fprintf('\n\n ****************************************************\n');
fprintf(    ' %4s-%-17s:',chstr,paramstr); disp(chrange);
fprintf(    '   Frame Error Rates   :'); disp(FER);
fprintf(    '   Bit Error Rates     :'); disp(BER);
fprintf(    ' ****************************************************\n\n');

% IMPORTANT: Make FER and BER available on commandline, for later use
evalin('base','global FER BER channel_param_range');  %If not used, user
% should explicitly run "global FER BER" to get access to result values.
% (Until then these two are not even visible, though made accessible).
% Alternatively, one may use "assignin()" with a different logic. This will allow us to skip globals completely.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

PCparams=PCparams1; %restore the original
end