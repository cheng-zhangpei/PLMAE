function updateLLR_BEC(i)
% PCparams structure is implicit parameter
%
% Non-Recursive implementation of the SCD update Likelihoods routine
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

n = PCparams.n;


if i==1
    nextlevel=n;
else
    
%     %%%%%%TO IMPROVE LATER%%%%%%%%%
%       %FINDING FIRST INDEX OF '1'
%     i_bin = dec2bin(i-1,n);
%     for lastlevel = 1:n
%         if i_bin(lastlevel) == '1'
%             break;
%         end
%     end
    %%%%%%%% IMPROVED %%%%%%%%%%%%%
        lastlevel = PCparams.index_of_first1_from_MSB(i);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%% Initialize with lowerconv() %%%%%%
        st = 2^(lastlevel-1);
        ed = 2^(lastlevel) -1;
        for indx = st:ed
            if PCparams.BITS(1,indx)==0
            PCparams.LLR(indx) = lowerconv_BEC0(...
                           PCparams.LLR(ed+2*(indx-st)+1), ...
                           PCparams.LLR(ed+2*(indx-st)+2) ...
                           );
            else
            PCparams.LLR(indx) = lowerconv_BEC1(...
                           PCparams.LLR(ed+2*(indx-st)+1), ...
                           PCparams.LLR(ed+2*(indx-st)+2) ...
                           );
            end
        end
        nextlevel = lastlevel-1;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Proceed upwards the TREE, with upperconv() till ROOT node
for lev = nextlevel :-1: 1
    st = 2^(lev-1);
    ed = 2^lev - 1;
    for indx = st:ed
        PCparams.LLR(indx) = upperconv_BEC(PCparams.LLR(ed+2*(indx-st)+1), ...
                       PCparams.LLR(ed+2*(indx-st)+2));
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end