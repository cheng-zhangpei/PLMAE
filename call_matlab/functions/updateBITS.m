function updateBITS(latestbit, i)
% USAGE: 
%       updateBITS(latestbit, i)
% 
%   Updates the latest bit decision in to the tree structure (heap-like-storage)
%   so that it enables subsequent LLR computation
% 
% PCparams structure is implicit parameter
%
% Non-Recursive implementation of the SCD update BITS routine
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

N = PCparams.N;
n = PCparams.n;

if i==N  %no bits need to be updated
    return;
elseif i<=(PCparams.N/2)
    PCparams.BITS(1,1) = latestbit;
else
    
%     %%%%%%TO IMPROVE LATER%%%%%%%%%
%       %FINDING FIRST INDEX OF '0'
%     i_bin = dec2bin(i-1,n);
%     for lastlevel = 1:n
%         if i_bin(lastlevel) == '0'
%             break;
%         end
%     end
    %%%%%%%% IMPROVED %%%%%%%%%%%
        lastlevel = PCparams.index_of_first0_from_MSB(i);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    PCparams.BITS(2,1) = latestbit;
    for lev=1:lastlevel-2 %forms level (lev+1) using values @ level (lev)
        st = 2^(lev-1);
        ed = 2^lev -1 ;
        for indx = st:ed
            PCparams.BITS(2,ed+2*(indx-st)+1) = mod( PCparams.BITS(1,indx)+PCparams.BITS(2,indx), 2 );
            PCparams.BITS(2,ed+2*(indx-st)+2) = PCparams.BITS(2,indx);
        end
    end
    
    lev=lastlevel-1;
    st = 2^(lev-1);
    ed = 2^lev -1 ;
    for indx = st:ed
        PCparams.BITS(1,ed+2*(indx-st)+1) = mod( PCparams.BITS(1,indx)+PCparams.BITS(2,indx), 2 );
        PCparams.BITS(1,ed+2*(indx-st)+2) = PCparams.BITS(2,indx);
    end
end

end