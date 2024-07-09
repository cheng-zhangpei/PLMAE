function i = bitreversed(j,n)
% Consider an "n" bit representation of "j" and
% reverse the order of bits to get "i"
%
% This being a fast version, assumes the availability of precomputed
% "PCparams.bitreversedindices" value.
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

i=zeros(length(j),1);

for indx=1:length(j)
i(indx) = PCparams.bitreversedindices(j(indx)+1);
end

end
