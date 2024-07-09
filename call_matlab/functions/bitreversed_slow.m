function i = bitreversed_slow(j,n)
% Consider an "n" bit representation of "j" and
% reverse the order of bits to get "i"
% 
%    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
%    WRITTEN BY: Harish Vangala, Emanuele Viterbo, and Yi Hong,
%                Dept of ECSE, Monash University, Australia.
% 
%    - Latest as on 2016-March-03
%    - Available ONLINE for free: is.gd/polarcodes
%    - Freely distributed for educational and research purposes
%    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

i=zeros(length(j),1);

for indx=1:length(j)
i(indx) = bin2dec(wrev(dec2bin(j(indx),n)));
end
% dec2bin() : produces a string of binary form of "j" in at least "n" bits
% wrev()    : produces a reversed string (here again a binary string)
% bin2dec() : binary "string" to decimal conversion

end
