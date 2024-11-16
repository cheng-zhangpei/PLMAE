function x=pencode(u,myfrozenlookup) % "myfrozenlookup" is optional
%       USAGE:
%            x = pencode(u,myfrozenlookup)
% 
%                        u     - A Kx1 binary message
% 
%    myfrozenlookup (optional) - A lookup vector of Nx1 size, with elements {-1, 0, 1}
%                                where i-th element = :
%                                     -1 :  if i-th bit is information
%                                      0 :  if i-th bit is frozen to bit '0'
%                                      1 :  if i-th bit is frozen to bit '1'
%                           When not supplied, all frozenbits will be what
%                           "PCparams" structure defines them by default.
% 
%                        x     - A Nx1 binary code (encoded form of "u")
% 
% PCparams structure is implicit parameter
%
% Encode 'K' message bits in 'u' and
% Return 'N' encoded bits in 'x'
%       Optionally, using "myfrozenlookup", the user-defined frozen-bits
%       supplied as a lookup-vector 
%
% Polar coding parameters (N,K,FZlookup etc) are taken
% from "PCparams" structure FZlookup is a vector, to lookup each integer
% index 1:N and check if it is a message-bit location or frozen-bit location.
%
% FZlookup(i)==0 or 1 ==> bit-i is a frozenbit
% FZlookup(i)==-1 ==> bit-i is a messagebit
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

%%%%%%AVOIDING MOST CHECKS LIKE THIS. Keeps responsibility on the users calling it.
% if(length(u)~=PCparams.K)
%     fprintf('Error: pencode() - invalid size of the input-message=%d (it should be %d)',length(u),PCparams.K);
%     return;
% end

% Actual logic:
% d(PCparams.FZlookup == -1) = u; %Dimensions should match. Otherwise ERRR here.
% 
% d(PCparams.FZlookup ~= -1) = PCparams(PCparams.FZlookup==-1);
%
% Replaced, better logic:
if(nargin==1) %no special (user-defined) frozenbits
  x = PCparams.FZlookup; %loads all frozenbits, incl. -1
else %if special frozenbits are supplied
  x = myfrozenlookup;
end

x (x == -1) = u; % -1's will get replaced by message bits below

n=PCparams.n;

for i=1:n
    B = 2^(n-i+1);
    nB = 2^(i-1);
    for j=1:nB
        base = (j-1)*B;
        for l=1:B/2
            x(base+l) = mod( x(base+l)+x(base+B/2+l), 2 );
        end
    end
end

%returns x
end