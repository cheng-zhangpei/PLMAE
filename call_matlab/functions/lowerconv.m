function llr = lowerconv(upperdecision, upperllr, lowerllr)
% 
%   USAGE:
%       llr = lowerconv(upperdecision, upperllr, lowellr)
% 
%    The likelihood transformation of second kind
%
% *****PERFORMS IN LOG DOMAIN****
% llr = lowerllr * upperllr  -- if uppperdecision == 0
% llr = lowerllr / upperllr  -- if uppperdecision == 1
%

if upperdecision==0
    llr = lowerllr + upperllr;
else
    llr = lowerllr - upperllr;
end

end