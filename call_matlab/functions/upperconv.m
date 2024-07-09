function llr=upperconv(llr1,llr2)
% 
%       USAGE:
%           llr=upperconv(llr_upper,llr_lower)
% 
% The likelihood transformation of first-kind
%
% *****PERFORMS IN LOG DOMAIN*****
% llr = (llr1*llr2 +1) / (llr1 + llr2)
%

llr = logdomain_sum(llr1+llr2,0) - logdomain_sum(llr1,llr2);
end