function llr_out=upperconv_BEC(llr_upper,llr_lower)
% 
%       USAGE:
%           llr_out=upperconv_BEC(llr_upper,llr_lower)
% 
%       BEC-optimized equivalent of upperconv() in AWGN/BSC
%
% llr = (llr1*llr2 +1) / (llr1 + llr2) for BEC, where all three
% llr1,llr2,llr will be a part of a 3-size alphabet.
%    ---- we denote the three alphabets with "0,1,anything" where anything
%    other than 0/1 represents an erasure. But when we output, we choose
%    "2" to represent an erasure, as a standard choice.
% 
% The operator tabulated:
%                0  1  e
%             -----------
%           0| [ 0  1  e;
%           1|   1  0  e;
%           e|   e  e  e ]
%

if llr_upper==0

    if llr_lower==0
        llr_out=0;
    elseif llr_lower==1
        llr_out=1;
    else
        llr_out=2; %erasure
    end
       
elseif llr_upper==1
    
    if llr_lower==0
        llr_out=1;
    elseif llr_lower==1
        llr_out=0;
    else %erasure
        llr_out=2; %erasure
    end
    
else %erasure 
%     for any llrlower
        llr_out=2; %erasure
end

end