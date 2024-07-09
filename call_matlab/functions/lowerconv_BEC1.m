function llr_out=lowerconv_BEC1(llr_upper,llr_lower)
%
% llr = llr2/llr1 for BEC, where all three
% llr1,llr2,llr will be a part of a 3-size alphabet.
%    ---- we denote the three alphabets with "0,1,anything" where anything
%    other than 0/1 represents an erasure. But when we output, we choose
%    "2" to represent an erasure, as a standard.
% 
% The operator tabulated:
%                0  1  e
%             -----------
%           0| [ e  1  1;
%           1|   0  e  0;
%           e|   0  1  e ]
%

if llr_upper==0

    if llr_lower==0
        llr_out=2;
    elseif llr_lower==1
        llr_out=1;
    else
        llr_out=1;
    end
       
elseif llr_upper==1
    
    if llr_lower==0
        llr_out=0;
    elseif llr_lower==1
        llr_out=2;
    else
        llr_out=0;
    end

else %erasure

    if llr_lower==0
        llr_out=0;
    elseif llr_lower==1
        llr_out=1;
    else
        llr_out=2;
    end

end

end