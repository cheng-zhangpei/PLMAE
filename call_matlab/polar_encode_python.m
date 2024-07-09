function [mess,encoded_mess] = polar_encode_python(avail_pix_num,N,K,crossover_p)
    avail_pix_num = double(avail_pix_num);
    N = double(N);
    K = double(K);
    crossover_p = double(crossover_p);
    mess_len = floor(avail_pix_num/N)*K;
    %disp(mess_len);
    code_num = floor(mess_len/K);
    encode_len = code_num*N;
    initPC(N,K,'BSC',crossover_p);
    mess = randi([0,1],mess_len,1);
    encoded_mess = zeros(1,encode_len);
    for c = 1:code_num
        u = mess((c-1)*K+1:c*K);
        x=pencode(u);
        %encoded_mess = [encoded_mess,x];
        encoded_mess((c-1)*N+1:c*N) = x;
    end
       