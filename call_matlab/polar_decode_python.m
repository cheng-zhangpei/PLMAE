function decoded_mess = polar_decode_python(ext_mess,N,K,crossover_p)
    ext_mess = double(ext_mess);
    N = double(N);
    K = double(K);
    crossover_p = double(crossover_p);
    encoded_mess_len = length(ext_mess);
    code_num = floor(encoded_mess_len/N);
    mess_len = code_num*K;
    initPC(N,K,'BSC',crossover_p);

    decoded_mess = zeros(1,mess_len);
    for c = 1:code_num
        ex_word = ext_mess((c-1)*N+1:c*N);
        de_word = pdecode(ex_word,'BSC',crossover_p);
        decoded_mess((c-1)*K+1:c*K) = de_word;
    end