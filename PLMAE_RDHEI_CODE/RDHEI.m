clear;
filename = '45';
root_path = './pre_im0.75/';

bitplane = 3; %LSB plane
crossover_p = 0.3;%preset crossover probability

N = 16384;%codeword length
K = 420; %message bits
t = 2^(bitplane-1);


im_mask = double(imread(strcat(root_path,'im_mask.bmp')));
imo = double(imread(strcat(filename,'.bmp')));
im_pre = load(strcat(filename,'.txt'));



s = find(im_mask==255);
avail_pix_num = length(s);

 
initPC(N,K,'BSC',crossover_p);


pix_set = zeros(avail_pix_num,2);
pix_set(:,1) = mod(s-1,512)+1; 
pix_set(:,2) = ceil(s/512); 
pixel_shuffle = randperm(avail_pix_num);

mess_len = floor(avail_pix_num/N)*K;
code_num = floor(mess_len/K);
encode_len = code_num*N;
mess = randi([0,1],mess_len,1);
encoded_mess = zeros(1,encode_len);
for c = 1:code_num
    u = mess((c-1)*K+1:c*K);
    x=pencode(u);
    %encoded_mess = [encoded_mess,x];
    encoded_mess((c-1)*N+1:c*N) = x;
end

%«∂»Î–≈œ¢
immark = imo;
for c = 1:encode_len
    b = encoded_mess(c);
    px = pix_set(pixel_shuffle(c),1);
    py = pix_set(pixel_shuffle(c),2);
    if b == 1
        immark(px,py) = bitxor(immark(px,py),t);
    end
end

ext_mess = zeros(1,encode_len);
for c = 1:encode_len
    px = pix_set(pixel_shuffle(c),1);
    py = pix_set(pixel_shuffle(c),2);
    pv = immark(px,py);
    pvf = bitxor(immark(px,py),t);
    pre_pix = im_pre(px,py);
    if abs(pv-pre_pix) <= abs(pvf-pre_pix)
        ext_mess(c) = 0;
    else
        ext_mess(c) = 1;
    end
end

decoded_mess = zeros(1,mess_len);
for c = 1:code_num
    ex_word = ext_mess((c-1)*N+1:c*N);
    de_word = pdecode(ex_word,'BSC',crossover_p);
    decoded_mess((c-1)*K+1:c*K) = de_word;
end
if sum(decoded_mess ~= mess') ~= 0            
      disp('-----------------------------end-------------------------------------------------')
    disp('Fail, the RDHEI framework cannot achieve fully reversibility using the code rate');

else
    disp('-----------------------------end-------------------------------------------------')
    disp(strcat('The available K is  ',string(K)));
    disp(strcat('The capacity is  ',string(K*code_num/262144)));
end


        
      

