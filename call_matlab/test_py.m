clear;
[mess,encoded_mess] = polar_encode_python(16384,8192,4096,0.01);

r = randperm(16384);
encoded_mess2 = encoded_mess;
for n=1:160
    encoded_mess(r(n)) = 1 - encoded_mess(r(n));
end

decoded_mess = polar_decode_python(encoded_mess,8192,4096,0.01);
disp(sum(encoded_mess2==encoded_mess))
disp(sum(decoded_mess==mess.'))