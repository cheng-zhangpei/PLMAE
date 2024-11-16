function test_systematic()
%
% Check the working of systematic encoder by encoding multiple samples
%

global PCparams;
initPC(64,32,2,2,0,1);

nsamples=100;
N=PCparams.N;
K=PCparams.K;
n=PCparams.n;

tic; count=0;
for i=1:nsamples
    u=(rand(K,1)>0.5);
    [x,y]=systematic_pencode(u,'A');
    
    if( x(PCparams.FZlookup==-1)==u )
        uu=y(PCparams.FZlookup==-1);
        xx=pencode(uu);
        if(xx == x)
            count=count+1;
        end
    end
end
fprintf('\n EncoderA: %d out of %d samples successfully SPEncoded!\n',count,nsamples);
toc;


tic; count=0;
for i=1:nsamples
    u=(rand(K,1)>0.5);
    [x,y]=systematic_pencode(u,'B');
    
    if( x(PCparams.FZlookup==-1)==u )
        uu=y(PCparams.FZlookup==-1);
        xx=pencode(uu);
        if(xx == x)
            count=count+1;
        end
    end
end
fprintf('\n EncoderB: %d out of %d samples successfully SPEncoded!\n',count,nsamples);
toc;

tic; count=0;
for i=1:nsamples
    u=(rand(K,1)>0.5);
    [x,y]=systematic_pencode(u,'C');
    
    if( x(PCparams.FZlookup==-1)==u )
        uu=y(PCparams.FZlookup==-1);
        xx=pencode(uu);
        if(xx == x)
            count=count+1;
        end
    end
end
fprintf('\n EncoderC: %d out of %d samples successfully SPEncoded!\n',count,nsamples);
toc;

end
