 function BPOUTE=Elman1(x)
load ('-mat','elman1');
 load('outputps.mat')
 load('inputps.mat')
%����Ԥ�����
an=sim(net,x);
BPOUTE=mapminmax('reverse',an',outputps);
 end


