 function BPOUTE=Elman2(x)
load ('-mat','elman2');
 load('outputps.mat')
 load('inputps.mat')
%����Ԥ�����
an=sim(net,x);
BPOUTE=mapminmax('reverse',an',outputps);
 end


