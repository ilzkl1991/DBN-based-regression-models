 function BPOUTE=Elman1(x)
load ('-mat','elman1');
 load('outputps.mat')
 load('inputps.mat')
%ÍøÂçÔ¤²âÊä³ö
an=sim(net,x);
BPOUTE=mapminmax('reverse',an',outputps);
 end


