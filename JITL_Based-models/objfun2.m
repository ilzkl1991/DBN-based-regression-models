function BPout=object2(x)
%global net; 
 load('outputps.mat')
 load('inputps.mat')
 
% y=mapminmax('apply',x,inputps);
load ('-mat','data2BP');

%����Ԥ�����
an=sim(net,x);

%�����������һ��
BPout=mapminmax('reverse',an',outputps);
 end
