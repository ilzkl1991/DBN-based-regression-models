function BPout=rbf2(x)
%global net; 
 load('outputps.mat')
 load('inputps.mat')
 
% y=mapminmax('apply',x,inputps);
load ('-mat','RBF2');

%����Ԥ�����
an=sim(net,x);

%�����������һ��
BPout=mapminmax('reverse',an',outputps);
 end

