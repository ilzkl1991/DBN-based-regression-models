function BPout=rbf1(x)
%global net; 
 load('outputps.mat')
 load('inputps.mat')
 
% y=mapminmax('apply',x,inputps);
load ('-mat','RBF1');

%����Ԥ�����
an=sim(net,x);

%�����������һ��
BPout=mapminmax('reverse',an',outputps);
 end


