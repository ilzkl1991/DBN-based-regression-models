clc;
clear;


A=xlsread('BB.xlsx'); %The data for traininig and testing

inputdata=A(1:750,1:6);  %Training input data
outputdata=A(1:150,7); %Training output data

inputdata1=inputdata';
outputdata1=outputdata';

%ѡ����������������ݹ�һ��[0,1]
[inputdata11,inputps]=mapminmax(inputdata1,0,1);

[outputdata11,outputps]=mapminmax(outputdata1,0,1);

P=inputdata11';
T=outputdata11';

%  ex1 train a 100 hidden unit RBM and visualize its weights
numcases=80;
numdims=size(P,2);
numbatches=10;
% ѵ������
for i=1:numbatches
    train1=P((i-1)*numcases+1:i*numcases,:);
    batchdata(:,:,i)=train1;
end%���ֺõ�10�����ݶ�����batchdata��



%% 2.ѵ��RBM
%% rbm����
maxepoch=100;%ѵ��rbm�Ĵ���
numhid=30; numpen=30; numpen2=30; numpen3=30;%dbn������Ľڵ���
disp('����һ��4��������������DBN����������ȡ');
%% �޼ලԤѵ��
fprintf(1,'Pretraining Layer 1 with RBM: %d-%d ',numdims,numhid);
restart=1;
rbm1;%ʹ��cd-kѵ��rbm��ע���rbm�Ŀ��Ӳ㲻�Ƕ�ֵ�ģ����������Ƕ�ֵ��
vishid1=vishid;hidrecbiases=hidbiases;


fprintf(1,'\nPretraining Layer 2 with RBM: %d-%d ',numhid,numpen);
batchdata=batchposhidprobs;%����һ��RBM��������������Ϊ�ڶ���RBM ������
numhid=numpen;%��numpen��ֵ����numhid����Ϊ�ڶ���rbm������Ľڵ���
restart=1;
rbm1;
hidpen=vishid; penrecbiases=hidbiases; hidgenbiases=visbiases;

fprintf(1,'\nPretraining Layer 3 with RBM: %d-%d ',numpen,numpen2);%200-100
batchdata=batchposhidprobs;%��Ȼ�����ڶ���RBM�������Ϊ������RBM������
numhid=numpen2;%������������Ľڵ���
restart=1;
rbm1;
hidpen2=vishid; penrecbiases2=hidbiases; hidgenbiases2=visbiases;
% 
% fprintf(1,'\nPretraining Layer 4 with RBM: %d-%d\n ',numpen2,numpen3);%200-100
% batchdata=batchposhidprobs;%��Ȼ�����ڶ���RBM�������Ϊ������RBM������
% numhid=numpen3;%������������Ľڵ���
% restart=1;
% rbm1;
% hidpen3=vishid; penrecbiases3=hidbiases; hidgenbiases3=visbiases;


%%%% ��Ԥѵ���õ�RBM���ڳ�ʼ��DBNȨ��%%%%%%%%%
w1=[vishid1; hidrecbiases]; 
w2=[hidpen; penrecbiases]; 
w3=[hidpen2; penrecbiases2];
% w4=[hidpen3; penrecbiases3];

%% �мල�ع��ѵ��
%===========================ѵ������=====================================%
%==========DBN�޼ල������ȡ��������Ҫ�����мල�Ļع��==================%
%���ں���ƫִ������ʵ������Ӧ�ð���һ��ȫΪ1��������w0x0+w1x1+..+wnxn ����x0Ϊ1������ w0Ϊƫ��b
N1 = size(P,1);
digitdata = [P ones(N1,1)];

w1probs = 1./(1 + exp(-digitdata*w1));
w1probs = [w1probs  ones(N1,1)];

w2probs = 1./(1 + exp(-w1probs*w2));
w2probs = [w2probs  ones(N1,1)];

w3probs = 1./(1 + exp(-w2probs*w3));
H= w3probs'; %DBN�����  Ҳ��BP������
%% BP����ѵ��
inputn=H(1:30,1:150);
outputn=T';
% %��ʼ������ṹ
s1=40;%������ڵ�
disp('ѵ��bp������')
net=newff(inputn,outputn,s1);
net.trainParam.epochs=100;%ѵ������
net.trainParam.lr=1;%ѧϰ��
net.trainParam.goal=0.03;%ѧϰĿ��
net.trainParam.max_fail = 200;% ��Сȷ��ʧ�ܴ��� 
net.trainParam.showWindow = false; 
net.trainParam.showCommandLine = false; 
%��������
%����ѵ��
net=train(net,inputn,outputn);
% ѵ�����
disp('����ѵ��bp������')

%% ELMANѵ�����
% net=newelm(minmax(inputn),[40,1],{'tansig','tansig'});
% net.trainparam.show=100;%ÿ����100����ʾ1��
% net.trainparam.epochs=3000;%����������2000
% net.trainparam.goal=0.03;%����Ŀ��
% net=init(net);%��ʼ������
% %����ѵ��
% [net,tr]=train(net,inputn,outputn);

%% RBFѵ��
% switch 2
% case 1 
%          
% % ��Ԫ����ѵ���������� 
% spread = 1;                % ��ֵԽ��,���ǵĺ���ֵ�ʹ�(Ĭ��Ϊ1) 
% net = newrb(inputn,outputn);    
% % save BRPRBF net;
% case 2 
%      
% % ��Ԫ��������,������ѵ���������� 
% P = train_x2; 
% T = train_y2;  
% goal = 0.013;                % ѵ������ƽ����(Ĭ��Ϊ0) 
% spread =1;                % ��ֵԽ��,��Ҫ����Ԫ��Խ��(Ĭ��Ϊ1) 
% MN = size(P,2);% �����Ԫ��(Ĭ��Ϊѵ����������) 
% DF = 1;                     % ��ʾ���(Ĭ��Ϊ25) 
% net = newrb(inputn,outputn,goal,spread,MN,DF); 
%    
%     case 3     
% P = Ytrain1; 
% T = train_y1;  
% spread = 1;                % ��ֵԽ��,��Ҫ����Ԫ��Խ��(Ĭ��Ϊ1) 
% net = newgrnn(inputn,outputn,spread); 
%      
% end

input_test1=A(751:800,1:6);     %��������(������һ��)
output_test1=A(751:800,7);
input_test=input_test1';
output_test=output_test1';
test_x=mapminmax('apply',input_test,inputps,0,1)'; 

N1 = size(test_x,1);
digitdata1 = [test_x ones(N1,1)];

w1probs1 = 1./(1 + exp(-digitdata1*w1));
w1probs1 = [w1probs1  ones(N1,1)];

w2probs1 = 1./(1 + exp(-w1probs1*w2));
w2probs1 = [w2probs1  ones(N1,1)];

w3probs1 = 1./(1 + exp(-w2probs1*w3)); 
test_x1 =w3probs1';
% 
% w4probs1 = 1./(1 + exp(-w3probs1*w4)); 
% test_x1 = w4probs1';

an=sim(net,test_x1);
%�޸Ĳ��֣����Ԥ��ֵ

BPoutput=mapminmax('reverse',an,outputps);
error2=BPoutput-output_test;
% MSE1=sum((BPoutput-output_test).^2)/length(BPoutput);
figure(1)
plot(BPoutput,'r-*')
hold on
%title('ʵ��ֵ��Ԥ��ֵ���ͼ','fontsize',10,'fontangle','normal')
plot(output_test,'b.-')
legend('Ԥ��ֵ','ʵ��ֵ','Location','NorthEast');
xlabel('��������','fontsize',10)
ylabel('btp','fontsize',10)
grid on;
hold off
figure(2)
plot(error2)
title('���','fontsize',10,'fontangle','normal')


figure(3)
bf=error2./output_test;
plot(100*bf,'r.-')
%title('���ٷֱ�','fontsize',10,'fontangle','normal')
xlabel('��������','fontsize',10)
ylabel('��%��','fontweight','bold')
grid on;
%% �������

