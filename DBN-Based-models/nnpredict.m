
function output = nnpredict(nn, x)            %Ԥ��ʱ����Ҫyֵ
    nn.testing = 1;
    nn = nnff(nn, x, zeros(size(x,1), nn.size(end)));     %nn.size(end)Ϊ����ڵ�ĸ���
    nn.testing = 0;
    n=nn.n;
    
    output=nn.a{n};

end
