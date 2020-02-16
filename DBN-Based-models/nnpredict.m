
function output = nnpredict(nn, x)            %预测时不需要y值
    nn.testing = 1;
    nn = nnff(nn, x, zeros(size(x,1), nn.size(end)));     %nn.size(end)为输出节点的个数
    nn.testing = 0;
    n=nn.n;
    
    output=nn.a{n};

end
