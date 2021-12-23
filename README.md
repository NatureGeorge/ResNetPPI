# ResNetPPI

## 模型总体设计 (Model Design)

-   MSA Encoder
    -   对输入宿主序列(长度为L)的含有K个同源序列的MSA，转为K个pairwise比对；未忽略同源序列的insertion区域，每个比对的长度可能不一，用$L_{k}$表示
    -   对宿主序列中可能有的20种氨基酸+gap+未知种类氨基酸(20+1+1)共22种情况进行onehot编码，同源序列亦是如此；这样一来，每个位置便有长度为44的onehot编码张量
    -   通过1D-ResNet来接收这K个$44\times L_{k}$张量，其用以捕获每个位点自身的突变信息以及临近位点间的关系，输出K个$64\times L_{k}$embedding张量
    -   由于上一步是依次将每个$44\times L_{k}$张量进行输入，ResNet每接受到一个新的同源序列就相当于更新卷积层中的权重
    -   对输入病毒序列(长度为$L^{'}$)，也对每个位置构建出长度为44的onehot编码张量；其中后22个元素固定为0值，相当于mask，而后输入1D-ResNet得到$64\times L^{'}$embedding张量
-   Aggregation Module
    -   Property Aggregator
        -   首先进行pooling，去掉K个$64\times L_{k}$张量中的insertion相关列，构成统一的$K\times64\times L$张量
        -   对于某一同源序列(k)构成的$64\times L$张量，其可表示为$x_{k}\in {R}^{64\times L}$，其中位置序号为i的氨基酸表示为$x_{k}(i)$
        -   利用max函数对每个位置取K个同源序列中元素的最大值，即$A(i,c)=\max \{ x_{k}(i,c),\, 0 \leq k < K \}$，c为channel，得到$64\times L$的aggregated张量A，以此来利用整合后的精炼MSA张量，方便进行后续每个氨基酸位点性质方面的学习
    -   Co-evolution Aggregator
        -   这里开始对上一部分得到的$x_{k}$，进行基于进化与共进化的整合
        -   首先用原始MSA计算每个同源序列的权重$w_k$，其为该同源序列在MSA所有同源序列中有80%及以上sequence
            identity的序列的数目的倒数
        -   $M_{\text{eff}}=\sum_{k}^{K}w_{k}$
        -   计算one body term:
            $f(i) = \frac{1}{M_{\text{eff}}}\sum_{k}^{K}w_{k}x_{k}(i)$
        -   计算two body term:
            $g(i,j)=\frac{1}{M_{\text{eff}}}\sum_{k}^{K}w_{k}[x_{k}(i)\otimes x_{k}(j)],\,g(i,j)\in R^{64\times 64}$
        -   整合one body term与two body term:
            $h(i,j)=\text{CONCAT}(f(i),f(j),g(i,j)),\, h(i,j)\in R^{4224}$
            (64+64+64\*64=4224)
        -   将h输入下面的Intra-chain Distance Estimator
    -   Paired-evolution Aggregator
        -   对上一部分得到的$A$张量与病毒序列(长度为$L^{'}$)经过embedding构成的$64\times L^{'}$B张量进行整合，计算two
            body term:
            $s(i,i') = [A(i)\otimes B(i')],\, s(i,i') \in R^{64\times64}$
        -   将$s(i,i')$以及Intra-chain Distance
            Estimator中的2D-ResNet输出共同输入下面的Inter-chain Distance
            Estimator
-   Distance Estimator
    -   Intra-chain Distance Estimator
        -   通过2D-ResNet来接收Co-evolution
            Aggregator得来的$h(i,j)$，将其转为$4224\times L \times L$的对称张量
        -   2D-ResNet输入共进化方面的信息来估计残基间的距离，将4224维度降下来，最终输出$96\times L \times L$的张量
        -   再通过一层2d卷积层来将维度降为$37\times L \times L$；在这里，37对应的是宿主蛋白三维结构依据$[2,2.5), [2.5, 3), ..., [19.5, 20), [20, +\infty)$的距离阈值进行离散化后的预测目标的channel数
        -   由于链内残基间距离是个对称矩阵，而上面输出不一定是对称的，因而进行显式的对称化操作$(M+M^{T})/2$
        -   而后将结果输入SoftMax函数进行多分类的概率预测
        -   利用CrossEntropyLoss来计算这部分的预测值与真实值之间的差异
    -   Inter-chain Distance Estimator (2D-ResNet)
