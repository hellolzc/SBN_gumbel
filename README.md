# Training stochastic binary networks (SBN) using gumbel-sigmoid and gumbel-softmax

thing is largely a work in progress

[Bernoulli SBN](https://github.com/hellolzc/SBN_gumbel/tree/master/demo_gumbel_sigmoid.ipynb)
[Categorical SBN](https://github.com/hellolzc/SBN_gumbel/tree/master/demo_gumbel_sigmoid.ipynb)


# Main idea:

Explaination and motivation: https://arxiv.org/abs/1611.01144

There has recently been a trick that allows train networks with quasi-discrete categorical activations via gumbel-softmax or gumbel-sigmoid nonlinearity. A great explaination of how it works can be found [here](http://blog.evjang.com/2016/11/tutorial-categorical-variational.html).

The trick is to add a special noize to the softmax distribution that favors almost-1-hot outcomes. Such noize can be obtained from gumbel distribution. Since sigmoid can be viewed as a special case of softmax of 2 classes(logit and 0), we can use the same technique to implement an LSTM network with gates that will ultimately be forced to converge to 0 or 1. Here's a [demo](https://github.com/yandexdataschool/gumbel_lstm/blob/master/demo_gumbel_sigmoid.ipynb) of gumbel-sigmoid on a toy task.

Such network can then be binarized: multiplication can be replaced with if/else operations and fp16 operations to drastically improve execution speed, especially when implemented in a special-purpose device, [see here](https://www.engadget.com/2016/04/28/movidius-fathom-neural-compute-stick/ ) and [here](https://arxiv.org/abs/1602.02830).



# Contributors so far
- Lambda Lab
- Arseniy Ashukha (advice & useful comments)
- hellolzc

# Environments
- Python 3
- PyTorch
- numpy
- scikit-learn