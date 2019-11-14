model:
$$
\bm{o}=W^T\bm{\phi(x)}
$$
$$
\bm{p(x)}=\bm{softmax(o)}
$$
$$
softmax_i(\bm{o})=\frac{exp(o_i)}{\sum\limits_j exp(o_j)}
$$
Then use multi-Bernoulli distribution
$$
p(x_n)=\prod\limits_ip_i(\bm{x_n})^{y_{ni}}
$$
Minum -log-likehood
$$
L(W)=-lnp(X)=-\sum\limits_{n,i}y_{ni}p_i(\bm{x_n})=-\sum\limits_n \bm{y_n}^Tln\bm{p(x_n)}\\
\min\limits_w L(W)
$$
Surely,we can use regulation to choose model