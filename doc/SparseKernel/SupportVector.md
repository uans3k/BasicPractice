主要的思想是数据到判定分界超平面的距离最大:
$$
L(\bm{w},b,\bm{x_n})=\frac{1}{||\bm{w}||_2}y_n(\bm{w^T\phi(x_n)}+b)\\
\max\limits_{\bm{w},b} \ \min\limits_{n}L(\bm{w},b,\bm{x_n})
$$
事实上
$$
\bm{w} \rightarrow k\bm{w}\\
b \rightarrow kb
$$
$L(\bm{w},b,\bm{x_n})$不变
因此引入限制使得
$$\min_{n}(\bm{w^T\phi(x_n)}+b)=\pm 1$$
问题化为了
$$
\min\limits_{w,b} \frac{1}{2}||\bm{w}||_2^2\\
s.t. \ y_n(\bm{w^T\phi(x_n)}+b) \ge 1
$$
做拉格朗日乘数
$$
\hat{L}(\bm{w},b,\lambda)=\frac{1}{2}||\bm{w}||_2^2- \sum\limits_{n}\lambda_n(y_n(\bm{w^T\phi(x_n)}+b)-1)
$$
优化目标
$$
\min\limits_{\bm{w},b}\max\limits_{\lambda} \hat{L}(\bm{w},b,\lambda)
$$
KKT条件
$$
\lambda_n \ge 0 \\
y_n(\bm{w^T\phi(x_n)}+b)-1 \ge 0 \\
\lambda_n(y_n(\bm{w^T\phi(x_n)}+b)-1)=0
$$
使
$$
\partial_{\bm{w}}\hat{L}=\bm{w}-\sum\limits_n \lambda_n y_n\bm{\phi(x_n)}=0\\
\partial_{b}\hat{L}=\sum\limits_n \lambda_ny_n=0
$$
得到
$$
\bm{w}=\sum\limits_n \lambda_n y_n\bm{\phi(x_n)}\\
\sum\limits_n \lambda_ny_n=0
$$
$\lambda_n$在由于KKT条件,那些不在$(\bm{w^T\phi(x_n)}+b)=\pm 1$平面上的的会取0，因此只用到$\lambda_n\ne 0$的数据，因此称为支持向量机\
上式代入$L(\bm{w},b,\bm{\lambda})$
$$
\hat{L}(\bm{\lambda})
=\sum\limits_n\lambda_n
-\frac{1}{2}\sum\limits_{m,n}\lambda_my_m(\bm{\phi^T(x_m) \phi(x_n)})\lambda_ny_n
\\
\max\limits_{\bm{\lambda}}\hat{L}(\bm{\lambda}) \\
s.t. \sum\limits_n \lambda_ny_n=0\\
\lambda_n \ge 0
$$
记
$$
k(\bm{x_m,x_n})=\bm{\phi^T(x_n) \phi(x_m)}\\
K=[k(\bm{x_m,x_n})]\\
t_n=\lambda_ny_n\\
\bm{t}=\begin{pmatrix}
    t_1\\
    t_2\\
    .\\
    .\\
    t_N
\end{pmatrix}
$$
因此简记为
$$
\hat{L}(\bm{\lambda})=\sum\limits_n\lambda_n-\frac{1}{2}\bm{t^T}Kt\\
\max\limits_{\bm{\lambda}}\hat{L}(\bm{\lambda})\\
s.t. \sum\limits_n \lambda_ny_n=0\\
\lambda_n \ge 0
$$
相应的预测为
$$
y(\bm{x})=\sum\limits_n \lambda_ny_nk(\bm{x_n,x})+b=\bm{t}^TK_{predict}+b
$$
KKT条件为
$$
\lambda_n \ge 0 \\
y_ny(x_n)-1 \ge 0 \\
\lambda_n (y_ny(x_n)-1)=0
$$

记$\lambda_n \ne 0$的$\bm{x_n}$的集合为S,由KKT条件的满足
$$
y_n(\sum\limits_{x_m \in S}\lambda_my_mk(\bm{x_m,x_n})+b)=1
$$
为了系统更健壮,可取遍y_n,则
$$
b=\frac{1}{|S|}\sum\limits_{x_n \in S}(y_n-\sum\limits_{x_m \in S}\lambda_my_mk(\bm{x_m,x_n}))
$$
上述的形式要求分界面能完美分割所有样本,实践上基本不可能。因此需要弱化限制条件:
$$
\ y_n(\bm{w^T\phi(x_n)}+b) \ge 1-\xi_n \\
\xi_n>0
$$
同时又不能让$\xi_n$ 过分大导致分类其他样本分类错误,加上惩罚项
$$
C \sum\limits_n \xi_n
$$
问题化为
$$
L(\bm{w,\lambda,\xi,\mu})=\frac{1}{2}||\bm{w}||_2^2+C\sum\limits_n\xi_n-\sum\limits_{n}\lambda_n(y_n(\bm{w^T\phi(x_n)}+b)-1)
-\sum\limits_n \mu_n\xi_n\\
\min_{\lambda,\xi} \max\limits_{\lambda,\mu}L(\bm{w,\lambda,\xi,\mu})
$$
同样的得到对偶问题
$$
\hat{L}(\bm{\lambda})=\sum\limits_n\lambda_n-\frac{1}{2}\bm{t^T}Kt\\
\max\limits_{\bm{\lambda}} -\hat{L}(\bm{\lambda})\\
s.t. \sum\limits_n \lambda_ny_n=0\\
0 \le \lambda_n  \le C
$$
记$\lambda_n \ne 0$的$\bm{x_n}$集合S，$0 < \lambda_n  < C$的集合H,.根据KKT条件：
$$
b=\frac{1}{|H|}\sum\limits_{x_n \in H} (y_n-\sum\limits_{x_m \in S}\lambda_my_mk(x_m,x_n))
$$
SVM问题求解通常用SMO算法.