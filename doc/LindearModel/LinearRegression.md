model:
$$
\bm{y}=W^T\bm{\phi(x)}
$$
$$
\hat{\bm{y}}=W^T\bm{\phi(x)} \\
L(W)=\frac{1}{2}\sum\limits_{n}||\hat{\bm{y_n}}-\bm{y_n}||_2^2
$$
Where
$$
W=\begin{pmatrix}
\bm{w_1},\bm{w_2},..,\bm{w_n}
\end{pmatrix}
$$
Let
$$
\partial_\bm{w_k}L(W^)=
\sum\limits_{n} \bm{\phi(x_n)}(\bm{w_k^T\phi(x_n)}-y_{nk})=0 \\
\Phi=\begin{pmatrix}
\bm{\phi^T(x_1)}\\
\bm{\phi^T(x_2)}\\
.\\
.\\
\bm{\phi^T(x_n)}\\
\end{pmatrix}\\
\bm{y_k}=\begin{pmatrix}
y_{1k}\\
y_{2k},\\
.\\
.\\
y_{nk}
\end{pmatrix}\\
Y=\begin{pmatrix}
\bm{y_1},\bm{y_2},..,\bm{y_K}    
\end{pmatrix}
$$
Then
$$
\bm{w_k}=(\Phi^T\Phi)^{-1}\Phi \bm{y_k}\\
W=(\Phi^T\Phi)^{-1}\Phi Y
$$
Also wo can use GradientDown and add regular term:
$$
L_1=\frac{1}{2}||w||_1\\
L_2=\frac{1}{2}||w||_2^2\\
\hat{L}(W)=L(w)+\lambda\Omega(w)\\
\min\limits_{w}\hat{L}(w)
$$