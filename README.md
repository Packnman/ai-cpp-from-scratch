# ai-cpp-from-scratch

# Linear
$$
\boldsymbol{X}=\boldsymbol{U}^{(1)}
\\
\boldsymbol{Z}^{(i)} = f^{(i)}( \boldsymbol{U^{(i)}})
\\
U^{(i+1)} = \boldsymbol{W}^{(i)}\boldsymbol{Z}^{(i)} + \boldsymbol{b}^{(i)}
\\
\Delta^{(i)} = (\boldsymbol{W^{(i+1)T}} \Delta^{(i+1)}) \odot f^{(i)'}( \boldsymbol{U}^{(1)})
\\
\frac{\partial E_p}{\partial b} = \Delta^{(i)}\boldsymbol{v}^T
$$

# ReLU
$$
max(0,x)
$$

# GeLU
順伝播
$$
x\Phi(x)
\\
\Phi(x) = \frac{1}{2}
\left(
1+erf\left(\frac{x}{\sqrt{2}}\right)
\right)
$$
\
近似式は
$$
\frac{1}{2}x
\left[
1+tanh
\left(
\sqrt{\frac{2}{\pi}}(x+0.0447115x^3)
\right)
\right]
$$
整理すると
$$
\\
y=\frac{1}{2}x(1+tanh(u))
\\
u=c(x+0.044715x^3)
$$
\
逆伝播
$$
\left(
x\Phi(x)
\right)'
=
\Phi(x) + x\phi(x)
\\
=\frac{1}{2}
\left(
1+erf\left(\frac{x}{\sqrt{2}}\right)
\right)
+
\frac{x}{\sqrt{2\pi}}e^{-x^2/2}
$$
近似式は、
$$
\frac{dy}{dx}=\frac{1}{2}x(1+tanh(u))+\frac{1}{2}x(1-tanh^2(u))c(1+3・0.044715x^2)
$$

# Softmax/CrossEntropy
順伝播
\
| 記号 | 意味 |
| -- | :-- |
| $B$ | バッチ数 |
| $L$ | 誤差関数 |
$$
p_{ib}  = \frac{e^{z_{ib}}}{\sum_je^{z_{jb}}}
\\
L=-\frac{1}{B}\sum_b\sum_it_{ib}\log{p_{ib}}
\\
$$
\
逆伝播
$$
\frac{\partial L}{\partial z}=\frac{p-t}{B}
$$


# Adam
更新手順
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
\\
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
\\
\hat{m}_t =\frac{m_t}{1-\beta_1^t}
\\
\hat{v}_t =\frac{v_t}{1-\beta_2^t}
\\
W_t = W_{t-1} - \eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$