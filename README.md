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
