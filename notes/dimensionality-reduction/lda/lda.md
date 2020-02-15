### 1. 二分类

给定 $m$ 个 $d$ 维输入数据，用矩阵 $\mathbf{X}$ 表示如下:

$$\begin{align}
\mathbf{X} = 
\left[
\begin{matrix}
| & | & \cdots & | \\
\mathbf{x}^{[1]} & \mathbf{x}^{[2]} & \cdots & \mathbf{x}^{[m]} \\
| & | & \cdots & | \\
\end{matrix}
\right]_{d \times m}
\tag{1}
\end{align}
$$

其中， $\mathbf{x}^{[i]}=\left[\begin{matrix}\mathbf{x}_{1}^{[i]}&\mathbf{x}_{2}^{[i]}&\cdots&\mathbf{x}_{d}^{[i]}\end{matrix}\right]^{\rm{T}}$, $i=1,2,\cdots,m$。

给定这 $m$ 个样本相应的输出为:

$$\begin{align}
\mathbf{y}=
\left[
\begin{matrix}
y^{[1]} \\
y^{[2]} \\
\vdots \\
y^{[m]}
\end{matrix}
\right]_{m \times 1}
\end{align} \tag{2}$$

其中， $y^{[i]}\in{\{0,1\}}$，$i=1,2,\cdots,m$。


lda 的思想是，给定样本集合，设法将样本点投影到一条直线上，使得同类样本的投影点尽可能的相近，不同类样本的投影点尽可能的远离。  
在对新样本进行分类时，将其投影到同样的这条直线上，在根据投影点的位置来确定新的样本的类别。

投影向量 $\mathbf{\omega}=\left[\begin{matrix}\omega_{1} & \omega_{2} & \cdots & \omega_{d} \end{matrix}\right]^{\rm{T}}$
样本 $\mathbf{x}=\left[\begin{matrix}x_{1} & x_{2} & \cdots & x_{d}\end{matrix}\right]^{\rm{T}}$ 到投影直线为:

$$\hat{y}=\mathbf{\omega}^{\rm{T}}\mathbf{x}$$


#### 1.1 不同类样本之间投影点尽可能远离

> 不同类样本之间投影点尽可能远离，表现在让类中心之间的距离尽可能大。

假设每一类样本的集合为 $\Omega_{i}$，每一类样本点投影到直线上后，其类中心为:

$$\hat{\mu}_{i}=\frac{1}{N_{i}}\sum_{\mathbf{x}\in\Omega_{i}}{\mathbf{\omega}^{\rm{T}}\mathbf{x}}=\mathbf{\omega}^{\rm{T}}\frac{1}{N_{i}}\sum_{\mathbf{x}\in\Omega_{i}}{\mathbf{x}}=\mathbf{\omega}^{\rm{T}}\mathbf{\mu}_{i}$$

其中，$i$ 表示第 $i$ 类样本，$\hat{\mu}_{i}$ 表示第 $i$ 类样本投影到直线上的中心点，$\mathbf{\mu}_{i}$ 表示第 $i$ 类样本在原始空间上的中心点。$i=1,2$。

则，让类中心之间的距离尽可能大。表现在公式上为：$\max  || \mathbf{\omega}^{\rm{T}}\mathbf{\mu}_{1} - \mathbf{\omega}^{\rm{T}}\mathbf{\mu}_{2} ||^{2}$


#### 1.2 同类样本之间投影点尽可能相近

> 同类样本之间投影点尽可能相近，表现在让同类样本点投影点的协方差尽可能小。

如1.1小节所示，假设每一类样本的集合为 $\Omega_{i}$，每一类样本点投影到直线上后，其类中心为 $\hat{\mu}_{i}$，$i=1,2$

其每一类中，定义度量值**散列值（scatter）**如下:

$$\hat{s}_{i}^{2}=\sum_{\hat{y}\in\Omega_{i}}{(\hat{y}-\hat{\mu}_{i})^{2}}$$ 

$i=1,2$

上述公式的散列值的几何含义是样本点的密集程度，散列值越大，表示各个样本点之间越分散，散列值越小，表示各个样本点之间越密集。  
则，让同类样本之间的投影点尽可能相近，表现在公式上为: $\min \hat{s}_{1}+\hat{s}_{2}$


#### 1.3 综合

综合 1.1 和 1.2 小节的 lda 的思想，最后，lda 的优化函数是

$$J(\mathbf{\omega})=\frac{||\hat{\mu}_{1}-\hat{\mu}_{2}||^{2}}{\hat{s}_{1}+\hat{s}_{2}}$$

针对上式，我们做一个形式上的简化

首先，对散列公式进行展开

$$\begin{align} \hat{s}_{i}^{2}=\sum_{\hat{y}\in\Omega_{i}}{(\hat{y}-\hat{\mu}_{i})^{2}}=\sum_{\mathbf{x}\in\Omega_{i}}{(\mathbf{\omega}^{\rm{T}}\mathbf{x}-\mathbf{\omega}^{\rm{T}}\mu)^{2}}
=\sum_{\mathbf{x}\in\Omega_{i}}{\mathbf{\omega}^{\rm{T}}(\mathbf{x}-\mathbf{\mu})(\mathbf{x}-\mathbf{\mu})^{\rm{T}}\mathbf{\omega}}=\mathbf{\omega}^{\rm{T}}[\sum_{\mathbf{x}\in\Omega_{i}}{(\mathbf{x}-\mathbf{\mu})(\mathbf{x}-\mathbf{\mu})^{T}}]\mathbf{\omega}\end{align}$$

我们定义中间的部分如下所示:

$$\mathbf{S}_{w}=\sum_{\mathbf{x}\in\Omega_{i}}{(\mathbf{x}-\mathbf{\mu}_{i})(\mathbf{x}-\mathbf{\mu}_{i})^{T}}$$


这一部分我们称其为**类内离散度矩阵**。

另外，我们可以将优化公式分母展开：
$$||\hat{\mu}_{1}-\hat{\mu}_{2}||^{2}=(\mathbf{\omega}^{\rm{T}}\mathbf{\mu}_{1}-\mathbf{\omega}^{\rm{T}}\mathbf{\mu}_{2})(\mathbf{\omega}^{\rm{T}}\mathbf{\mu}_{1}-\mathbf{\omega}^{\rm{T}}\mathbf{\mu}_{2})=\mathbf{\omega}^{\rm{T}}(\mathbf{\mu}_{1}-\mathbf{\mu}_{2})(\mathbf{\mu}_{1}-\mathbf{\mu}_{2})^{\rm{T}}\mathbf{\omega}$$

$$\mathbf{S}_{b}=(\mathbf{\mu}_{1}-\mathbf{\mu}_{2})(\mathbf{\mu}_{1}-\mathbf{\mu}_{2})^{\rm{T}}$$

这一部分我们称其为**类间离散度矩阵**。

则，优化公式可以写成如下形式:

$$\max \arg J(\mathbf{\omege})=\frac{\mathbf{\omega}^{\rm{T}}\mathbf{S}_{b}\mathbf{\omega}}{\mathbf{\omega}^{\rm{T}}\mathbf{S}_{w}\mathbf{\omega}}$$


#### 1.4 求解优化目标函数

因为 $\mathbf{\omega}$ 只是表示的是一个方向（即向量），假设已经求解出 $\mathbf{\omega}$， 则 $2\mathbf{\omege}$，$3\mathbf{\omega}$ 等等均是成立的。为了“固定”住 $\mathbf{\omega}$ 的值，我们可以另上式的分母等于1。则，优化目标函数变成如下形式：

