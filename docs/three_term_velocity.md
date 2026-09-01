# Three-term DFM probability velocity

The implemented conditional path is

\[
p_t(z\mid z_0,z_1)=\kappa_1(t)\delta_{z_1}(z)+\kappa_2(t)U(z)+\kappa_3(t)\delta_{z_0}(z).
\]

The derivation follows Theorem 3, equations (21)–(23), of [Discrete Flow Matching](https://arxiv.org/abs/2407.15595). For mixture components `w_j` and weights `κ_j`, choose at every time

\[
\ell(t)=\arg\min_j \frac{\dot\kappa_j(t)}{\kappa_j(t)},\quad
b=\frac{\dot\kappa_\ell}{\kappa_\ell},\quad
a_j=\dot\kappa_j-\kappa_j b.
\]

Then `a_j ≥ 0`, `a_ell=0`, and the marginal probability velocity is

\[
u_t(a,z)=a_1\,p_{1|t}(a\mid z,x)+a_2\,U(a)+a_3\,p_{0|t}(a\mid z,x)+b\,\delta_z(a).
\]

Off-diagonal CTMC transition rates are therefore

\[
Q_t(z,a)=a_1p_{1|t}(a\mid z,x)+a_2/K+a_3p_{0|t}(a\mid z,x),\qquad a\ne z,
\]

and `Q_t(z,z)` is minus the row sum. The implementation must select `ell` dynamically: for `power_uniform_bump`, the minimizing component can change with time.

## Analytic source posterior

No second neural network is needed. Conditional on image `x`, `z0 ~ p0(.|x)` and `z1 ~ q(.|x)` are independent (the supervised dataset makes `q` a point mass). Let current state be `z`, `r(a)=p_{1|t}(a|z,x)`, and

\[
c_z=\kappa_2/K+\kappa_3p_0(z\mid x).
\]

Bayes' rule gives

\[
A^{-1}=\sum_{a\ne z}r(a)+r(z)\frac{c_z}{\kappa_1+c_z},\qquad p_t(z\mid x)=A c_z,
\]

\[
q(z\mid x)=A r(z)\frac{c_z}{\kappa_1+c_z}.
\]

With `d_z=κ2/K+κ1 q(z|x)`, the required source posterior is

\[
p_{0|t}(a\mid z,x)=
\frac{p_0(a\mid x)\,[d_z+\kappa_3\mathbf{1}(a=z)]}{p_t(z\mid x)}.
\]

This is implemented in `source_posterior_from_target_posterior` and used only when its Theorem-3 coefficient is nonzero.

When `uniform_strength=0`, `κ2=0`, `κ1=g`, `κ3=1-g`; the rates reduce exactly to

\[
Q_t(z,a)=\frac{\dot g}{1-g}p_{1|t}(a\mid z,x),\quad a\ne z,
\]

which is the two-term x-prediction velocity.

## Numerical verification

`tests/test_three_term_continuity.py` constructs non-degenerate categorical source and target distributions, evaluates the exact marginal `p_t`, builds the implemented rate matrix, and checks

\[
\dot p_t \approx p_tQ_t
\]

against both the analytic derivative and a central finite difference. It also verifies exact reduction to the power two-term rate when `uniform_strength=0`.

