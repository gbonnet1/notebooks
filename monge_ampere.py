# %% [markdown]
"""
# Monge-Ampère equations
"""


# %%
# !python3 -c 'import agd' 2> /dev/null || pip install agd


# %%
from dataclasses import dataclass

import agd.LinearParallel as lp
import matplotlib.pyplot as plt
import numpy as np
from agd import Domain, Selling
from agd.AutomaticDifferentiation.Optimization import newton_root

# %% [markdown]
"""
In this notebook, we aim to solve Monge-Ampère equations of the form
\begin{equation}
    \det (D^2 u(x) - A(x, u(x), D u(x))) = B(x, u(x), D u(x))
\end{equation}
on an open domain $\Omega \subset \mathbb{R}^2$, where functions $A$ and $B$ take values in $\mathcal{S}_2$ and $\mathbb{R}_+$ respectively. We look for an *admissible* solution to the previous equation, that is, a solution $u \colon \overline \Omega \to \mathbb{R}$ such that
\begin{equation}
    D^2 u(x) - A(x, u(x), D u(x)) \geq 0.
\end{equation}
"""


# %% [markdown]
"""
## 1. Reformulation of the equation
"""


# %% [markdown]
"""
Let $M \in \mathcal{S}_2^{++}$, $\mathcal{D} \in \mathcal{S}_2^{++}$ be such that $\det\mathcal{D} = 1$, and $\lambda_1$, $\lambda_2 \geq 0$ be the eigenvalues of the product $\mathcal{D} M$. Then by the inequality of arithmetic and geometric means,
\begin{equation}
    2 (\det M)^{1/2}
    = 2 (\det (\mathcal{D} M))^{1/2}
    = 2 (\lambda_1 \lambda_2)^{1/2}
    \leq \lambda_1 + \lambda_2
    = \langle \mathcal{D}, M \rangle,
\end{equation}
with equality when $\mathcal{D} = (\det M)^{1/2} M^{-1}$. Thus
\begin{equation}
    2 (\det M)^{1/2}
    = \inf_{\substack{\mathcal{D} \in \mathcal{S}_2^{++} \\ \det \mathcal{D} = 1}} \langle \mathcal{D}, M \rangle
    = \inf_{\substack{\mathcal{D} \in \mathcal{S}_2^{++} \\ \operatorname{Tr}(\mathcal{D}) = 1}} (\det \mathcal{D})^{-1/2} \langle \mathcal{D}, M \rangle,
\end{equation}
and this equality extends to $M \in \mathcal{S}_2^+$. Therefore admissible solutions to the Monge-Ampère equation are also solution to
\begin{equation}
    \sup_{\substack{\mathcal{D} \in \mathcal{S}_2^{++} \\ \operatorname{Tr}(\mathcal{D}) = 1}} \left(2 B(x, u(x), D u(x))^{1/2} - (\det \mathcal{D})^{-1/2} \langle \mathcal{D}, D^2 u(x) - A(x, u(x), D u(x)) \rangle\right) = 0.
\end{equation}
Multiplying by $(\det \mathcal{D})^{1/2}$ inside the supremum yields the reformulation
\begin{equation}
    \max_{\substack{\mathcal{D} \in \mathcal{S}_2^+ \\ \operatorname{Tr}(\mathcal{D}) = 1}} \left(2 (\det \mathcal{D})^{1/2} B(x, u(x), D u(x))^{1/2} - \langle \mathcal{D}, D^2 u(x) - A(x, u(x), D u(x)) \rangle\right) = 0.
\end{equation}
We propose to discretize the last equation, which has the benefit that its solutions are always admissible for the Monge-Ampère equation.
"""


# %% [markdown]
"""
## 2. Discretization
"""


# %% [markdown]
"""
We discretize the reformulated equation on a grid $\mathcal{G}_h := \Omega \cap h \mathbb{Z}^2$, for some discretization step $h > 0$.

Let $u \colon \mathcal{G}_h \to \mathbb{R}$ and $x \in \mathcal{G}_h$. If it makes sense (we discuss handling of boundary conditions in later sections), we define
\begin{equation}
    D_h u(x) := \frac{1}{2 h} \begin{pmatrix}
        u(x + h e_1) - u(x - h e_1) \\
        u(x + h e_2) - u(x - h e_2)
    \end{pmatrix},
\end{equation}
where $(e_1, e_2)$ is the canoncial basis of $\mathbb{R}^2$. We also define, for $e \in \mathbb{Z}^2$ and $A \in \mathcal{S}_2$,
\begin{equation}
    \Delta_h^e [u, A](x) := \frac{u(x + h e) + u(x - h e) - 2 u(x)}{h^2} - \langle e, A e \rangle,
\end{equation}
and for any family $v = (v_i)_{1 \leq i \leq m}$ of vectors of $\mathbb{Z}^2$,
\begin{equation}
    \Delta_h^v [u, A](x) := (\Delta_h^{v_i} [u, A](x))_{1 \leq i \leq m}.
\end{equation}
For any such family $v$, we define $\mathcal{S}^+(v) \subset \mathcal{S}_2^+$ as
\begin{equation}
    \mathcal{S}^+(v) := \sum_{i=1}^m \mathbb{R}_+ v_i v_i^\top.
\end{equation}

The *stencil* of the numerical scheme is described by three sets $V_1 \subset \mathbb{Z}^2$, $V_2 \subset (\mathbb{Z}^2)^2$, and $V_3 \subset (\mathbb{Z}^2)^3$ such that

* Any family $v \in V_3$ is a superbase of $\mathbb{Z}^2$, meaning that $\det(v_1, v_2) = \pm 1$ and $v_1 + v_2 + v_3 = 0$. Note that this also implies that $\det(v_2, v_3) = \det(v_3, v_1) = \pm 1$.
* Any family $v \in V_2$ is a basis of $\mathbb{Z}^2$, meaning that $\det(v_1, v_2) = \pm 1$, and moreover $\bigcup_{v \in V_2} \{\pm v_1, \pm v_2\} = \bigcup_{v \in v_3} \bigcup_{1 \leq i < j \leq 3} \{\pm v_i, \pm v_j\}$.
* Any vector $v \in V_1$ is irreductible, meaning that its components are coprime, and moreover $\bigcup_{v \in V_1} \{\pm v\} = \bigcup_{v \in V_2} (\{\pm v_1\} \cup \{\pm v_2\})$.

For any superbase $v \in V_3$ and matrix $\mathcal{D} \in \mathcal{S}_2$, the coordinates of $\mathcal{D}$ in the basis $(v_1 v_1^\top, v_2 v_2^\top, v_3 v_3^\top)$ of $\mathcal{S}_2$ are given by *Selling's formula*:
\begin{equation}
    \mathcal{D} = -\sum_{i=1}^3 \langle v_{i+1}^\perp, \mathcal{D} v_{i+2}^\perp \rangle v_i v_i^\top,
\end{equation}
where indices are taken modulo three. Note that $\mathcal{D} \in \mathcal{S}^+(v)$ if and only if $\langle v_i^\perp, \mathcal{D} v_j^\perp \rangle \leq 0$, for any $1 \leq i < j \leq 3$.

We use the following numerical scheme:
\begin{equation}
    \max_{v \in V_3} H(v, B(x, u(x), D_h u(x)), \Delta_h^v [u, A(x, u(x), D_h u(x))](x)) = 0,
\end{equation}
where for any $v \in V_3$, $b \geq 0$, and $\delta \in \mathbb{R}^3$,
\begin{equation}
    H(v, b, \delta) := \max_{\substack{\mathcal{D} \in \mathcal{S}^+(v) \\ \operatorname{Tr}(\mathcal{D}) = 1}} \left(2 (\det \mathcal{D})^{1/2} b^{1/2} + \sum_{i=1}^3 \langle v_{i+1}^\perp, \mathcal{D} v_{i+2}^\perp \rangle \delta_i\right).
\end{equation}
"""


# %% [markdown]
"""
### 2.1. Computing the maximum
"""


# %% [markdown]
"""
The maximum in the definition of $H$ may be computed explicitly: for any $\rho \in \overline B(0, 1)$, the closed unit ball of $\mathbb{R}^2$, let
\begin{equation}
    \mathfrak{D}(\rho) := \frac{1}{2} \begin{pmatrix}
        1 + \rho_0 & \rho_1 \\
        \rho_1 & 1 - \rho_0
    \end{pmatrix}.
\end{equation}
Then $\{\mathcal{D} \in \mathcal{S}_2^+ \mid \operatorname{Tr}(\mathcal{D}) = 1\} = \{\mathfrak{D}(\rho) \mid \rho \in \overline B(0, 1)\}$. Moreover, for $\rho \in \overline B(0, 1)$, $2 \det (\mathfrak{D}(\rho))^{1/2} = (1 - |\rho|^2)^{1/2}$, and for $v \in V_3$,
\begin{equation}
    (-\langle v_{i+1}^\perp, \mathfrak{D}(\rho) v_{i+2}^\perp \rangle)_{1 \leq i \leq 3} = W \rho + w
\end{equation}
where
\begin{equation}
    W := \frac{1}{2} \begin{pmatrix}
        v_{2,1} v_{3,1} - v_{2,2} v_{3,2} & v_{2,1} v_{3,2} + v_{2,2} v_{3,1} \\
        v_{1,1} v_{3,1} - v_{1,2} v_{3,2} & v_{1,1} v_{3,2} + v_{1,2} v_{3,1} \\
        v_{1,1} v_{2,1} - v_{1,2} v_{2,2} & v_{1,1} v_{2,2} + v_{1,2} v_{2,1}
    \end{pmatrix},
\end{equation}
\begin{equation}
    w := -\frac{1}{2} \begin{pmatrix}
        \langle v_2, v_3 \rangle \\
        \langle v_1, v_3 \rangle \\
        \langle v_1, v_2 \rangle
    \end{pmatrix}.
\end{equation}
Remark that the expression of the matrix $Q := W W^\top$ simplifies to
\begin{equation}
    Q = \frac{1}{4} \begin{pmatrix}
        |v_2|^2 |v_3|^2 & \langle v_1, v_2 \rangle |v_3|^2 & \langle v_1, v_3 \rangle |v_2|^2 \\
        \langle v_1, v_2 \rangle |v_3|^2 & |v_1|^2 |v_3|^2 & \langle v_2, v_3 \rangle |v_1|^2 \\
        \langle v_1, v_3 \rangle |v_2|^2 & \langle v_2, v_3 \rangle |v_1|^2 & |v_1|^2 |v_2|^2
    \end{pmatrix}.
\end{equation}
The definition of $H$ may be rewritten as
\begin{equation}
    H(v, b, \delta) = \max_{\substack{\rho \in \overline B(0, 1) \\ W \rho + w \geq 0}} \left(b^{1/2} (1 - |\rho|^2)^{1/2} - \langle W \rho + w, \delta \rangle\right),
\end{equation}
where the inequality $W \rho + w \geq 0$ must hold elementwise.

If the optimal $\rho$ is such that $W \rho + w > 0$, then by convexity,
\begin{equation}
    H(v, b, \delta) = \max_{\rho \in \overline B(0, 1)} \left(b^{1/2} (1 - |\rho|^2)^{1/2} - \langle W \rho + w, \delta \rangle\right) = (b + \langle \delta, Q \delta \rangle)^{1/2} - \langle w, \delta \rangle,
\end{equation}
and the maximum is attained at point
\begin{equation}
    \rho_*(v, b, \delta) := -\frac{W^\top \delta}{(b + \langle \delta, Q \delta \rangle)^{1/2}}.
\end{equation}

Note that the inequality $W \rho_*(v, b, \delta) \geq 0$ is equivalent to $Q \delta \leq (b + \langle \delta, Q \delta \rangle)^{1/2} w$. To handle cases where the constraint $W \rho + w \geq 0$ is saturated in the definition of $H$, we rewrite the numerical scheme as
\begin{equation}
    \max_{i \in \{2, 3\}} \max_{v \in V_i} H_i(v, B(x, u(x), D_h u(x)), \Delta_h^v [u, A(x, u(x), D_h u(x))](x)) = 0,
\end{equation}
where
\begin{equation}
    H_3(v, b, \delta) := \begin{cases}
        (b + \langle \delta, Q \delta \rangle)^{1/2} - \langle w, \delta \rangle &\text{if } Q \delta \leq (b + \langle \delta, Q \delta \rangle)^{1/2} w, \\
        -\infty &\text{else},
    \end{cases}
\end{equation}
and
\begin{equation}
    H_2(v, b, \delta) := \max_{t \in [-1, 1]} \left(2 \left(\det \left(\frac{1+t}{2} \frac{v_1 v_1^\top}{|v_1|^2} + \frac{1-t}{2} \frac{v_2 v_2^\top}{|v_2|^2}\right)\right)^{1/2} b^{1/2} - \frac{1+t}{2 |v_1|^2} \delta_1 - \frac{1-t}{2 |v_2|^2} \delta_2\right).
\end{equation}

The maximum in the definition of $H_2$ may also be computed exactly: for any $v \in V_2$ and $t \in [-1, 1]$, we compute that
\begin{equation}
    2 \left(\det \left(\frac{1+t}{2} \frac{v_1 v_1^\top}{|v_1|^2} + \frac{1-t}{2} \frac{v_2 v_2^\top}{|v_2|^2}\right)\right)^{1/2}
    = \frac{|\det(v_1, v_2)|}{|v_1| |v_2|} (1 - t^2)^{1/2}
    = \frac{1}{|v_1| |v_2|} (1 - t^2)^{1/2}.
\end{equation}
We deduce that
\begin{equation}
    H_2(v, b, \delta) = \left(\frac{b}{|v_1|^2 |v_2|^2} + \left(\frac{\delta_1}{2 |v_1|^2} - \frac{\delta_2}{2 |v_2|^2}\right)^2\right)^{1/2} - \left(\frac{\delta_1}{2 |v_1|^2} + \frac{\delta_2}{2 |v_2|^2}\right),
\end{equation}
which we write as
\begin{equation}
    H_2(v, b, \delta) = (\omega_0 b + \langle \omega_1, \delta \rangle^2)^{1/2} - \langle \omega_2, \delta \rangle
\end{equation}
where
\begin{equation}
    \omega_0 := \frac{1}{|v_1|^2 |v_2|^2},
\end{equation}
\begin{equation}
    \omega_1 := \frac{1}{2} \begin{pmatrix}
        1 / |v_1|^2 \\
        -1 / |v_2|^2
    \end{pmatrix},
\end{equation}
\begin{equation}
    \omega_3 := \frac{1}{2} \begin{pmatrix}
        1 / |v_1|^2 \\
        1 / |v_2|^2
    \end{pmatrix}.
\end{equation}
"""


# %% [markdown]
"""
3. Implementation of the scheme
"""


# %%
x = np.stack(np.meshgrid(*(2 * [np.linspace(-1, 1, 40)]), indexing="ij"))
domain_ball = Domain.Ball(radius=0.9)


# %%
@dataclass
class Stencil:
    V1: np.ndarray
    V2: np.ndarray
    V2_indices: np.ndarray
    V3: np.ndarray
    V3_indices: np.ndarray
    Q: np.ndarray
    w: np.ndarray
    omega0: np.ndarray
    omega1: np.ndarray
    omega2: np.ndarray


# %%
def StencilForConditioning(cond):
    V3 = Selling.SuperbasesForConditioning(cond)
    offsets = V3.reshape((2, -1))

    # Make offsets positive for the lexicographic order, inversing their sign if needed.
    offsets[:, offsets[0] < 0] *= -1
    offsets[:, np.logical_and(offsets[0] == 0, offsets[1] < 0)] *= -1

    V1, indices = np.unique(offsets, axis=1, return_inverse=True)
    V3_indices = indices.reshape(V3.shape[1:])
    V2_indices = np.unique(
        np.sort(
            np.concatenate(
                (V3_indices[[0, 1]], V3_indices[[0, 2]], V3_indices[[1, 2]]), axis=1
            ),
            axis=0,
        ),
        axis=1,
    )
    V2 = V1[:, V2_indices]

    Q = np.zeros((3, 3, V3.shape[2]))
    w = np.zeros((3, V3.shape[2]))

    for i in range(3):
        Q[i, i] = (
            lp.dot_VV(V3[:, (i + 1) % 3], V3[:, (i + 1) % 3])
            * lp.dot_VV(V3[:, (i + 2) % 3], V3[:, (i + 2) % 3])
            / 4
        )
        Q[i, (i + 1) % 3] = (
            lp.dot_VV(V3[:, i], V3[:, (i + 1) % 3])
            * lp.dot_VV(V3[:, (i + 2) % 3], V3[:, (i + 2) % 3])
            / 4
        )
        Q[i, (i + 2) % 3] = (
            lp.dot_VV(V3[:, i], V3[:, (i + 2) % 3])
            * lp.dot_VV(V3[:, (i + 1) % 3], V3[:, (i + 1) % 3])
            / 4
        )
        w[i] = -lp.dot_VV(V3[:, (i + 1) % 3], V3[:, (i + 2) % 3]) / 2

    omega0 = 1 / (lp.dot_VV(V2[:, 0], V2[:, 0]) * lp.dot_VV(V2[:, 1], V2[:, 1]))
    omega1 = 1 / (
        2 * np.stack([lp.dot_VV(V2[:, 0], V2[:, 0]), -lp.dot_VV(V2[:, 1], V2[:, 1])])
    )
    omega2 = 1 / (
        2 * np.stack([lp.dot_VV(V2[:, 0], V2[:, 0]), lp.dot_VV(V2[:, 1], V2[:, 1])])
    )

    return Stencil(V1, V2, V2_indices, V3, V3_indices, Q, w, omega0, omega1, omega2)


# %%
stencil = StencilForConditioning(15)


# %%
def H3(b, delta, stencil):
    Q = np.expand_dims(stencil.Q, (3, 4))
    w = np.expand_dims(stencil.w, (2, 3))
    delta = delta[stencil.V3_indices]

    Q_delta = lp.dot_AV(Q, delta)
    r = np.sqrt(b + lp.dot_VV(delta, Q_delta))
    return np.where(np.all(Q_delta <= r * w, axis=0), r - lp.dot_VV(w, delta), -np.inf)


# %%
def H2(b, delta, stencil):
    omega0 = np.expand_dims(stencil.omega0, (1, 2))
    omega1 = np.expand_dims(stencil.omega1, (2, 3))
    omega2 = np.expand_dims(stencil.omega2, (2, 3))
    delta = delta[stencil.V2_indices]

    return np.sqrt(omega0 * b + lp.dot_VV(omega1, delta) ** 2) - lp.dot_VV(
        omega2, delta
    )


# %%
def H1(delta, stencil):
    return -delta / np.expand_dims(lp.dot_VV(stencil.V1, stencil.V1), (1, 2))


# %%
def Scheme(a, b, d2u, stencil):
    delta = d2u - lp.dot_VAV(
        np.expand_dims(stencil.V1, (2, 3)),
        np.expand_dims(a, 2),
        np.expand_dims(stencil.V1, (2, 3)),
    )

    b_zero = b == 0

    return np.where(
        b_zero,
        np.max(H1(delta, stencil), axis=0),
        np.maximum(
            np.max(H2(np.where(b_zero, 1, b), delta, stencil), axis=0),
            np.max(H3(np.where(b_zero, 1, b), delta, stencil), axis=0),
        ),
    )


# %% [markdown]
"""
4. Dirichlet boundary conditions
"""


# %%
def SchemeDirichlet(u, x, domain, A, B, g, stencil):
    bc = Domain.Dirichlet(domain, g, x)

    du = bc.DiffCentered(u, [[1, 0], [0, 1]])
    d2u = bc.Diff2(u, stencil.V1)

    return np.where(
        bc.interior, Scheme(A(x, u, du), B(x, u, du), d2u, stencil), u - bc.grid_values
    )


# %%
# TODO: bug in agd freeway?


def A_zero(x, r, p):
    return np.zeros((2, 2) + x.shape[1:])


def B_quartic(x, r, p):
    return 48 * lp.dot_VV(x, x) ** 2


u = newton_root(
    SchemeDirichlet,
    np.zeros(x.shape[1:]),
    (x, domain_ball, A_zero, B_quartic, 0.9 ** 4, stencil),
)
u = np.where(domain_ball.level(x) < 0, u, np.nan)

plt.contourf(*x, u)
plt.show()


def ExactQuartic(x):
    return lp.dot_VV(x, x) ** 2


error = np.where(domain_ball.level(x) < 0, u - ExactQuartic(x), 0)
print("Error:", np.max(np.abs(error)))


# %% [markdown]
"""
## 4. Optimal transport boundary conditions
"""


# %%
def SchemeBV2(u, x, domain, A, B, C, sigma, stencil):
    bc = Domain.Dirichlet(Domain.Box([[-1, 1], [-1, 1]]), np.inf, x)

    du0 = bc.DiffUpwind(u, [[1, 0], [0, 1]])
    du1 = bc.DiffUpwind(u, [[-1, 0], [0, -1]])
    assert np.sum(np.logical_and(du0 == np.inf, du1 == np.inf)) == 0
    du0 = np.where(du0 == np.inf, -du1, du0)
    du1 = np.where(du1 == np.inf, -du0, du1)
    du = (du0 - du1) / 2

    du0 = bc.DiffUpwind(u, stencil.V1)
    du0 = np.where(
        du0 == np.inf,
        sigma(
            np.expand_dims(x, 1),
            np.expand_dims(u, 0),
            np.multiply.outer(stencil.V1, np.ones(x.shape[1:])),
        ),
        du0,
    )
    du1 = bc.DiffUpwind(u, -stencil.V1)
    du1 = np.where(
        du1 == np.inf,
        sigma(
            np.expand_dims(x, 1),
            np.expand_dims(u, 0),
            np.multiply.outer(-stencil.V1, np.ones(x.shape[1:])),
        ),
        du1,
    )
    d2u = (du0 + du1) / bc.gridscale

    return np.where(
        bc.interior,
        Scheme(A(x, u, du), np.where(domain.level(x) < 0, B(x, u, du), 0), d2u, stencil)
        + u.flatten()[np.argmin(domain.level(x))]
        - C,
        u - bc.grid_values,
    )


# %%
def f(x):
    return np.full(x.shape[1:], 1 / 0.9 ** 2)


def A_reflector(x, r, p):
    tmp = 1 + np.sqrt(1 - lp.dot_VV(p, p) / r ** 4)
    return (2 + tmp) / r * lp.outer(p, p) - r ** 3 * tmp * lp.identity(x.shape[1:])


def B_reflector(x, r, p):
    tmp = 1 + np.sqrt(1 - lp.dot_VV(p, p) / r ** 4)
    return r ** 6 * (tmp ** 3 - tmp ** 2) * f(x)


def sigma_reflector(x, r, e):
    return 2 * r ** 3 * (np.sqrt(lp.dot_VV(e, e)) - lp.dot_VV(x, e))


u = newton_root(
    SchemeBV2,
    np.full(x.shape[1:], 0.1),
    (x, domain_ball, A_reflector, B_reflector, 0.1, sigma_reflector, stencil),
)
u = np.where(domain_ball.level(x) < 0, u, np.nan)

plt.contourf(*x, u)
plt.show()


# %%
