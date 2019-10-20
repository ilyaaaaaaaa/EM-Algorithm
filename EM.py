import numpy as np
import scipy.stats as st
from scipy.signal import fftconvolve


def calculate_log_probability(X, F, B, s):
    H, W, K = X.shape
    h, w = F.shape
    ll = np.zeros((H - h + 1, W - w + 1, K))

    for i in range(H - h + 1):
        for j in range(W - w + 1):
            mu = np.copy(B)
            mu[i:i + h, j:j + w] = np.copy(F)
            ll[i, j] += np.sum((X - mu.reshape(H, W, 1))**2, axis=(0, 1))
    ll /= -2 * s**2
    ll += -(np.log(s) + 0.5 * np.log(2 * np.pi)) * H * W

    return ll


def calculate_lower_bound(X, F, B, s, A, q, use_MAP=False):
    H, W, K = X.shape
    h, w = F.shape
    c = 1e-10
    if use_MAP == False:
        L = np.sum(
            (calculate_log_probability(X, F, B, s) + np.log(A + c).reshape(
                H - h + 1, W - w + 1, 1) - np.log(q + c)) * q)
        return L
    if use_MAP == True:
        ll = calculate_log_probability(X, F, B, s)
        logA = np.log(A + c)
        L = 0
        for k, (i, j) in enumerate(q.T):
            L += ll[i, j, k] + logA[i, j]
        return L


def run_e_step(X, F, B, s, A, use_MAP=False):
    H, W, K = X.shape
    h, w = F.shape
    c = 1e-10

    ll_A = calculate_log_probability(X, F, B, s) + np.log(A + c).reshape(
        H - h + 1, W - w + 1, 1)
    q = np.exp(ll_A - (np.max(ll_A, axis=(0, 1))).reshape(1, 1, K))
    q /= np.sum(q, axis=(0, 1))
    if use_MAP == False:
        return q
    if use_MAP == True:
        qs = []
        for k in range(K):
            q_ = q[:, :, k]
            i_ = np.unravel_index(np.argmax(q_), q_.shape)
            qs.append(np.array([i_[0], i_[1]]))
        qs = np.array(qs).T
        return qs


def run_m_step(X, q, h, w, use_MAP=False):
    H, W, K = X.shape
    if use_MAP == False:
        A = np.mean(q, axis=2)

        F = np.zeros((h, w))
        B = np.zeros((H, W))
        B_ = np.zeros((H, W))
        for i in range(H - h + 1):
            for j in range(W - w + 1):
                F_B = np.sum(q[i, j].reshape(1, 1, K) * X, axis=2)
                F += F_B[i:i + h, j:j + w]
                F_B[i:i + h, j:j + w] = 0
                B += F_B
                B__ = np.sum(q[i, j]) * np.ones((H, W))
                B__[i:i + h, j:j + w] = 0
                B_ += B__
        F /= K
        B_[B_ == 0] = 1
        B /= B_

        s = 0
        for i in range(H - h + 1):
            for j in range(W - w + 1):
                mu = np.copy(B)
                mu[i:i + h, j:j + w] = np.copy(F)
                s += np.sum(q[i, j].reshape(1, 1, K) *
                            (X - mu.reshape(H, W, 1))**2)
        s /= H * W * K
        s = s**0.5

        return F, B, s, A

    if use_MAP == True:
        A = np.zeros((H - h + 1, W - w + 1))
        for i, j in q.T:
            A[i, j] += 1
        A /= K

        F = np.zeros((h, w))
        B = np.zeros((H, W))
        B_ = np.zeros((H, W))
        for k, (i, j) in enumerate(q.T):
            F += X[i:i + h, j:j + w, k]
            B__ = np.ones((H, W))
            B__[i:i + h, j:j + w] = 0
            B += X[:, :, k] * B__
            B_ += B__
        F /= K
        B_[B_ == 0] = 1
        B /= B_

        s = 0
        for k, (i, j) in enumerate(q.T):
            mu = np.copy(B)
            mu[i:i + h, j:j + w] = np.copy(F)
            s += np.sum((X[:, :, k] - mu)**2)
        s /= H * W * K
        s = s**0.5
        return F, B, s, A


def run_EM(X,
           h,
           w,
           F=None,
           B=None,
           s=None,
           A=None,
           tolerance=0.001,
           max_iter=50,
           use_MAP=False):
    H, W, K = X.shape
    if F is None: F = np.random.randint(0, 255, (h, w))
    if B is None: B = np.random.randint(0, 255, (H, W))
    if s is None: s = np.random.uniform(1e-2, 10)
    if A is None:
        A = np.random.uniform(1e-100, 1, (H - h + 1, W - w + 1))
        A /= np.sum(A)
    LL = []
    for i in range(max_iter):
        q = run_e_step(X, F, B, s, A, use_MAP)
        F, B, s, A = run_m_step(X, q, h, w, use_MAP)
        LL.append(calculate_lower_bound(X, F, B, s, A, q, use_MAP))
        if i > 0 and np.abs(LL[i] - LL[i - 1]) < tolerance: break
    LL = np.array(LL)

    return F, B, s, A, LL


def run_EM_with_restarts(X,
                         h,
                         w,
                         tolerance=0.001,
                         max_iter=50,
                         use_MAP=False,
                         n_restarts=10):
    F_, B_, s_, A_, LL_ = [], [], [], [], []
    for i in range(n_restarts):
        F, B, s, A, LL = run_EM(
            X,
            h,
            w,
            F=None,
            B=None,
            s=None,
            A=None,
            tolerance=tolerance,
            max_iter=max_iter,
            use_MAP=use_MAP)
        F_.append(F)
        B_.append(B)
        s_.append(s)
        A_.append(A)
        LL_.append(LL[-1])
    best = np.argmax(LL_)

    return F_[best], B_[best], s_[best], A_[best], LL_[best]


def run_fast_m_step(X, q, h, w, use_MAP=False):
    H, W, K = X.shape
    if use_MAP == False:
        A = np.mean(q, axis=2)

        F = np.zeros((h, w))
        B = np.zeros((H, W))
        B_ = np.zeros((H, W))
        for i in range(H - h + 1):
            for j in range(W - w + 1):
                F_B = np.sum(q[i, j].reshape(1, 1, K) * X, axis=2)
                F += F_B[i:i + h, j:j + w]
                F_B[i:i + h, j:j + w] = 0
                B += F_B
                B__ = np.sum(q[i, j]) * np.ones((H, W))
                B__[i:i + h, j:j + w] = 0
                B_ += B__
        F /= K
        B_[B_ == 0] = 1
        B /= B_

        s = 1

        return F, B, s, A

    if use_MAP == True:
        A = np.zeros((H - h + 1, W - w + 1))
        for i, j in q.T:
            A[i, j] += 1
        A /= K

        F = np.zeros((h, w))
        B = np.zeros((H, W))
        B_ = np.zeros((H, W))
        for k, (i, j) in enumerate(q.T):
            F += X[i:i + h, j:j + w, k]
            B__ = np.ones((H, W))
            B__[i:i + h, j:j + w] = 0
            B += X[:, :, k] * B__
            B_ += B__
        F /= K
        B_[B_ == 0] = 1
        B /= B_

        s = 1
        return F, B, s, A


def run_fast_EM(X,
                h,
                w,
                F=None,
                B=None,
                s=None,
                A=None,
                tolerance=0.001,
                max_iter=50,
                use_MAP=False):
    H, W, K = X.shape
    if F is None: F = np.random.randint(0, 255, (h, w))
    if B is None: B = np.random.randint(0, 255, (H, W))
    if s is None: s = np.random.uniform(1e-2, 10)
    if A is None:
        A = np.random.uniform(1e-100, 1, (H - h + 1, W - w + 1))
        A /= np.sum(A)
    LL = []
    for i in range(max_iter):
        q = run_e_step(X, F, B, s, A, use_MAP)
        F, B, s, A = run_fast_m_step(X, q, h, w, use_MAP)
        LL.append(calculate_lower_bound(X, F, B, s, A, q, use_MAP))
        if i > 0 and np.abs(LL[i] - LL[i - 1]) < tolerance: break
    LL = np.array(LL)

    return F, B, s, A, LL


def run_fast_EM_with_restarts(X,
                              h,
                              w,
                              tolerance=0.001,
                              max_iter=50,
                              use_MAP=False,
                              n_restarts=10):
    F_, B_, s_, A_, LL_ = [], [], [], [], []
    for i in range(n_restarts):
        F, B, s, A, LL = run_fast_EM(
            X,
            h,
            w,
            F=None,
            B=None,
            s=None,
            A=None,
            tolerance=tolerance,
            max_iter=max_iter,
            use_MAP=use_MAP)
        F_.append(F)
        B_.append(B)
        s_.append(s)
        A_.append(A)
        LL_.append(LL[-1])
    best = np.argmax(LL_)

    return F_[best], B_[best], s_[best], A_[best], LL_[best]
