import numpy as np
import scipy.stats as st
from scipy.signal import fftconvolve


def calculate_log_probability(X, F, B, s):
    """
    Calculates log p(X_k|d_k,F,B,s) for all images X_k in X and
    all possible displacements d_k.
    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    Returns
    -------
    ll : array, shape(H-h+1, W-w+1, K)
        ll[dh,dw,k] - log-likelihood of observing image X_k given
        that the villain's face F is located at displacement (dh, dw)
    """
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
    """
    Calculates the lower bound L(q,F,B,s,A) for the marginal log likelihood.
    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    q : array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.
    Returns
    -------
    L : float
        The lower bound L(q,F,B,s,A) for the marginal log likelihood.
    """
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
    """
    Given the current esitmate of the parameters, for each image Xk
    esitmates the probability p(d_k|X_k,F,B,s,A).
    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    F  : array_like, shape(h, w)
        Estimate of villain's face.
    B : array shape(H, W)
        Estimate of background.
    s : scalar, shape(1, 1)
        Eestimate of standard deviation of Gaussian noise.
    A : array, shape(H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    use_MAP : bool, optional,
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.
    Returns
    -------
    q : array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    """
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
    """
    Estimates F,B,s,A given esitmate of posteriors defined by q.
    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    q  :
        if use_MAP = False: array, shape (H-h+1, W-w+1, K)
           q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
           of villain's face given image Xk
        if use_MAP = True: array, shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    h : int
        Face mask height.
    w : int
        Face mask width.
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.
    Returns
    -------
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    """
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
    """
    Runs EM loop until the likelihood of observing X given current
    estimate of parameters is idempotent as defined by a fixed
    tolerance.
    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    F : array, shape (h, w), optional
        Initial estimate of villain's face.
    B : array, shape (H, W), optional
        Initial estimate of background.
    s : float, optional
        Initial estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1), optional
        Initial estimate of prior on displacement of face in any image.
    tolerance : float, optional
        Parameter for stopping criterion.
    max_iter  : int, optional
        Maximum number of iterations.
    use_MAP : bool, optional
        If true then after E-step we take only MAP estimates of displacement
        (dh,dw) of villain's face given image Xk.
    Returns
    -------
    F, B, s, A : trained parameters.
    LL : array, shape(number_of_iters + 2,)
        L(q,F,B,s,A) at initial guess, after each EM iteration and after
        final estimate of posteriors;
        number_of_iters is actual number of iterations that was done.
    """
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
    """
    Restarts EM several times from different random initializations
    and stores the best estimate of the parameters as measured by
    the L(q,F,B,s,A).
    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    tolerance, max_iter, use_MAP : optional parameters for EM.
    n_restarts : int
        Number of EM runs.
    Returns
    -------
    F : array,  shape (h, w)
        The best estimate of villain's face.
    B : array, shape (H, W)
        The best estimate of background.
    s : float
        The best estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        The best estimate of prior on displacement of face in any image.
    L : float
        The best L(q,F,B,s,A).
    """
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
