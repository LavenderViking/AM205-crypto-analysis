import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
np.set_printoptions(precision=3, suppress=True)
pd.set_option('display.precision', 3)

FILES = [('ripple_price', 'rip'), ('bitcoin_price', 'btc'),
         ('ethereum_price', 'eth'), ('litecoin_price', 'ltc')]


#####################################################################
def load_returns_matrix (center=True, scale=True):
    """Prepare rolling monthly return matrix."""
    dfs = []
    for file, name in FILES:
        path = 'cryptocurrencypricehistory//{}.csv'.format(file)
        df = pd.read_csv(path, usecols=['Date', 'Close'])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[['Date', 'Close']]
        df.set_index('Date', drop=True, inplace=True)
        df.rename(columns={'Close':name}, inplace=True)
        dfs.append(df)
    out = pd.concat(dfs, axis=1, join='inner')
    out = out.pct_change(periods=1, freq=pd.Timedelta(days=30))
    out.dropna(axis=0, how='any', inplace=True)
    out = preprocessing.scale(out, axis=0, with_mean=center, with_std=scale)
    return np.matrix(out)


def create_proportion_of_variation_df (eigvals):
    total_var = sum(eigvals)
    cols = ['component', 'eigenvalue', 'proportion', 'cumulative']
    df = pd.DataFrame(columns=cols)
    cum = 0
    for i, e in enumerate(eigvals):
        cum += e
        row = {'component':i + 1,
               'eigenvalue':e,
               'proportion':e/total_var,
               'cumulative':cum/total_var}
        df = df.append(row, ignore_index=True)
    df['component'] = df['component'].astype(int)
    df.set_index('component', drop=True, inplace=True)
    return df


def create_eigvec_df (V):
    """Create DataFrame of eigenvectors from V made in SVD."""
    m, _ = V.shape
    eigvecs = [np.ravel(V[:, i]) for i in range(m)]  # extract columns
    df = pd.DataFrame()
    for i, v in enumerate(eigvecs):
        df['V{}'.format(i + 1)] = v
    return df


#####################################################################

X = load_returns_matrix(center=True, scale=True)
n, p = X.shape
C = np.cov(X, rowvar=False)  # covariance matrix

# Get SVD breakdown, cast into matrices.
U, s, V = np.linalg.svd(X)
U = np.matrix(U)
S = np.zeros((n, p))
S[:p, :p] = np.diag(s)
S = np.matrix(S)
V = np.matrix(V).T

print('Shapes of SVD components:')
print('U: {}'.format(U.shape))
print('S: {}'.format(S.shape))
print('V: {}\n'.format(V.shape))

# Get eigenvalues through the singular values.
eigvals = [(s_**2)/(n - 1) for s_ in s]

# Reconstruct data matrix X and covariance matrix C using SVD properties.
# Check equality by looking at the norm of their difference.
Xreconstr = U*S*V.T
print('Norm[Xreconstr - X] = {:.3f}\n'.format(np.linalg.norm(Xreconstr - X)))

Creconstr = V*((S.T*S)/(n - 1))*V.T
print('Norm[Creconstr - C] = {:.3f}\n'.format(np.linalg.norm(Creconstr - C)))

# Get principal components (n-by-p matrix).
XV = X*V

# Inner products b/w eigenvectors and principal components should be 0.
pc1, pc2 = np.ravel(XV[:, 0]), np.ravel(XV[:, 1])
v1, v2 = np.ravel(V[:, 0]), np.ravel(V[:, 1])
print('Inner product between 1st and 2nd eigenvectors: {:.4f}\n'.format(
      np.inner(v1, v2)))
print('Inner product between 1st and 2nd PCs: {:.4f}\n'.format(
      np.inner(pc1, pc2)))

# Create tables for proportion of variation and eigenvectors.
df_variation = create_proportion_of_variation_df(eigvals)
df_eigvec = create_eigvec_df(V)
print('Eigenvectors:\n{}\n\nProportion of Variation:\n{}\n\n'.format(
      df_eigvec, df_variation))
