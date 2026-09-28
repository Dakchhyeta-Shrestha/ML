"""Four-class car-price model, adapted from the course multinomial notebook."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

FEATURES = ['max_power', 'mileage', 'year', 'brand']


def prepare_data(path, test_size=0.30, seed=42):
    df = pd.read_csv(path)
    df = df[(df.owner != 'Test Drive Car') & (~df.fuel.isin(['CNG', 'LPG']))].copy()
    df['brand'] = df.name.str.split().str[0]
    for col in ['mileage', 'max_power']:
        df[col] = pd.to_numeric(df[col].astype(str).str.split().str[0], errors='coerce')
    df = df.dropna(subset=['selling_price', 'year', 'brand'])
    # Quantiles provide roughly equal support across four classes. Fit boundaries
    # on the training prices only, so the held-out set cannot influence them.
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=test_size,
                                            random_state=seed)
    train, test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
    edges = np.quantile(train.selling_price, [0.25, 0.5, 0.75])
    edges = np.unique(edges)
    if len(edges) != 3:
        raise ValueError('Need three distinct price cutoffs for four classes')
    boundaries = np.r_[-np.inf, edges, np.inf]
    brands = sorted(train.brand.unique())
    brand_ids = {name: i for i, name in enumerate(brands)}
    fallback_brand = train.brand.mode().iloc[0]
    medians = {col: float(train[col].median()) for col in ['max_power', 'mileage', 'year']}

    def transform(part):
        x = part[FEATURES].copy()
        x['brand'] = x.brand.where(x.brand.isin(brand_ids), fallback_brand).map(brand_ids)
        for col, value in medians.items():
            x[col] = x[col].fillna(value)
        y = pd.cut(part.selling_price, boundaries, labels=False).to_numpy(dtype=int)
        return x.astype(float), y

    x_train, y_train = transform(train)
    x_test, y_test = transform(test)
    scaler = MinMaxScaler().fit(x_train)
    x_train = np.c_[np.ones(len(x_train)), scaler.transform(x_train)]
    x_test = np.c_[np.ones(len(x_test)), scaler.transform(x_test)]
    preprocessing = dict(edges=edges.tolist(), brands=brands, brand_ids=brand_ids,
                         fallback_brand=fallback_brand, medians=medians, scaler=scaler)
    return x_train, x_test, y_train, y_test, preprocessing


class LogisticRegression:
    """Course softmax regression with an optional ridge penalty on non-bias weights."""
    def __init__(self, k=4, n=None, method='batch', alpha=0.3, max_iter=1200,
                 ridge=False, lam=0.0, seed=42):
        self.k, self.n, self.method = k, n, method
        self.alpha, self.max_iter, self.ridge, self.lam, self.seed = (
            alpha, max_iter, ridge, lam, seed)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).reshape(-1)
        if X.ndim != 2 or len(X) != len(y) or len(y) == 0:
            raise ValueError('Expected X=(rows, features) and y=(rows,)')
        if not np.all(np.isin(y, np.arange(self.k))):
            raise ValueError('Class labels must be integers from 0 to k-1')
        if self.method not in ('batch', 'minibatch', 'sto'):
            raise ValueError('method must be batch, minibatch or sto')
        self.n = X.shape[1]
        self.W = np.zeros((self.n, self.k))
        self.losses = []
        rng = np.random.default_rng(self.seed)
        onehot = np.eye(self.k)[y]
        for _ in range(self.max_iter):
            if self.method == 'batch':
                ix = np.arange(len(X))
            elif self.method == 'minibatch':
                ix = rng.choice(len(X), min(128, len(X)), replace=False)
            else:
                ix = rng.choice(len(X), 1)
            loss, grad = self.gradient(X[ix], onehot[ix])
            self.W -= self.alpha * grad
            self.losses.append(loss)
        return self

    def softmax(self, scores):
        scores = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict_proba(self, X):
        return self.softmax(np.asarray(X) @ self.W)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def gradient(self, X, Y):
        m = len(X)
        probabilities = self.predict_proba(X)
        loss = -np.sum(Y * np.log(np.clip(probabilities, 1e-15, 1))) / m
        grad = X.T @ (probabilities - Y) / m
        if self.ridge:
            # J = mean cross entropy + lambda * sum(W[1:] ** 2).
            # Bias row is excluded, matching standard ridge practice.
            loss += self.lam * np.sum(self.W[1:] ** 2)
            grad[1:] += 2 * self.lam * self.W[1:]
        return loss, grad

    @staticmethod
    def accuracy(y_true, y_pred):
        return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))

    @staticmethod
    def per_class(y_true, y_pred, k=4):
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        result = {}
        for c in range(k):
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            support = int(np.sum(y_true == c))
            p = float(tp / (tp + fp)) if tp + fp else 0.0
            r = float(tp / (tp + fn)) if tp + fn else 0.0
            f = 2 * p * r / (p + r) if p + r else 0.0
            result[c] = dict(precision=p, recall=r, f1=f, support=support)
        return result

    @classmethod
    def report(cls, y_true, y_pred, k=4):
        rows = cls.per_class(y_true, y_pred, k)
        total = len(y_true)
        result = dict(accuracy=cls.accuracy(y_true, y_pred), per_class=rows)
        for metric in ('precision', 'recall', 'f1'):
            result['macro_' + metric] = sum(rows[c][metric] for c in rows) / k
            result['weighted_' + metric] = (
                sum(rows[c][metric] * rows[c]['support'] for c in rows) / total
                if total else 0.0)
        return result
