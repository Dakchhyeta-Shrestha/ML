import numpy as np

from model import LogisticRegression


def test_model_accepts_expected_input():
    # Four feature columns plus bias; class labels are integers 0 through 3.
    X = np.array([[1, 0.1, 0.2, 0.3, 0.4], [1, 0.4, 0.3, 0.2, 0.1],
                  [1, 0.8, 0.1, 0.3, 0.5], [1, 0.2, 0.7, 0.1, 0.3]])
    y = np.array([0, 1, 2, 3])
    model = LogisticRegression(max_iter=5, ridge=True, lam=0.01).fit(X, y)
    assert np.isfinite(model.W).all()


def test_prediction_output_shape():
    X = np.c_[np.ones(8), np.arange(32).reshape(8, 4) / 32]
    y = np.tile(np.arange(4), 2)
    model = LogisticRegression(max_iter=5).fit(X, y)
    assert model.predict(X).shape == (8,)
    assert model.predict_proba(X).shape == (8, 4)
