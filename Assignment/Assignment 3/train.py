"""Train, evaluate and optionally publish A3 runs to course MLflow."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from model import LogisticRegression, prepare_data

try:
    from mlflow.pyfunc import PythonModel
except ImportError:
    PythonModel = object  # Local training and app do not require MLflow.


def run(data_path='Cars.csv', output='app/model.pkl', tracking_uri=None):
    x_train, x_test, y_train, y_test, preprocessing = prepare_data(data_path)
    print('Training classes:', np.bincount(y_train, minlength=4))
    print('Test classes:', np.bincount(y_test, minlength=4))
    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train, y_train, test_size=0.20, random_state=42, stratify=y_train)
    results = []
    mlflow = None
    if tracking_uri:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment('126671-a3')
    for ridge, lam in [(False, 0.0), (True, 0.001), (True, 0.01), (True, 0.1)]:
        model = LogisticRegression(method='batch', alpha=0.3, max_iter=1200,
                                   ridge=ridge, lam=lam).fit(x_fit, y_fit)
        predictions = model.predict(x_val)
        report = model.report(y_val, predictions)
        record = {'ridge': ridge, 'lambda': lam, 'val_accuracy': report['accuracy'],
                  'val_macro_f1': report['macro_f1'], 'val_weighted_f1': report['weighted_f1']}
        results.append((record, model))
        print(record)
        if mlflow:
            with mlflow.start_run(run_name=f"ridge-{lam}" if ridge else 'baseline'):
                mlflow.log_params({'ridge': ridge, 'lambda': lam, 'alpha': 0.3,
                                   'max_iter': 1200, 'split_seed': 42})
                mlflow.log_metrics({k: v for k, v in record.items() if k.startswith('val_')})
                # Save the full prediction bundle; no dataset is logged.
                mlflow.pyfunc.log_model(name='model', python_model=CarPricePyFunc(model, preprocessing),
                                        code_paths=['model.py', 'train.py'],
                                        pip_requirements=['numpy', 'pandas', 'scikit-learn', 'mlflow'])
    best_record, _ = max(results, key=lambda item: item[0]['val_macro_f1'])
    best_model = LogisticRegression(method='batch', alpha=0.3, max_iter=1200,
                                    ridge=best_record['ridge'], lam=best_record['lambda']).fit(x_train, y_train)
    bundle = {'model': best_model, 'preprocessing': preprocessing}
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('wb') as stream:
        pickle.dump(bundle, stream)
    Path('experiment_results.json').write_text(json.dumps([r for r, _ in results], indent=2))
    print('Best:', best_record)
    final_report = best_model.report(y_test, best_model.predict(x_test))
    print('Final held-out manual report:', final_report)
    print('Scikit-learn report:\n', classification_report(y_test, best_model.predict(x_test),
                                                            labels=[0, 1, 2, 3], zero_division=0))
    if mlflow:
        with mlflow.start_run(run_name='selected-final-model'):
            mlflow.log_params({'ridge': best_record['ridge'], 'lambda': best_record['lambda'],
                               'alpha': 0.3, 'max_iter': 1200, 'split_seed': 42})
            mlflow.log_metrics({key: final_report[key] for key in
                                ('accuracy', 'macro_f1', 'weighted_f1')})
            mlflow.pyfunc.log_model(name='model', python_model=CarPricePyFunc(best_model, preprocessing),
                                    code_paths=['model.py', 'train.py'],
                                    pip_requirements=['numpy', 'pandas', 'scikit-learn', 'mlflow'])
    return bundle


def make_features(frame, preprocessing):
    """Apply the exact training mappings to an app input DataFrame."""
    from model import FEATURES
    data = frame[FEATURES].copy()
    data['brand'] = data.brand.where(data.brand.isin(preprocessing['brand_ids']),
                                     preprocessing['fallback_brand']).map(preprocessing['brand_ids'])
    for column, median in preprocessing['medians'].items():
        data[column] = data[column].fillna(median)
    scaled = preprocessing['scaler'].transform(data.astype(float))
    return np.c_[np.ones(len(scaled)), scaled]


class CarPricePyFunc(PythonModel):
    # mlflow's PythonModel interface only needs predict; avoids loading mlflow locally.
    def __init__(self, model, preprocessing):
        self.model, self.preprocessing = model, preprocessing

    def predict(self, context, model_input):
        return self.model.predict(make_features(model_input, self.preprocessing))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='Cars.csv')
    parser.add_argument('--tracking-uri', default=None)
    args = parser.parse_args()
    run(data_path=args.data, tracking_uri=args.tracking_uri)
