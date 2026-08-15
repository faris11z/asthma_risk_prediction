import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Retrain + evaluate + visualize the PEFR model.')
    parser.add_argument('--fast', action='store_true',
                        help='Skip full hyperparameter grid (use small grid for quick runs).')
    args = parser.parse_args()

    here = os.path.dirname(__file__)
    model_path = os.path.join(here, 'PEFR_predictor.joblib')

    if os.path.exists(model_path):
        os.remove(model_path)
        print('Removed cached model (PEFR_predictor.joblib) — retraining fresh.')

    from model_evaluation import run_full_evaluation
    from model_visualization import plot_all
    from model_training import save_best_model

    results = run_full_evaluation(verbose=True, fast=args.fast)

    save_best_model(results['best_model'], params=results['best_params'])

    y_test = results['y_test']
    y_pred = results['y_pred']
    plot_all(results['best_model'], y_test, y_pred)

    print('\nEvaluation complete.')

if __name__ == '__main__':
    main()
