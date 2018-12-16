import argparse

if __name__ == 'main':
    parser = argparse.ArgumentParser(description='Train CongressIdeology model')

    parser.add_argument('input_file', type=str)
    parser.add_argument('meta_info_file', type=str)
    parser.add_argument('model_type', type=str)
    parser.add_argument('param_grid_file', type=str)
    parser.add_argument('train_congress', type=int, nargs='*')
    parser.add_argument('test_congress', type=int, nargs='*')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--random_state', type=int, default=24)
