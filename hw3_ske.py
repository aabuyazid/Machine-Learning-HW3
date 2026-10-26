import numpy as np
import pandas as pd
from sklearn.svm import SVC

'''
Problem: University Admission Classification using SVMs

Instructions:
1. Do not use any additional libraries. Your code will be tested in a pre-built environment with only 
   the library specified in question instruction available. Importing additional libraries will result in 
   compilation errors and you will lose marks.

2. Fill in the skeleton code precisely as provided. You may define additional 
   default arguments or helper functions if necessary, but ensure the input/output format matches.
'''

kernels = ['linear', 'rbf', 'poly']

class DataLoader:
    '''
    Put your call to class methods in the __init__ method. Autograder will call your __init__ method only. 
    '''
    
    def __init__(self, data_path: str, random_state=42, split=0.8):
        """
        Initialize data processor with paths to train dataset. You need to have train and validation sets processed.
        
        Args:
            data_path: absolute path to your data file
        """
        self.train_data = pd.DataFrame()
        self.val_data = pd.DataFrame()
        self.data_path = data_path
        self.random_state = random_state
        self.split = split
        self.threshold = None
        self.mean_std_pairs = {}
        
        # TODO：complete your dataloader here!
        self.data = pd.read_csv(data_path)

        n_train = int(self.split * len(self.data))
        self.train_data = self.data.sample(n=n_train, random_state=self.random_state)
        self.val_data = self.data.drop(self.train_data.index.to_list())

        self.train_data = self.create_binary_label(self.train_data)
        self.val_data = self.create_binary_label(self.val_data)

        self.train_data = self.preprocess_data(self.train_data)
        self.val_data = self.preprocess_data(self.val_data)
    
    def create_binary_label(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Create a binary label for the training data.
        '''
        if self.threshold is None:
            self.threshold = df['Chance of Admit'].median()

        ret_val = df.copy()
        binary_label = (df['Chance of Admit'] > self.threshold).astype(int)
        ret_val['label'] = binary_label

        return ret_val
    
    def normalize_series(self, s: pd.Series | pd.DataFrame | None, mean: float, std: float ) -> pd.Series:
        assert s is not None and isinstance(s, pd.Series), f"{type(s)}" 
        ret_val = s.copy()

        return (ret_val - mean) / std
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        ret_val = df.copy()
        ret_val = ret_val.drop(columns='Serial No.')

        cols = ['GRE Score', 'TOEFL Score', 'CGPA', 'University Rating', 'SOP', 'LOR']
        for col in cols:
            if col not in self.mean_std_pairs:
                self.mean_std_pairs[col] = (ret_val[col].mean(), ret_val[col].std())

            mean, std = self.mean_std_pairs[col]
            ret_val[col] = self.normalize_series(ret_val[col], mean, std)

        return ret_val

def F1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate F1 score with handling of edge cases.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        F1 score (between 0 and 1), or 0.0 for edge cases
    """
    tp = ((y_true == 1.0) & (y_pred == 1.0)).astype(float).sum()
    fp = ((y_true == 0.0) & (y_pred == 1.0)).astype(float).sum()
    fn = ((y_true == 1.0) & (y_pred == 0.0)).astype(float).sum()

    return (2 * tp) / ((2 * tp) + fp + fn)

def get_auroc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate AUROC score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        
    Returns:
        AUROC score (between 0 and 1)
    """
    order = np.argsort(-y_pred)
    y_true_sorted = y_true[order]
    num_positives = y_true.sum()
    num_negatives = len(y_true) - num_positives

    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)

    tpr = np.concatenate((np.zeros(1), tp / num_positives))
    fpr = np.concatenate((np.zeros(1), fp / num_negatives))
    
    return float(np.trapezoid(tpr, fpr))

def get_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    count = (y_true == y_pred).astype(float).sum()

    return count / len(y_true)

class SVMTrainer:
    def __init__(self):
        pass

    def my_train(self, X_train: np.ndarray, y_train: np.ndarray, kernel: str, **kwargs) -> SVC:
        '''
        Train the SVM model with the given kernel and parameters.

        Parameters:
            X_train: Training features
            y_train: Training labels
            kernel: Kernel type
            **kwargs: Additional arguments you may use
        Returns:
            SVC: Trained sklearn.svm.SVC model
        '''
        model = SVC(kernel=kernel, **kwargs)
        model.fit(X_train, y_train)

        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray, kernel: str, **kwargs) -> SVC:
        '''
        Train the SVM model with the given kernel and parameters.

        Parameters:
            X_train: Training features
            y_train: Training labels
            kernel: Kernel type
            **kwargs: Additional arguments you may use
        Returns:
            SVC: Trained sklearn.svm.SVC model
        '''
        model = SVC(kernel=kernel)
        model.fit(X_train, y_train)

        return model

    def get_support_vectors(self, model: SVC) -> np.ndarray:
        '''
        Get the support vectors from the trained SVM model.
        '''
        return model.support_vectors_

'''
Initialize my_best_model with the best model you found.
'''
my_best_model = SVC(kernel='linear', C=100)

def get_support_vectors_and_predictions(dataloader: DataLoader, trainer: SVMTrainer) -> dict:
    ret_val = {}

    feature_combs = [
        ('CGPA', 'SOP'),
        ('CGPA', 'GRE Score'),
        ('LOR', 'SOP'),
        ('LOR', 'GRE Score')
    ]

    train_data = dataloader.train_data

    for kernel in kernels:
        for fs in feature_combs:
            X_train, y_train = train_data[list(fs)].to_numpy(), train_data['label'].to_numpy()
            model = trainer.train(X_train, y_train, kernel)

            ret_val[kernel + '-' + '-'.join(fs)] = (model.support_vectors_, model.predict(X_train))

    return ret_val

def grid_search(dataloader: DataLoader, trainer: SVMTrainer) -> dict:
    best_hyper_params = {}
    best_accuracy = 0.0

    train_data = dataloader.train_data
    val_data = dataloader.val_data

    X_train, y_train = train_data.drop(columns='label').to_numpy(), train_data['label'].to_numpy()
    X_val, y_val = val_data.drop(columns='label').to_numpy(), val_data['label'].to_numpy()

    Cs = [0.01, 0.1, 1, 10, 100]
    gammas = [0.001, 0.01, 0.1, 1, 10]
    coef0s = [0, 0.5, 1]
    
    for kernel in kernels:
        if kernel == 'linear':
            param_grid = [{'C': c} for c in Cs]
        elif kernel == 'rbf':
            param_grid = [{'C': c, 'gamma': g} for c in Cs for g in gammas]
        elif kernel == 'poly':
            param_grid = [{'C': c, 'gamma': g, 'coef0': c0} for c in Cs for g in gammas for c0 in coef0s]
        else:
            raise ValueError()
        
        for params in param_grid:
            model = trainer.train(X_train, y_train, kernel, **params)
            y_pred = model.predict(X_val)
            accuracy = get_accuracy(y_val, y_pred)

            if best_accuracy < accuracy - 1e-12:
                best_accuracy = accuracy
                best_hyper_params = params

    print(best_accuracy)
    return best_hyper_params

if __name__ == "__main__":
    dataloader = DataLoader('hw3-univ-app-data.csv')
    trainer = SVMTrainer()

    sup_vec_pred_dict = get_support_vectors_and_predictions(dataloader, trainer)

    print(grid_search(dataloader, trainer))