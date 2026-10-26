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
    
    def create_binary_label(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Create a binary label for the training data.
        '''
        self.threshold = df['Chance of Admit'].median()
        ret_val = df.copy()
        binary_label = (df['Chance of Admit'] > self.threshold).astype(int)
        ret_val['label'] = binary_label

        return ret_val
    
    def normalize_series(self, s: pd.Series | pd.DataFrame | None ) -> tuple[pd.Series, float, float]:
        assert s is not None and isinstance(s, pd.Series), "How did this happen?" 

        ret_val = s.copy()
        mean, std = s.mean(), s.std()

        assert isinstance(std, float), "How did this happen?"

        return (ret_val - mean) / std, mean, std
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop('Serial No.')

        cols = ['GRE Score', 'TOELF Score', 'CGPA', 'University Rating', 'SOP', 'LOR']
        for col in cols:
            df[cols], mean, std = self.normalize_series(df[cols])
            self.mean_std_pairs[col] = (mean, std)

        return df

class SVMTrainer:
    def __init__(self):
        pass

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
        if kernel == 'linear':
            model = SVC(kernel=kernel)

        elif kernel == 'rbf':
            # assert all(k in kwargs for k in ['C', 'gamma']), "RBF: Why is it not here?"

            # C, gamma = kwargs['C'], kwargs['gamma']
            model = SVC(kernel=kernel)

        elif kernel == 'poly':
            # assert all(k in kwargs for k in ['C', 'gamma', 'coef0']), "Poly: Why is it not here?"

            # C, gamma, coef0 = kwargs['C'], kwargs['gamma'], kwargs['coef0']
            # model = SVC(kernel=kernel, C=C, gamma=gamma, coef0=coef0)
            model = SVC(kernel=kernel)

        else:
            raise ValueError(f'Unsupported kernel: {kernel}')

        model.fit(X_train, y_train)

        return model

    def get_support_vectors(self, model: SVC) -> np.ndarray:
        '''
        Get the support vectors from the trained SVM model.
        '''
        return model.support_vectors_

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
    
'''
Initialize my_best_model with the best model you found.
'''
my_best_model = SVC()

if __name__ == "__main__":
    print("Hello, World!")