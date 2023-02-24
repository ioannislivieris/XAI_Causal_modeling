import os
import random
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split

seed = 42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)

class Synthetic:
    ''' 
        Class for creating a synthetic dataset

        Parameters
        ----------
        type: str
            'linear' or 'sin'
        seed: int
            random seed
        size: int
            number of instances
        n_f: int
            number of features
        n_i: int
            number of irrelevant features
        p: float
            probability 
    '''

    def __init__(self, type='linear', size=1000, n_f=10, n_i=5, p=0.3, seed=42):
        self.type = type
        self.size = size
        self.n_f = n_f
        self.n_i = n_i
        self.p = p
        self.seed = seed

        np.random.seed(seed=self.seed)

    def create_dataset(self, train_size=0.8):
        ''' 
        Create a synthetic dataset

        Parameters
        ----------
        train_size: float
            percentage of data for training (0 < train_size < 1)
        '''

        if self.type == 'linear':
            X, t, y, y_potential = self._dataset_linear()
        elif self.type == 'sin':
            X, t, y, y_potential = self._dataset_sin()
        else:
            raise Exception('Unknown dataset')

        X_train, X_test, t_train, t_test, y_train, y_test, y_potential_train, y_potential_test = train_test_split(
            X, t, y, y_potential, train_size=train_size, random_state=self.seed, stratify=t)

        self.X_train = X_train
        self.X_test = X_test
        self.t_train = t_train
        self.t_test = t_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_potential_train = y_potential_train
        self.y_potential_test = y_potential_test

    def get_training_data(self):
        ''' 
        Returns training data

        Returns
        -------
        X_train: np.array
            Covariate matrix of training dataset
        t_train: np.array
            treatment vector of training dataset
        y_train: np.array
            target vector of training dataset
        y_potential_train: np.array 
            factual and counterfactual target vector of training dataset
        '''

        return self.X_train, self.t_train, self.y_train, self.y_potential_train

    def get_testing_data(self):
        ''' 
        Returns testing data

        Returns
        -------
        X_test: np.array
            Covariate matrix of testing dataset
        t_test: np.array
            treatment vector of testing dataset
        y_test: np.array
            target vector of testing dataset
        y_potential_test: np.array 
            factual and counterfactual target vector of testing dataset
        '''

        return self.X_test, self.t_test, self.y_test, self.y_potential_test

    def _dataset_linear(self):
        ''' 
        1st Dataset

        Returns
        -------
        x: np.array
            Covariate matrix
        t: np.array
            treatment vector
        y: np.array
            target vector
        y_potential: np.array
            factual and counterfactual target vector
        '''        
        t = stats.bernoulli.rvs(self.p, size=self.size)
        u_1 = 20*stats.norm.rvs(size=self.size) + 50
        u_2 = 10*stats.norm.rvs(size=self.size) + 20

        means = np.log(50*stats.uniform.rvs(size=self.n_f))
        std = np.log(20*stats.uniform.rvs(size=self.n_f))
        coefs = 5*stats.uniform.rvs(size=self.n_i) - 2
        u_x = stats.multivariate_normal.rvs(
            means, std*np.eye(self.n_f), size=self.size)
        # Make it log-normal
        u_x = np.exp(u_x)

        y = u_1 + t*u_2 + np.dot(coefs, u_x[:, self.n_i:self.n_f].T)

        y_potential = np.array([u_1 + np.zeros(self.size)*u_2 + np.dot(coefs, u_x[:, self.n_i:self.n_f].T),
                               u_1 + np.ones(self.size)*u_2 + np.dot(coefs, u_x[:, self.n_i:self.n_f].T)]).T

        return u_x, t, y, y_potential

    def _dataset_sin(self):
        ''' 
        2nd Dataset

        Returns
        -------
        x: np.array
            Covariate matrix
        t: np.array
            treatment vector
        y: np.array
            target vector
        y_potential: np.array
            factual and counterfactual target vector
        '''
        t = stats.bernoulli.rvs(self.p, size=self.size)
        u_1 = 20*stats.norm.rvs(size=self.size) + 50
        u_2 = 10*stats.norm.rvs(size=self.size) + 20

        means = np.log(50*stats.uniform.rvs(size=self.n_f))
        std = np.log(20*stats.uniform.rvs(size=self.n_f))
        coefs = 5*stats.uniform.rvs(size=self.n_i) - 2
        u_x = stats.multivariate_normal.rvs(
            means, std*np.eye(self.n_f), size=self.size)
        # Make it log-normal
        u_x = np.exp(u_x)

        y = u_1 + t*u_2 + 100*np.sin(np.sum(u_x[:, self.n_i:self.n_f], axis=1))

        y_potential = np.array([u_1 + np.zeros(self.size)*u_2 + 100*np.sin(np.sum(u_x[:, self.n_i:self.n_f], axis=1)),
                               u_1 + np.ones(self.size)*u_2 + 100*np.sin(np.sum(u_x[:, self.n_i:self.n_f], axis=1))]).T

        return u_x, t, y, y_potential
