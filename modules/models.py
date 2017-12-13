import pandas as pd
import numpy as np


class MvG:

    # Sets intial variables for an instance of the class
    def __init__(self, name, model=0):
        self.name = name
        self.model = model

    # Fits a model to given data
    def selection(self, data, save=True):

        # Get the data and initialize some variables
        data = pd.read_csv(data, index_col=0)
        numb_cols = len(data.T)
        numb_rows = len(data)
        cols = np.array(data.columns)

        # Calculate parameter mu
        mu = data.mean().values

        # Calculate parameter sigma
        centr = np.array(data - mu)
        sigma = np.zeros([numb_cols, numb_cols])

        for entry in centr:
            sigma += np.mat(entry).T * np.mat(entry)
        sigma = sigma / numb_rows

        # Save model parameter
        self.model = [cols, mu, sigma]

    def density(self, x):

        # Get model parameter
        mu = np.array(self.model[1])
        sigma = np.array(self.model[2])
        dim = len(mu)
        x = np.array(x)
        factor = (0.5)**(-dim * 0.5) * np.linalg.det(sigma)**(-0.5)

        # Return density
        return (factor * np.exp(-0.5 * np.mat(x - mu) * np.mat(np.linalg.inv(sigma)) * np.mat(x - mu).T))[0, 0]

    def maximum(self):
        # return mean vector as the argmax
        return np.array(self.model[1])

    def schur(self, mu, sigma, ind):

        # Get number of variables
        le = len(sigma)

        # Extract new mus
        mu1 = np.delete(mu, ind)
        mu2 = np.array([mu[ind]])

        # Reorder matrix sigma
        # Put row and column at index 'ind' to the last row and column
        i = np.array(range(0, le))
        i = np.delete(i, ind)
        i = np.append(i, ind)
        reordered = sigma[i, :][:, i]

        # Get block matrices
        sig1 = reordered[0:le - 1, 0:le - 1]
        sig2 = reordered[0:le - 1, le - 1:le]
        sig4 = reordered[le - 1:le, le - 1:le]

        # Calculate schurcomplement
        s = np.mat(sig1) - np.mat(sig2) * \
            np.mat(np.linalg.inv(sig4)) * np.mat(sig2.T)
        s = np.array(s)

        # Return splitted mus, block matrices and schurkomplement
        return [mu1, mu2, sig1, sig2, sig4, s]

    def marg(self, margout):

        # Get parameters from model
        cols = np.array(self.model[0])
        mu = np.array(self.model[1])
        sigma = np.array(self.model[2])

        # Get splitted mus, block matrices and schurkomplement
        schurkomp = self.schur(mu, sigma, margout)

        # Calculate parameters of marg model
        cols = np.delete(cols, margout)
        mu = schurkomp[0]
        sigma = schurkomp[2]

        # Create new MvG Object with marg model parameters
        return MvG("margmod", [cols, mu, sigma])

    def cond(self, cond, value):

        # Get parameters from model
        cols = np.array(self.model[0])
        mu = np.array(self.model[1])
        sigma = np.array(self.model[2])

        # Get splitted mus, block matrices and schurkomplement
        schurkomp = self.schur(mu, sigma, cond)

        # Calculate parameters of cond model
        cols = np.delete(cols, cond)
        mu = np.mat(schurkomp[0]).T + np.mat(schurkomp[3]) * np.mat(
            np.linalg.inv(schurkomp[4])) * np.mat(np.array(value) - schurkomp[1])
        mu = np.array(np.transpose(mu))[0]
        sigma = schurkomp[5]

        # Create new MvG Object with cond model parameters
        return MvG("condmod", [cols, mu, sigma])
