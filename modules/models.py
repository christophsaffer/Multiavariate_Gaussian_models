import pandas as pd
import numpy as np


def decomposition(mu, sigma, ind):

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
    schur = np.mat(sig1) - np.mat(sig2) * \
        np.mat(np.linalg.inv(sig4)) * np.mat(sig2.T)
    schur = np.array(schur)

    # Return splitted mus, block matrices and schurcomplement
    return [mu1, mu2, sig1, sig2, sig4, schur]


class MvG:

    # Sets initial variables for an instance of the class
    def __init__(self, name, model=None):
        self.name = name
        self.model = model

    # Fits a model to given data
    def selection(self, data):

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
        return (factor * np.exp(-0.5 * np.mat(x - mu) * np.mat(
            np.linalg.inv(sigma)) * np.mat(x - mu).T))[0, 0]

    def argmaximum(self):
        # return mean vector as the argmax
        return np.array(self.model[1])

    def marg(self, margout):

        # Get parameters from model
        cols = np.array(self.model[0])
        mu = np.array(self.model[1])
        sigma = np.array(self.model[2])

        # Check if paramter is given as string or index (int)
        if type(margout) == str:
            margout = np.where(np.isin(cols, margout))[0][0]

        # Get splitted mus, block matrices and schurcomplement
        schurkomp = decomposition(mu, sigma, margout)

        # Calculate parameters of marg model
        cols = np.delete(cols, margout)
        mu = schurkomp[0]
        sigma = schurkomp[2]

        # Create new MvG object with marg model parameters
        return MvG("margmod", [cols, mu, sigma])

    def marg_list(self, margout):
        margmod = self
        margout.sort(reverse=True)
        for mar in margout:
            margmod = margmod.marg(mar)

        return margmod

    def cond(self, cond, value):

        # Get parameters from model
        cols = np.array(self.model[0])
        mu = np.array(self.model[1])
        sigma = np.array(self.model[2])

        # Check if paramter is given as string or index (int)
        if type(cond) == str:
            cond = np.where(np.isin(cols, cond))[0][0]

        # Get splitted mus, block matrices and schurcomplement
        schurkomp = decomposition(mu, sigma, cond)

        # Calculate parameters of cond model
        cols = np.delete(cols, cond)
        mu = np.mat(schurkomp[0]).T + np.mat(schurkomp[3]) * np.mat(
            np.linalg.inv(schurkomp[4])) * np.mat(
            np.array(value) - schurkomp[1])
        mu = np.array(np.transpose(mu))[0]
        sigma = schurkomp[5]

        # Create new MvG object with cond model parameters
        return MvG("condmod", [cols, mu, sigma])

    def sampling(self, k=100, save=True):

        # Get parameters from model
        cols = np.array(self.model[0])
        mu = np.array(self.model[1])
        sigma = np.array(self.model[2])

        # Cholesky decomposition of Sigma:
        A = np.linalg.cholesky(np.matrix(sigma))

        # Initialize the table of the sample points
        cols = list(cols)
        samples = pd.DataFrame(columns=cols)

        for i in range(0, k):
            x = []
            for j in range(0, round(len(A) / 2)):
                u = np.random.uniform()
                v = np.random.uniform()
                x.append(np.sqrt(-2 * np.log(u)) * np.cos(2 * np.pi * v))
                x.append(np.sqrt(-2 * np.log(u)) * np.sin(2 * np.pi * v))

            if len(A) % 2 != 0:
                x.pop()

            x = np.mat(x)
            x = A * x.T + np.mat(mu).T
            x = np.array(x.T)[0]
            samples.loc[i] = x

        if save:
            samples.to_csv('samples/sampling_' +
                           str(k) + '_' + self.name + '.csv')
            print('Saved successfully')

        return samples

    def testmarg(self):

        # Generate example
        testmu = np.array([1, 2, 3])
        testsig = np.array([[2, 3, 1], [3, -4, 2], [1, 2, -2]])

        margmu = np.array([1, 3])
        margsig = np.array([[2, 1], [1, -2]])

        testmod = MvG("testmod", [["a", "b", "c"], testmu, testsig])
        testmodmarg = testmod.marg("b")

        if ((np.array_equal(testmodmarg.model[1], margmu)) & (np.array_equal(testmodmarg.model[2], margsig))):
            return True
        else:
            return False

    def testcond(self):

        # Generate example
        testmu = np.array([1, 2, 3])
        testsig = np.array([[2, 3, 1], [3, -4, 2], [1, 2, -2]])

        condmu = np.array([3.5, 3.5])
        condsig = np.array([[-8.5, 0.5], [0.5, -2.5]])

        testmod = MvG("testmod", [["a", "b", "c"], testmu, testsig])
        testmodcond = testmod.cond("a", 2)

        if ((np.array_equal(testmodcond.model[1], condmu)) & (np.array_equal(testmodcond.model[2], condsig))):
            return True
        else:
            return False
