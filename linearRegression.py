

from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np

X, y= make_regression(n_samples=100, n_features=1, noise=1.4, bias=1)


data_x = np.linspace(1.0, 10.0, 100)[:,np.newaxis]
data_y = np.sin(data_x)+0.1*np.power(data_x,2)+0.5*np.random.randn(100,1)

#for the sake of generic code, assuming b to be theta0 and w to be theta1

#plotting the line and out datapoints
def plotLine(theta0, theta1, X, y):
    #generate a line from theta0 and theta1
    max_x = np.max(X) + 1
    min_x = np.min(X) - 1
    xplot = np.linspace(min_x, max_x, 1000)
    yplot = theta0 + theta1 * xplot

    #plot the line
    plt.plot(xplot, yplot, color='#58b970', label='Regression Line')
    
    #scatter the points in dataset
    plt.scatter(X,y)
    plt.show()



def hypothesis(theta0, theta1, x):
    #out assumed hypothesis function i.e. y=wx+b
    return theta0 + (theta1*x) 

def cost(theta0, theta1, X, y):
    costValue = 0 
    #calculating the cost value for all points in dataframe and adding it to get the cost funciton

    for (xi, yi) in zip(X, y):
        #cost = (1/2)*(y-(wx+b))
        costValue += 0.5 * ((hypothesis(theta0, theta1, xi) - yi)**2)
        print(costValue)
    return costValue




def derivatives(theta0, theta1, X, y):
    dtheta0 = 0
    dtheta1 = 0

    #calculating the partial differenial for minimizing the cost
    for (xi, yi) in zip(X, y):
        dtheta0 += hypothesis(theta0, theta1, xi) - yi
        dtheta1 += (hypothesis(theta0, theta1, xi) - yi)*xi
    
    #divided by m=number of samples in the dataframe
    dtheta0 /= len(X)
    dtheta1 /= len(X)

    return dtheta0, dtheta1

def updateParameters(theta0, theta1, X, y, alpha):

    #updatating w and b from the differential calculated
    dtheta0, dtheta1 = derivatives(theta0, theta1, X, y)
    theta0 = theta0 - (alpha * dtheta0)
    theta1 = theta1 - (alpha * dtheta1)

    return theta0, theta1
    

def LinearRegression(X, y):

    #randomly initializing w and b
    theta0 = np.random.rand()
    theta1 = np.random.rand()
    

    #running linear regression 1000 times to get a good fit
    for i in range(0, 1000):
        if i % 100 == 0:
            plotLine(theta0, theta1, X, y)
        # print(cost(theta0, theta1, X, y))
        theta0, theta1 = updateParameters(theta0, theta1, X, y, 0.005)



    


LinearRegression(X,y)
