from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import numpy as np


poly = PolynomialFeatures(5)

data_x = np.linspace(1.0, 10.0, 100)[:,np.newaxis]
data_x=data_x.reshape(-1,1)
data_y = np.sin(data_x)+0.1*np.power(data_x,2)+0.5*np.random.randn(100,1)

model = make_pipeline(poly, Ridge())
model.fit(data_x,data_y)
y_plot = model.predict(data_x)

plt.scatter(data_x,data_y)
plt.plot(data_x, y_plot, color='red')
plt.subplot()
plt.show()