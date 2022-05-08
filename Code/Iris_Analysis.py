#%%
import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler ,StandardScaler
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from KMeans_Class import KMeansAnalysis
#%%

kmeans_kwargs = {
             "init": "random",
             "n_init": 10,
             "max_iter": 300
}



iris = pd.read_csv('iris.csv')

KM = KMeansAnalysis(k=4,data=iris.iloc[:,0:5],scaletype='Z',kmean_Kwargs=kmeans_kwargs,response= 'Species')
KM.runKmeans()
KM.plots()
KM.Scatterplot(3,x='Sepal.Length',y='Petal.Length')
KM.Scatterplot3d(3,x='Sepal.Length',y='Petal.Length',z='Petal.Width')
check = KM.ResponseAnalysis(3)


#%%
sns.scatterplot(x=iris['Sepal.Length'] , y =iris['Petal.Length'],data=iris, hue='Species',palette='rainbow')
plt.title('Iris Actual Species')
plt.show()

#%%

fig = plt.figure(figsize=(8,6 ))
ax = Axes3D(fig)
cmap = ListedColormap(sns.color_palette("rainbow", 256).as_hex())
colors = {'setosa':0,'versicolor':1,'virginica':2}
xp = iris['Sepal.Length']
yp = iris['Petal.Length']
zp = iris['Petal.Width']
sc = ax.scatter(xp,yp,zp,s=40,c=iris['Species'].map(colors),marker='o',cmap=cmap,alpha=1)
ax.set_xlabel(xp.name)
ax.set_ylabel(yp.name)
ax.set_zlabel(zp.name)
ax.set_title('Title',loc='center')
plt.legend()
plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

plt.tight_layout()
plt.show()

#%%
df1 = iris[iris['Species']=='setosa'].reset_index(drop=True)
df2 = iris[iris['Species']=='versicolor'].reset_index(drop=True)
df3 = iris[iris['Species']=='virginica'].reset_index(drop=True)
axes = plt.subplot(111, projection='3d')


x1=df1['Sepal.Length']
x2 =df2['Sepal.Length']
x3 =df3['Sepal.Length']

y1=df1['Petal.Length']
y2=df2['Petal.Length']
y3=df3['Petal.Length']

z1=df1['Petal.Width']
z2=df2['Petal.Width']
z3=df3['Petal.Width']

axes = plt.subplot(111, projection='3d')
axes.plot(x1,y1,z1,"o",label=df1['Species'][0])
axes.plot(x2,y2,z2,"o",label=df2['Species'][0])
axes.plot(x3,y3,z3,"o",label=df3['Species'][0])
axes.set_xlabel('Sepal.Length')
axes.set_ylabel('Petal.Length')
axes.set_zlabel('Petal.Width')


plt.legend(loc="upper right")
plt.show()