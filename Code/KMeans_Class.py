#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler ,StandardScaler
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

#%%
class KMeansAnalysis():
    '''
  This Class is to analyze K Means Clustering algorithm outputs by taking in a set of feature vectors
  Also, a response can be entered and its proportion in a cluster calculated
  The class has the following methods

  runKmeans - which iterates multiple values of k and records the metrics for sil and SSE

  plots - which creates plots for elbow method and sil score

  ResponseAnalysis - if a response was passed in calculates the proportion of that reponse for a given k
                    entered into the method. This also returns the data frame with the cluster labels for the given
                    :key
  Scatterplot - Takes in the K value decided for the number of clusters as well as an x var and a y var and returns
                a 2d plot of the x y space labeling each cluster by color

  Scatterplot3d - works the same as scatter plot but also takes in a z variable and returns a 3d scatterplot
    '''
    def __init__(self,k,data,scaletype,kmean_Kwargs,response =''):
        #Returns list of all cluster sizes up to k for trying over data
        self.k = list(range(2,k+1))
        #creates response column of data
        self.responsegroups = []
        if response != '':
            self.data = data.loc[:, data.columns != response]
            self.data2 = data.loc[:, data.columns != response]
            #Set aside response column
            self.response = data[response]
            #empty list for response groups

        else:
            self.data = data
            self.data2 = data
            self.response=response
        #creates scaling type to use
        self.scaletype = scaletype
        #dictionary of arguments for Kmeans to be created outside of class
        self.kmean_Kwargs = kmean_Kwargs
        #empty list for SSE Storage
        self.SSEList = []
        #empty List for Sil score storage
        self.SilList= []



    def runKmeans(self):
        '''
        If all columns in self.data are numeric runKmeans will scale variables based on the self.scaletype atribute
        and run k means for 2-self.k
        :return: runKmeans does not return any tables or plots, but updates the variance and sil list for every value
        of k as well as update self.responsegroup list with data frames that include the cluster assignment for each
        observation in the data
        '''
        if np.all(self.data.dtypes.apply(lambda x: x in ['int64','float64'] )):
            if self.scaletype == 'MinMax':
                minmax = MinMaxScaler()
                self.data = minmax.fit_transform(self.data)

            elif self.scaletype == 'Z':
                scales = StandardScaler()
                self.data = scales.fit_transform(self.data)

            for k in self.k:
                #run Kmeans
                kmeans = KMeans(n_clusters= k, **self.kmean_Kwargs)
                kmeans.fit(self.data)
                #Store Response Group DataFrame
                if (type(self.response) != str):
                    self.responsegroups.append(self.response.to_frame().join(pd.Series(kmeans.labels_,
                                                                                 name='Cluster').to_frame()))
                else:
                    self.responsegroups.append(pd.Series(kmeans.labels_,name='Cluster').to_frame())
                # Store SSE and SIL Scores
                self.SSEList.append(kmeans.inertia_)
                self.SilList.append(silhouette_score(self.data,kmeans.labels_))
        else:
            print('Some columns have incorrect data types. Only float64 and int64 acceptable')




    def plots(self):
        #Plot SSE and Silscores VS k
        fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
        ax1.set_title('Elbow Method')
        ax1.set(xlabel = 'K',ylabel= 'SSE')
        ax1.plot(self.k, self.SSEList,color='green', linestyle='dashed', marker='o',
                 markerfacecolor='red', markersize=10)

        ax2.set_title('SilhouetteMethod')
        ax2.set(xlabel = 'K',ylabel= 'Silhouette Score')
        ax2.plot(self.k, self.SilList, color='green', linestyle='dashed', marker='o',
                 markerfacecolor='red', markersize=10)

        plt.tight_layout()
        plt.show()

    def ResponseAnalysis(self, num):
        #calculates percent of a potential response given a certain value of K
        #Returns the dataframe of groupings which can be added to data set
        if type(self.response) != str:
            cmat = pd.crosstab(index=self.responsegroups[num - 2]['Cluster'],columns=self.response)
            print('Cluster Response Confusion Matrix')
            print(cmat)
            print('Proportion of Response in Cluster')
            print(cmat.divide(cmat.sum(axis=1),axis=0))
            return self.data2.join(self.responsegroups[num - 2]['Cluster'].to_frame()).join(self.response.to_frame())
        else:
            return self.data2.join(self.responsegroups[num - 2]['Cluster'].to_frame())

    def Scatterplot(self, num,x,y):
        if(isinstance(x,int) & isinstance(y,int)):
            plt.figure(figsize=(8, 6))
            new_df = self.data2.join(self.responsegroups[num-2]['Cluster'].to_frame())
            sns.scatterplot(x=new_df.iloc[:,x],y=new_df.iloc[:,y],data = new_df,hue='Cluster',palette='rainbow')
            plt.title('K-Means Scatter')
            plt.show()
        elif (isinstance(x,str) & isinstance(y,str)):
            plt.figure(figsize=(8, 6))
            new_df = self.data2.join(self.responsegroups[num - 2]['Cluster'].to_frame())
            sns.scatterplot(x=x, y=y, data=new_df, hue='Cluster', palette='rainbow')
            plt.title('K-Means Scatter')
            plt.show()

    def Scatterplot3d(self,num,x,y,z):
        if (isinstance(x, int) & isinstance(y, int) & isinstance(z, int)):
            fig = plt.figure(figsize=(8,6 ))
            ax = Axes3D(fig)
            cmap = ListedColormap(sns.color_palette("rainbow", 256).as_hex())
            colors = self.responsegroups[num-2]['Cluster']
            xp = self.data2.iloc[:,x]
            yp = self.data2.iloc[:, y]
            zp = self.data2.iloc[:, z]
            sc = ax.scatter(xp,yp,zp,s=40,c=colors,marker='o',cmap=cmap,alpha=1)
            ax.set_xlabel(xp.name)
            ax.set_ylabel(yp.name)
            ax.set_zlabel(zp.name)
            ax.set_title('Title',loc='center')
            plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

            plt.tight_layout()
            plt.show()
        elif (isinstance(x, str) & isinstance(y, str) & isinstance(z, str)):
            fig = plt.figure(figsize=(8,6 ))
            ax = Axes3D(fig)
            cmap = ListedColormap(sns.color_palette("rainbow", 256).as_hex())
            colors = self.responsegroups[num-2]['Cluster']
            xp = self.data2[x]
            yp = self.data2[y]
            zp = self.data2[z]
            sc = ax.scatter(xp,yp,zp,s=40,c=colors,marker='o',cmap=cmap,alpha=1)
            ax.set_xlabel(xp.name)
            ax.set_ylabel(yp.name)
            ax.set_zlabel(zp.name)
            ax.set_title('Title',loc='center')
            plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

            plt.tight_layout()
            plt.show()

#%%

