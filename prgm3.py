import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("data.csv")
x=dataset.iloc[:,[1,2]].values
y=dataset.iloc[:,0].values
print(x,y)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters='i', init="k-means++", random_state=42)
plt.scatter(x[:,0], x[:,1],c=y, s=50, cmap='spring')
plt.show()
wcss_list=[]
xfit=np.linspace(-1,3.5)
plt.scatter(x[:,0], x[:,1],c=y, s=50, cmap='spring')
for m,b,d in [(1,0.65,0.33), (0.5,1.6,0.55), (-0.2,2.9,0.2)]:
   yfit=m*xfit+b
   plt.plot(xfit,yfit,'-k')
   plt.fill_between(xfit,yfit-d,yfit+d,edgecolor='none',color='#AAAAAA',alpha=.4)
plt.xlim(-1,3.5)

for i in range(1,11):
    plt.plot(range(1,11),wcss_list)
    plt.title('Graph method')
    plt.xlabel('Number of clusters')
    plt.ylabel('wcss_list')
    plt.show()

kmeans= KMeans(n_clusters=3, init="k-means++", random_state=42)
y_predict=kmeans.predict(x)
print(y_predict)

plt.scatter(x[y_predict==0,0],x[y_predict==0,1],s=100,c='red',label='cluster 1')
plt.scatter(x[y_predict==1,0],x[y_predict==1,1],s=100,c='blue',label='cluster 2')
plt.scatter(x[y_predict==2,0],x[y_predict==2,2],s=100,c='green',label='cluster 3')
plt.scatter(kmeans.cluster.center_[:,0],kmeans.cluster.center_[:,1],s=300,c='black',label='cluster')

plt.title('Clusters of customers')
plt.xlabel('Annual income')
plt.ylabel('salary')
plt.show()






