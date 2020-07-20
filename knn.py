import numpy as np

class knn(object):
    
    def __init__(self, xTr, yTr, xTe, yTe, k):
        self.xTr = xTr
        self.yTr = yTr
        self.xTe = xTe
        self.yTe = yTe
        self.k = k
        
    def knn_prediction(self): 
        
        
        #Define euclidean distance:
        def l2distance(self):
            '''We express the squared euclidean distance given by: D_ij = (x_i - z_i)*(x_i-z_i).T
            We can also express the distance in some linear combination as follows:
            D = S + R - 2G, where
            S = x_i*x_i^T
            R = z_j * z_j^T
            G = dot(X,Z.T)'''

            n,d1=self.xTr.shape
            m,d2=self.xTe.shape
            assert (d1==d2), "Dimensions of input vectors must match!"

            S = np.sum(self.xTr*self.xTr,axis=1)[:, np.newaxis] 
            R = np.sum(self.xTe*self.xTe,axis=1)
            G = np.dot(self.xTr,self.xTe.T)

            D2 = S + R - 2*G
            D2[D2<0]=0
            D = np.sqrt(D2) 
            return D    
    
        def findknn(self):
            """
            Here we use the function to calculate the euclidean distance between some training data 
            and some test data and return the index. We also specify the number of nearest neighbors:
            """
            d = l2distance(self) #calculate euclidean distance

            #Use sort to find neighbors:
            dist = np.sort(d, axis=0)[:self.k] #find k nearest neighbors
            index = np.argsort(d, axis=0)[:self.k] #find index of neighbors
            return index, dist


            """
            We find the nearest neigbors(s) and return our predictions for some test data:
            """
        def mode(a):
            return (max(a,key=a.count))
            # fix array shapes
        self.yTr = self.yTr.flatten()

        index, dist = findknn(self)
        predictions = [] #empty list for saving preidction output

        for i,v in enumerate(index.T): #loop through all indexes (but transposed to have index as we need it)
            predictions.append(mode((self.yTr[v]).tolist())) #find index from clostest neighbor(s) and return the label    
    
        return np.array(predictions) #return array of predictions


    def accuracy(self,preds):
        """
        Calculate a zero-one loss as accuracy:
        """
        truth = self.yTe.flatten()
        preds = preds.flatten()
        count = 0

        #Loop through all true values:
        for i,v in enumerate(truth):
            if truth[i] == preds[i]: #create 0/1 loss depending on if true equals prediction
                  count += 1
        acc = count/len(truth) #calculate share of correctly classified cases
    
        return np.float64(acc) #return accuracy as float64