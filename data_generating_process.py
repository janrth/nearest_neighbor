import numpy as np

class data_generating_process(object):
    np.random.seed(12)
    
    def __init__(self, n_samples, groups):
        self.n_samples = n_samples
        self.groups = groups

    def women_data(self):
        class_women = np.random.multivariate_normal([50, 165], [[15, .1],[.1, 15]], self.n_samples)
        class_women_labels = -np.ones(self.n_samples)
        return class_women, class_women_labels    
     
    def men_data(self):
        class_men = np.random.multivariate_normal([80, 175], [[15, .1],[.1, 15]], self.n_samples)
        class_men_labels = np.ones(self.n_samples)
        return class_men, class_men_labels    
    
    def alien_women_data(self):
        class_alien_women = np.random.multivariate_normal([70, 170], [[30, .1],[.1, 30]], self.n_samples)
        class_alien_women_labels = -np.ones(self.n_samples)
        return class_alien_women, class_alien_women_labels
    
    def alien_men_data(self):
        class_alien_men = np.random.multivariate_normal([35, 135], [[30, .1],[.1, 30]], self.n_samples)
        class_alien_men_labels = np.ones(self.n_samples)
        return class_alien_men, class_alien_men_labels 
    
    def stacking(self,class_men,class_men_labels,class_women,class_women_labels,class_alien=None,class_alien_labels=None):
        if self.groups == 2:
            features = np.vstack((class_men,class_women))
            labels = np.hstack((class_men_labels, class_women_labels))
        else:
            features = np.vstack((class_men,class_women,class_alien))
            labels = np.hstack((class_men_labels, class_women_labels,class_alien_labels))
        return features, labels   