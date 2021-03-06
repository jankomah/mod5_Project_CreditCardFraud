# Binary Encoding of columns
def map_type(column):
    mapped = []
    for value in column:
        if value == "CASH_OUT":
            mapped.append(1)
        else:
            mapped.append(0)
    return mapped       

new_col = map_type(df1["type"])


# Base function
def base_func(element):
    """train and fit the model"""
    model = element()
    model.fit(X_train , y_train)
    
    """predict"""
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    """evaluation"""
    train_accuracy = roc_auc_score(y_train , train_preds)
    test_accuracy = roc_auc_score(y_test , test_preds)
    
    print(str(element))
    print("--------------------------------------------")
    print(f"Training Accuracy: {(train_accuracy * 100) :.4}%")
    print(f"Test Accuracy : {(test_accuracy * 100) :.4}%")
    
    """Store accuracy in a new DataFrame"""
    score_logreg = [element , train_accuracy , test_accuracy]
    models = pd.DataFrame([score_logreg])    



# Grid Search for Best Parameters
def grd_src(classifier , param_grid):
    param_grid = param_grid
  
  """Instantiate the tuned model"""
    grid_search = GridSearchCV(classifier, param_grid, cv=3, n_jobs=-1)
  
  """train the tuned model"""
    grid_search.fit(X_train , y_train)

  """print best paramets during the grid search"""
    print((str(classifier) + "Best Parameters"))
    print("-----------------------------------")
    print(grid_search.best_params_)
    return grid_search.best_params_



# Grid Search for best Parameters with estimator 2.0
def grid_srch(classifier , param_grid):
    param_grid = param_grid
    
    """Instantiate the tuned model"""
    grid_search = GridSearchCV(classifier , estimator , param_grid , cv = 3 , n_jobs = -1)
    
    """train the tuned model"""
    grid_search.fit(X_train , y_train)
    
    """print best parameters during the grid search"""
    print((str(classifier) + "Best Parameters"))
    print("------------------------------------")
    print(grid_search.best_params_)
    return grid_search.best_params_
    return grid_search.best_estimator_


# Run Models for best parameters - Main Model function
def run_model(model, X_train, y_train,X_test, y_test ):
    model.fit(X_train, y_train)

    """predictions"""
    train_preds = model.predict_proba(X_train).argmax(1)
    test_preds = model.predict_proba(X_test).argmax(1)

    
    """plotting ROC Curve"""
    fpr, tpr, threshold = metrics.roc_curve(y_test, test_preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.gcf().savefig('roc.png')

    """evaluations"""
    train_auc = roc_auc_score(y_train, train_preds)
    test_auc = roc_auc_score(y_test, test_preds)
    report = classification_report(y_test, test_preds)

    print(metrics.confusion_matrix(y_test, test_preds))

    test_preds[test_preds>roc_auc]= 1
    test_preds[test_preds<=roc_auc]= 0

    """print reports of the model accuracy"""
    print('Model Scores')
    print("------------------------")
    print(f"Training AUC: {(train_auc * 100):.4}%")
    print(f"Test AUC:     {(test_auc * 100):.4}%")
    print("------------------------------------------------------")
    print('Classification Report : \n', report)
    return test_preds






# Run Models for best parameters
def run_model2(model, X_train, y_train,X_test, y_test ):
    model.fit(X_train, y_train)

    """predict"""
    train_preds = model.predict_proba(X_train)
    test_preds = model.predict_proba(X_test)

    """evaluate"""
    train_auc = roc_auc_score(y_train, train_preds)
    test_auc = roc_auc_score(y_test, test_preds)
    report = classification_report(y_test, test_preds)

    """print reports of the model accuracy"""
    print('Model Scores')
    print("------------------------")
    print(f"Training AUC: {(train_auc * 100):.4}%")
    print(f"Test AUC:     {(test_auc * 100):.4}%")
    print("------------------------------------------------------")
    print('Classification Report : \n', report)


# Run models with Confusion Matrix
def run_model3(model, X_train, y_train,X_test, y_test ):
    model.fit(X_train, y_train)

    """predict"""
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    """evaluate"""
    train_accuracy = roc_auc_score(y_train, train_preds)
    test_accuracy = roc_auc_score(y_test, test_preds)
    report = classification_report(y_test, test_preds)
    
    """Confusion Matrix"""
    cnf_matrix = confusion_matrix(y_test , test_preds)
    print("Confusion Matrix:\n" , cnf_matrix)

    """print reports of the model accuracy"""
    print('Model Scores')
    print("------------------------")
    print(f"Training Accuracy: {(train_accuracy * 100):.4}%")
    print(f"Test Accuracy:     {(test_accuracy * 100):.4}%")
    print("------------------------------------------------------")
    print('Classification Report : \n', report)
    print("-----------------------------------------------------")
    print("Confusion Matrix:\n" , cnf_matrix)
    
    
    

# Confusion Matrix
import numpy as np
import itertools
import matplotlib.pyplot as plt
%matplotlib inline

"""Create the basic matrix"""
plt.imshow(cnf_matrix,  cmap=plt.cm.Blues) 

"""Add title and axis labels"""
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')

"""Add appropriate axis scales"""
class_names = set(target) # Get class labels to add to matrix
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

"""Add labels to each cell"""
thresh = cnf_matrix.max() / 2. # Used for text coloring below
"""Here we iterate through the confusion matrix and append labels to our visualization """
for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, cnf_matrix[i, j],
                 horizontalalignment='center',
                 color='white' if cnf_matrix[i, j] > thresh else 'black')

""'Add a legend"""
plt.colorbar()
plt.show()


# Creating Function for Confusion Matrix
def plot_confusion_matrix(cm, classes,
                          normalize = False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  
"""Create the basic matrix"""
  plt.imshow(cm,  cmap=plt.cm.Blues) 

"""Add title and axis labels"""
  plt.title('Confusion Matrix')
  plt.ylabel('True label')
  plt.xlabel('Predicted label')

"""Add appropriate axis scales"""
  class_names = set(target) # Get class labels to add to matrix
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

"""Add labels to each cell"""
  thresh = cm.max() / 2. # Used for text coloring below
"""Here we iterate through the confusion matrix and append labels to our visualization"""
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

"""Add a legend"""
  plt.colorbar()
  plt.show()


# Updated function for Confusion Matrix to a normaalized confusion matrix
def plot_norm_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    """Check if normalize is set to True"""
    """If so, normalize the raw confusion matrix before visualizing"""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, cmap=cmap)
    
    """Add title and axis labels"""
    plt.title('Confusion Matrix') 
    plt.ylabel('True label') 
    plt.xlabel('Predicted label')
    
    """Add appropriate axis scales"""
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    """Text formatting"""
    fmt = '.2f' if normalize else 'd'
    """Add labels to each cell"""
    thresh = cm.max() / 2.
    """Here we iterate through the confusion matrix and append labels to our visualization """
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
    
    """Add a legend"""
    plt.colorbar()
    plt.show() 
    
    

# Modelling labs start here
from sklearn.metrics import roc_auc_score
def scores(model,X_train3,X_val,y_train3,y_val):
    train_prob = model.predict_proba(X_train3)[:,1]
    val_prob = model.predict_proba(X_val)[:,1]
    train = roc_auc_score(y_train3,train_prob)
    val = roc_auc_score(y_val,val_prob)
    print('train:',round(train,2),'test:',round(val,2))

    
def annot(fpr,tpr,thr):
    k=0
    for i,j in zip(fpr,tpr):
        if k %50 == 0:
            plt.annotate(round(thr[k],2),xy=(i,j), textcoords='data')
        k+=1

# Roc Curve
from sklearn.metrics import roc_curve
def roc_plot(model,X_train3,y_train3,X_val,y_val):
    train_prob = model.predict_proba(X_train3)[:,1]
    val_prob = model.predict_proba(X_val)[:,1]
    plt.figure(figsize=(7,7))
    """[y_test, test_prob]"""
    for data in [[y_train3, train_prob],[y_val, val_prob]]: 
        fpr, tpr, threshold = roc_curve(data[0], data[1])
        plt.plot(fpr, tpr)
    annot(fpr, tpr, threshold)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.ylabel('TPR (power)')
    plt.xlabel('FPR (alpha)')
    plt.legend(['train','val'])
    plt.show()        
        


# Return and Plot optimised model        
def opt_plots(opt_model):
    """create a dataframe for optimised model"""
    opt = pd.DataFrame(opt_model.cv_results_)
    cols = [col for col in opt.columns if ('mean' in col or 'std' in col) and 'time' not in col]
    params = pd.DataFrame(list(opt.params))
    """concation of params DF and opt DF"""
    opt = pd.concat([params,opt[cols]],axis=1,sort=False)
    
    """plot optimised model"""
    plt.figure(figsize=[15,4])
    plt.subplot(121)
    sns.heatmap(pd.pivot_table(opt,index='max_depth',columns='min_samples_leaf',values='mean_train_score')*100)
    plt.title('ROC_AUC - Training')
    plt.subplot(122)
    sns.heatmap(pd.pivot_table(opt,index='max_depth',columns='min_samples_leaf',values='mean_test_score')*100)
    plt.title('ROC_AUC - Validation')



























