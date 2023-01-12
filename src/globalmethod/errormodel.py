import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import  mean_squared_error, accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.dummy import DummyRegressor, DummyClassifier


# The StandardErrorRule class is a model selection method that uses
# cross-validation and the standard error rule to select the best model from
# a list of models with different hyperparameter values. It provides methods
# for fitting, predicting, and evaluating the selected model.
class StandardErrorRule:
    
    """
    Parameters:
    - estimator: The estimator object (e.g. a classifier or regressor) to use.
    - param_name: The name of the hyperparameter to tune.
    - param_val: The values of the hyperparameter to try.
    - is_reg: Whether the task is regression or classification.
    - n_splits: The number of splits to use for cross-validation (default 10).
    - std_dist_to_mean: The multiple of the standard error to add to the mean 
      error (default 1).
    - error_function: A custom error function to use (if not provided, the mean
      squared error or misclassification rate will be used).
    """

    def __init__(self,
                 estimator,
                 param_name: str,
                 param_val: list[float],
                 is_reg: bool,
                 n_splits: int=10,
                 std_dist_to_mean: float=1,
                 error_function=None):
        
        self.n_splits = n_splits
        self.estimator = estimator
        self.param_name = param_name
        self.param_val = param_val
        self.is_reg = is_reg
        self.std_dist_to_mean = std_dist_to_mean
        self.error_function = error_function

        # Set the scoring function
        avg_squared_error = lambda y_true, y_pred: mean_squared_error(y_true,
                                                                      y_pred)
        misclass_rate = lambda y_true, y_pred: 1 - accuracy_score(y_true,
                                                                  y_pred)

        if error_function != None:
            scoring = make_scorer(error_function)
        else:
            if is_reg:
                scoring = make_scorer(avg_squared_error)
            else:
                scoring = make_scorer(misclass_rate)
                
        # Set the grid
        self.grid_search = GridSearchCV(estimator, 
                                        {param_name: param_val},
                                        cv=KFold(n_splits, shuffle=True),
                                        scoring=scoring,
                                        return_train_score=True)

    
    def fit_cv(self, X: np.ndarray, y: np.array) -> "StandardErrorRule":
        # Fit a grid search object to the input data
        self.grid_search.fit(X, y)
        
        # Store the mean cross-validation errors and mean standard errors
        # for each model in the grid search
        self.cv_mean_errors_ = self.grid_search.cv_results_["mean_test_score"]
        self.cv_mean_std_errors_ = (self.grid_search.cv_results_["std_test_score"] /
                                    np.sqrt(self.n_splits - 1))
        
        # Set the model with the smallest mean error as the best model and
        # store its ID, minimum error, and minimum standard error
        self.best_model_id_ = np.argmin(self.cv_mean_errors_)
        self.cv_min_error_ = self.cv_mean_errors_[self.best_model_id_]
        self.cv_min_std_error_= self.cv_mean_std_errors_[self.best_model_id_]
        
        # Define a threshold as the sum of the minimum error and a multiple
        # of the minimum standard error
        threshold = (self.cv_min_error_ +
                     self.std_dist_to_mean * self.cv_min_std_error_)
        
        # Select the model with the maximum mean error that is less than this
        # threshold and store its ID
        self.select_model_id_ = np.argmax(self.cv_mean_errors_ < threshold)
        
        # Fit the selected model and return it
        return self.fit_select_model(X, y)
    
    
    # Return the class instance of the selected mode
    def fit_select_model(self, X, y) -> "StandardErrorRule":
        # Set the model to the estimator specified in the class
        self.model_ = self.estimator
        
        # Set the hyperparameter value for the selected model
        self.model_.set_params(**{self.param_name: 
                                  self.param_val[self.select_model_id_]})
        
        # Fit the model to the input data and labels
        self.model_.fit(X, y)
        
        # Return the class instance
        return self
    
    
    # Use the fitted model to make predictions on new data
    def predict(self, X: np.ndarray) -> np.ndarray:
    
        predictions = self.model_.predict(X)
        
        # Return the predictions as a 1D array 
        return np.squeeze(predictions)


    # Evaluate the model's performance on new data.
    def score(self, X: np.ndarray, y: np.array) -> (float, float):
        # Make predictions using the fitted model
        y_pred = self.predict(X)
        
        if y.ndim == 2:
            warnings.warn("y is 2d so numpy.squeeze has been applied on y")
            y = np.squeeze(y)
        
        # Calculate the errors
        if self.error_function != None:
            errors = self.error_function(y, y_pred)
        else:
            if self.is_reg:
                # For regression, use mean squared error
                errors = (y - y_pred)**2
            else:
                # For classification, use misclassification rate
                errors = (y != y_pred) * 1
        
        # Calculate the mean error and the standard error
        mean_error = np.mean(errors)
        standard_error = np.std(errors, ddof=1) / np.sqrt(len(y))
        
        return mean_error, standard_error
    
    
    # Plot the mean error and standard deviation of the error for different
    # values of the hyperparameter. The error bars represent one standard
    # deviation of the error. The dashed line represents the optimal
    # hyperparameter value, as determined by the Standard Error Rule.
    # The vertical dashed line indicates the selected hyperparameter value.
    def plot_rule_selection(self, complexities: list=None,
                            X: np.ndarray=None, y: np.array=None):

        # Use the list of hyperparameter values specified in the object
        # if none are provided
        if complexities is None:
            complexities = self.param_val
            
        # If training data is provided, fit a dummy model as a baseline
        if (X is not None) and (y is not None):
            
            baseline = StandardErrorRule(DummyRegressor() if self.is_reg else DummyClassifier(),
                                         "constant", [None], self.is_reg).fit_cv(X, y)
            
            # Concatenate the errors from the dummy model with the errors from
            # the actual model
            mean_error = np.hstack((baseline.cv_mean_errors_, self.cv_mean_errors_))
            std_error = np.hstack((baseline.cv_mean_std_errors_, self.cv_mean_std_errors_)) 
            
            # Add one to the indices to account for the dummy model
            best_model_id = self.best_model_id_ + 1
            select_model_id = self.select_model_id_ + 1
            
        else:
            # If no training data is provided, use the errors from the actual
            # model only
            mean_error = self.cv_mean_errors_
            std_error = self.cv_mean_std_errors_
            best_model_id = self.best_model_id_
            select_model_id = self.select_model_id_
            
            
        # Set up the plot
        fig, ax = plt.subplots(figsize=(4, 3))

        # Plot the mean error as a function of the hyperparameter value
        ax.plot(complexities, mean_error, color="orange")
        
        # Add error bars representing one standard deviation of the error
        ax.errorbar(complexities , mean_error, yerr=std_error, ecolor="blue",
                    elinewidth=0.5, capsize=3, markersize=4, marker='o',
                    color="orange")
        
        # Add a horizontal line at the optimal hyperparameter value
        ax.axhline(mean_error[best_model_id] + std_error[best_model_id],
                   color="black", linewidth=0.5,
                   linestyle="--")
        
        # Add a vertical line at the selected hyperparameter value
        ax.axvline(complexities[select_model_id],
                   color="black", linewidth=0.5,
                   linestyle="--")

        return ax