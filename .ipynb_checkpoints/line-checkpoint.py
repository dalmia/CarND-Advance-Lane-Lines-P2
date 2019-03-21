from collections import deque
import numpy as np

class Line():
    """
    Line class that stores various intermediate variables
    when working with videos.
    
    """
    def __init__(self):
        #number of iterations to keep
        self.n = 10
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        
        self.reset()
    
    def reset(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = deque([], maxlen=self.n) 
        #polynomial coefficients for the n most recent fit
        self.current_fit = []
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None  
        
    
    def add_detection(self, polynomial, xfit, roc, offset, x, y):
        self.current_fit = polynomial
#         self.best_fit = np.mean(current_fit, axis=0)
        self.detected = True
        self.recent_xfitted.append(xfit)
        self.bestx = np.mean(self.recent_xfitted, axis=0)
        self.radius_of_curvature = roc
        self.left_base_pos = offset
        self.allx = x
        self.ally = y