import numpy as np

def sanity_check(left_line, right_line, left_fit, right_fit, left_roc, right_roc, ldiff, rdiff):
    """
    Sanity check of whether the detected lines are valid or not
    1) Check if the distances from the centers are roughly equal
    2) Check if both are > 200
    3) Check if they are parallel
    
    Returns false if any of the conditions are not met
    
    """
    if left_line.line_base_pos is not None and right_line.line_base_pos is not None:
        if not np.allclose(ldiff, left_line.line_base_poc, atol=1e-4) or not np.allclose(rdiff, right_line.line_base_poc, atol=1e-4):
            return False
    
    if left_roc < 200 or right_roc < 200:
        return False
    
    left_slope = left_fit[0]
    right_slope = right_fit[0]
    if abs(right_slope - left_slope) > 1e-2:
        return False
    return True