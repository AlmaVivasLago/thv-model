import numpy as np

    
def _define_profile(best_bounding_box_roi):
    
    y_bbox, x_bbox = np.where(best_bounding_box_roi != 0)
    
    x_bbox, y_bbox  = np.array(x_bbox), np.array(y_bbox) 
    x_bbox_l, y_bbox_l  = list(x_bbox), list(y_bbox) 
    
    # # ======= Profile limits (and guide) ========
    c_x_ymin, c_ymin =  x_bbox_l[np.where(y_bbox_l == np.min(y_bbox_l))[0][0]], np.min(y_bbox_l)
    c_xmin, c_y_xmin =  np.min(x_bbox_l), y_bbox_l[np.where(x_bbox_l == np.min(x_bbox_l))[0][0]]
    
    c_xmax, c_y_xmax = np.max(x_bbox_l),  y_bbox_l[np.where(x_bbox_l == np.max(x_bbox_l))[0][0]]
    
    # # ======= Profile lenght  ========
    length_h_approx =  np.abs(c_y_xmax -  c_ymin)

    # # ======= BBox intersection with key point fitted line ========
    x_bbox, y_bbox = np.array([c_x_ymin, c_xmax]), np.array([c_ymin, c_y_xmax])
    A_bbox = np.vstack([x_bbox, np.ones(len(x_bbox))]).T
    # #-----------------
    m_bbox_dir, c_bbox_dir =  np.linalg.lstsq(A_bbox, y_bbox, rcond=None)[0]
    

    init_x  = np.array([c_xmin, c_x_ymin])
    init_y  = np.array([c_y_xmin, c_ymin])
    
    return init_x, init_y, m_bbox_dir, length_h_approx
def _get_line_profile_coordinates(src, dst, linewidth=1):
    """Return the coordinates of the profile of an image along a scan line.
    Parameters
    ----------
    src : 2-tuple of numeric scalar (float or int)
        The start point of the scan line.
    dst : 2-tuple of numeric scalar (float or int)
        The end point of the scan line.
    linewidth : int, optional
        Width of the scan, perpendicular to the line
    Returns
    -------
    coords : array, shape (2, N, C), float
        The coordinates of the profile along the scan line. The length of the
        profile is the ceil of the computed length of the scan line.
    Notes
    -----
    This is a utility method meant to be used internally by skimage functions.
    The destination point is included in the profile, in contrast to
    standard numpy indexing.
    """
    src_row, src_col = src = np.asarray(src, dtype=float)
    dst_row, dst_col = dst = np.asarray(dst, dtype=float)
    d_row, d_col = dst - src
    theta = np.arctan2(d_row, d_col)

    length = int(np.ceil(np.hypot(d_row, d_col) + 1))

    line_col = np.linspace(src_col, dst_col, length)
    line_row = np.linspace(src_row, dst_row, length)

    col_width = (linewidth - 1) * np.sin(-theta) / 2
    row_width = (linewidth - 1) * np.cos(theta) / 2
    
    perp_rows = np.stack([np.linspace(row_i - row_width, row_i + row_width,
                                      linewidth) for row_i in line_row])
    perp_cols = np.stack([np.linspace(col_i - col_width, col_i + col_width,
                                      linewidth) for col_i in line_col])
    
    return np.stack([perp_rows, perp_cols])

def get_profile(search_roi_image=None, best_roi_bbox=None): 

    init_x, init_y, m_bbox_dir, length_h_approx = _define_profile(best_roi_bbox) 
    
    full_profile, index = [], []
   
    intensity_profile_results =  {
                                    'index': index,
                                    'full_profile': full_profile
                                }
    
    if m_bbox_dir==0:ratio_p = 0
    else: ratio_p = 1/m_bbox_dir
    
    storage_temp = []
    for n_line in range(length_h_approx):
        start = (init_y[0]+n_line, init_x[0]+(ratio_p*n_line))
        end  = (init_y[1]+n_line, init_x[1]+(ratio_p*n_line))
        
        if (np.array([start[1], end[1]]) >= search_roi_image.shape[1]-2).any() | (np.array([start[0], end[0]]) >= search_roi_image.shape[0]-2).any():

            break
           
        else:
            storage_temp.append([start, end])

            p_sample = _get_line_profile_coordinates(start, end)

            pf_x = (np.round(p_sample[1], 0)).astype(int)
            pf_y = (np.round(p_sample[0], 0)).astype(int)

            p_sample_line_extrema = [search_roi_image[pf_y[n], pf_x[n]][0] for n in range(len(pf_x)-2)]

            index.append(np.array([[pf_y[n][0], pf_x[n][0]] for n in range(len(pf_x)-2)]))

            full_profile.append(np.array(p_sample_line_extrema))

    return intensity_profile_results