import numpy as np

import scipy.interpolate
from scipy.signal import savgol_filter

from PyAstronomy import pyasl

def allocate_border(full_profile, index, W_R):
    #TOREFACTOR - (pathetic loops)
    border_w = 1

    left_borders =  []
    right_borders = []
    
    line_indexs = []
    offset_profile = 0
    interpolated_extrema = {}

    if full_profile:
        profile_lenght = len(full_profile)

        prcnt = 0.5

        for strut_line in range(profile_lenght):
            strut = np.where(np.array(full_profile[strut_line]) != 0.0)[0]

            if len(strut) >=2:
                init_strut, end_strut = strut[0], strut[-1]

                if end_strut-init_strut <= W_R*prcnt:
                    offset_profile += 1
                    continue

                else:
                    line_index = index[strut_line]
                    line_indexs.append(line_index)

                    left_borders.append([line_index[coords] for coords in range(init_strut, init_strut+border_w)])
                    right_borders.append([line_index[coords] for coords in range(end_strut-border_w, end_strut)])
        if len(list((np.array(sum(left_borders, [])).shape)))==2 or len(list((np.array(sum(left_borders, [])).shape)))==2:    
            
            left_borders_xs = np.array(sum(left_borders, []))[:, 1]
            left_borders_ys = np.array(sum(left_borders, []))[:, 0]

            right_borders_xs = np.array(sum(right_borders, []))[:, 1]
            right_borders_ys = np.array(sum(right_borders, []))[:, 0]

            keypoints_ip = {}
            mode = 'both'
            if len(list(left_borders_xs))>=2 and len(list(right_borders_xs))>=2:    

                iin_lb, iout_lb = pyasl.polyResOutlier(left_borders_xs, left_borders_ys, deg=1, mode=mode, stdlim=3, controlPlot=False);

                total_n_l_out = 0
                total_n_l_inn = 0

                total_n_l_out += len(iout_lb)
                total_n_l_inn += len(iin_lb)

                left_borders_ys_new, left_borders_xs_new = left_borders_ys[iin_lb], left_borders_xs[iin_lb];
                if np.std(left_borders_ys_new) >= 35:

                    iin_lb, iout_lb = pyasl.polyResOutlier(left_borders_xs_new, left_borders_ys_new, deg=1, mode=mode, stdlim=2, controlPlot=False);

                    total_n_l_out += len(iout_lb)
                    total_n_l_inn += len(iin_lb)

                    left_borders_ys_new, left_borders_xs_new = left_borders_ys_new[iin_lb], left_borders_xs_new[iin_lb];


                keypoints_ip.update({'LeftBorder': [left_borders_xs_new, left_borders_ys_new]})

                iin_rb, iout_rb = pyasl.polyResOutlier(right_borders_xs, right_borders_ys, deg=1, stdlim=3, mode=mode, controlPlot=False)
                total_n_r_out = 0
                total_n_r_inn = 0

                total_n_r_out += len(iout_rb)
                total_n_r_inn += len(iin_rb)

                right_borders_ys_new, right_borders_xs_new = right_borders_ys[iin_rb], right_borders_xs[iin_rb]

                if np.std(right_borders_ys_new) >= 35:
                    iin_rb, iout_rb = pyasl.polyResOutlier(right_borders_xs_new, right_borders_ys_new, deg=1, stdlim=2, mode=mode, controlPlot=False)

                    total_n_r_out += len(iout_rb)
                    total_n_r_inn += len(iin_rb)

                    right_borders_ys_new, right_borders_xs_new = right_borders_ys_new[iin_rb], right_borders_xs_new[iin_rb]

                keypoints_ip.update({'RightBorder':[right_borders_ys_new, right_borders_xs_new]})

                # defining arbitrary parameter to parameterize the curve
                path_x_right = np.array(right_borders_xs_new)
                path_x_left = np.array(left_borders_xs_new)

                path_y_right = np.array(right_borders_ys_new)
                path_y_left = np.array(left_borders_ys_new)

                paths_x = [path_x_right, path_x_left]
                paths_y = [path_y_right, path_y_left]
                check = np.array([check.shape[0] for check in sum([paths_x, paths_y], [])])
                if (check>0).all():

                    refs = ['right', 'left']

                    ref = 0

                    for paths in zip(paths_x, paths_y):
                        path_t = np.linspace(0, 1, paths[0].size)

                        r = np.vstack((paths[0].reshape((1, paths[0].size)), paths[1].reshape((1, paths[1].size))))

                        spline = scipy.interpolate.interp1d(path_t, r, kind='nearest')

                        t = np.linspace(np.min(path_t), np.max(path_t), 100)

                        r = spline(t)

                        filtered = savgol_filter(r, 11, 1) #Convolve two N-dimensional arrays.
                        interpolated_extrema.update({f'{refs[ref]}_original': r,
                                                    f'{refs[ref]}_filtered': filtered})
                        ref += 1 
        
    return interpolated_extrema