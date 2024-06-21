import glob
import ADV_Point_Class_final as ADV_Point_Class_final
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

class ADV_Grid(object):
    """
    A class used to represent a grid of ADV points

    ...

    Attributes
    ----------
        path : str
            The path to the folder with the ADV points
        turbine : Turbine_Class object
            Turbine_Class object associated with these points
        Flume : Flume_Class object
            Flume_Class object where the measurements were taken
        points_list :
            A list of ADV_Point objects

    Methods 

    mPST_all()
        Despikes all points in the grid

    plot_profiles(planes, parameterName = "umean", parameterlabel = r'$U/U_{hub}$', view = 'topdown', normalize = True, contraction = False, fontsize = 26, num_colors = 13, markersize = 13, linewidth = 5, colors = 'distinct', figsize = (18, 9))
        Plots the profiles of a given parameter for a list of planes in the grid

    check_height_grid(threshold)
        Checks the height of all points in the grid

    check_length_grid(expected)
        Checks the length of all points in the grid

    get_values_positions_for_plane(plane, midheight = 17, parameterName = 'umean', view = 'topdown', contraction = False, normalize = True, Z = 0, Y = 0, keep_positions = [])
        Returns the values and positions for a given plane

    calcNorm(parameterName='umean', plane=-2, midheight=17)
        Calculates the normalization factor

    reverse_list(input_list)
        Reverses a list

    sort_two_lists(list1, list2)
        Sorts two lists

    get_color_gradient(color1 = (0, 0, 0), color2 = (1, 1, 1), num_colors = 10)
        Returns a color gradient

    get_distinct_colors(num_colors = 10)
        Returns a list of distinct colors
    -------

    """

    def __init__(self, path, grid_name, turbine, flume):
        """
        Parameters
        ----------
        path : str
            The path to the folder with the ADV points
        turbine : Turbine_Class object
            Turbine_Class object associated with these points
        Flume : Flume_Class object
            Flume_Class object where the measurements were taken
        """
        self.path = path
        self.grid_name = grid_name
        self.turbine = turbine
        self.flume = flume
        self.points_list = []  # create empty array for points

        # Populate the points_list list with ADV_Point objects from the folder
        for dat, hdr in zip(glob.glob(path + "/*.dat"), glob.glob(path + "/*.hdr")):
            # Make sure all of the Header Files are good
            try:
                self.points_list.append(ADV_Point_Class_final.ADV_Point(dat, hdr))
            except ValueError:
                raise Exception("Could not read {}".format(hdr))

    """Despiking Methods (all points)"""

    def mPST_all(self, method = 'mPST' ):
        '''This method despikes all points in the grid.
        
        Arguements:
        method (str): the method to be used for despiking (default is 'mPST')
        
        Returns:
        None
        '''

        if method == 'mPST':
            for point in self.points_list:
                try:
                    point.mPST()
                    print("mPST worked for " + point.point_name)
                except:
                    print("mPST did not work for " + point.point_name)

    """Graphing Methods (all points)"""
    def plot_profiles(self, planes, parameterName = "umean", parameterlabel = r'$U/U_{hub}$', view = 'topdown', normalize = True, contraction = False, fontsize = 26, num_colors = 13, markersize = 13, linewidth = 5, colors = 'distinct', figsize = (18, 9)):
        '''This method plots the profiles of a given parameter for a list of planes in the grid. 

        Arguements:
            planes (list): the list of planes to be plotted
            parameterName (str): the parameter to be plotted (default is "umean") other options: 'vmean', 'wmean', 'tke', 'urms', 'vrms', 'wrms', 'turb_intensity', 'reynolds_stressesuv', 'reynolds_stressesuw', 'reynolds_stressesvw'
            parameterlabel (str): the label for the parameter (default is r'$U/U_{hub}$') other options: r'$\overline{{u\'w\'}}/\overline{u}_{\text{hub}}^2$' #r'$\Delta U_x$' r'$k/U_{hub}^2$' r'$U/U_{hub}$'
            view (str): the view of the plot (default is 'topdown') other options: 'side'
            normalize (bool): whether or not to normalize the values (default is True)
            contraction (bool): whether or not to apply the contraction factor (default is False)
            fontsize (int): the fontsize of the plot (default is 26)
            num_colors (int): the number of colors to be used in the plot (default is 13)
            markersize (int): the size of the markers in the plot (default is 13)
            linewidth (int): the width of the lines in the plot (default is 5)
            colors (str): the type of colors to be used in the plot (default is 'distinct') other options: 'gradient'
            figsize (tuple): the size of the figure (default is (18, 9))
        
        Returns:
        The plot of the profiles for the given planes.'''
        
        plt.figure(figsize=figsize)

        plt.xticks(fontsize=fontsize - 6)  # Adjust the fontsize as needed
        plt.yticks(fontsize=fontsize - 6)  # Adjust the fontsize as needed
        plt.rcParams['axes.linewidth'] = linewidth  # Set the linewidth for all axes
        font = {'size'   : fontsize}
        matplotlib.rc('font', **font)
        matplotlib.rc('xtick', labelsize=20) 
        matplotlib.rc('ytick', labelsize=30)

        if view == 'topdown':
            xlabel = 'Y/D'
            ylabel = parameterlabel
            marker = '^'
            plt.gca().invert_xaxis()
            if normalize:
                plt.axvline(x=-0.5, color='k', linestyle='--', linewidth = 2)
                plt.axvline(x=0.5, color='k', linestyle='--', linewidth = 2)
            else:
                plt.axvline(x=-self.turbine.blade_radius, color='k', linestyle='--', linewidth = 2)
                plt.axvline(x=self.turbine.blade_radius, color='k', linestyle='--', linewidth = 2)
        else:
            ylabel = 'Z/D'
            xlabel = parameterlabel
            marker = '>'
            if normalize:
                plt.axhline(y=self.turbine.hub_height/(2*self.turbine.blade_radius), color='k', linestyle='--', linewidth = 4)
                plt.axhline(y=self.turbine.hub_height/(2*self.turbine.blade_radius)+0.5, color='k', linestyle='--', linewidth = 2)
                plt.axhline(y=self.turbine.hub_height/(2*self.turbine.blade_radius)-0.5, color='k', linestyle='--', linewidth = 2)
            else:
                plt.axhline(y=self.turbine.hub_height, color='k', linestyle='--', linewidth = 4)
                plt.axhline(y=self.turbine.hub_height + self.turbine.blade_radius, color='k', linestyle='--', linewidth = 2)
                plt.axhline(y=self.turbine.hub_height - self.turbine.blade_radius, color='k', linestyle='--', linewidth = 2)
        
        plt.title(self.grid_name.split(' ')[0])
        plt.grid()
        plt.xlabel(xlabel, fontsize = fontsize)
        plt.ylabel(ylabel, fontsize = fontsize)
        plt.tick_params(axis='both', which='major', labelsize=fontsize)
        plt.tick_params(axis='both', which='minor', labelsize=fontsize)

        if colors == 'distinct':
            colors = self.get_distinct_colors(num_colors = num_colors)
        else:  
            colors = self.get_color_gradient(num_colors = num_colors)

        for plane, color in zip(planes, colors):
               
                points_turbine, positions_turbine = self.get_values_positions_for_plane(plane, midheight = self.turbine.hub_height, 
                                                                            parameterName = parameterName, view = view, contraction = contraction, normalize = normalize)
                if points_turbine != []:
                    if view == 'topdown':
                        plt.plot(positions_turbine, points_turbine, color = color, marker = marker, markersize=markersize, linewidth = linewidth, label = str(plane))
                    else:
                        plt.plot(points_turbine, positions_turbine, color = color, marker = marker, markersize=markersize, linewidth = linewidth, label = str(plane))
        plt.legend(fontsize = fontsize - 10, title = 'X/D', title_fontsize = fontsize - 6)

        plt.tight_layout()
        plt.show()

    """Checking Functions"""

    def check_height_grid(self, allowableHeightDifference):
        '''Compare actual height from ADV (z_data) with the desired height (Z) for each of the points in the grid
        
        Arguements:
        allowableHeightDifference (float): the allowable difference between the actual height from ADV (z_data) with the desired height (Z)
        
        Returns:
        Boolean: True if the height is within the allowable height difference, False otherwise'''
        for point in self.points_list:
            point.check_height(allowableHeightDifference)

    def check_length_grid(self, expectedNumPoints):
        '''Compare actual number of data points in the time series with the expected number of data points, i.e. expectedNumPoints for all points in the grid.

        Arguements:
        expectedNumPoints (int): the expected number of data points in the ADV time series

        Returns:
        Boolean: True if the length of the time series is greater than the expected number of data points, False otherwise
        '''
        for point in self.points_list:
            point.check_length(expectedNumPoints)

    '''Helper Functions'''
    def get_values_positions_for_plane(self, plane, midheight = 17, parameterName = 'umean', view = 'topdown', contraction = False, normalize = True, Z = 0, Y = 0, keep_positions = []):
            values = []
            positions = []

            for point in self.points_list:

                # Topdown Case
                if (plane == point.X) and (point.Z == midheight) and (view == 'topdown'):
                    values.append(point.getParameter(parameterName))
                    positions.append(point.Y) 

                # Side Case
                if (plane == point.X) and (point.Y == 0) and (view == 'side'):
                    values.append(point.getParameter(parameterName))
                    positions.append(point.Z)

                # Flow Recovery Case
                if ((view == 'flowrecovery') and (point.X == plane) and (point.Z == Z) and (point.Y == Y)): 
                    values.append(point.getParameter(parameterName))
                    positions.append(point.X)
        

            positions, values  = self.sort_two_lists(positions, values)
            positions = self.reverse_list(positions)
            values = self.reverse_list(values)

            norm = self.calcNorm(parameterName=parameterName, plane=-2, midheight=self.turbine.hub_height)

            if normalize:
                values = [element / norm for element in values]
                if view != 'flowrecovery': # don't normalize positions for flow recovery, they're already normalized (X)
                    positions = [element / (2*self.turbine.blade_radius) for element in positions]

            return values, positions

    def calcNorm(self, parameterName='umean', plane=-2, midheight=17):
        norm = 0
        
        # Find the normalization factor
        for point in self.points_list:
            if (point.X == plane) and (point.Z == midheight) and (point.Y == 0):
                norm = point.meanVel[0]
            elif (point.X == plane) and (point.Z == midheight + 1) and (point.Y == 0):
                # 26 is technically hub height but it was corrupted
                norm = point.meanVel[0]
        
        # Calculate normalization based on parameterName
        if parameterName in ['tke', 'turb_intensity', 'reynolds_stressesuv', 'reynolds_stressesuw', 'reynolds_stressesvw']:
            return norm * norm
        else:
            return norm
    
    def reverse_list(self, input_list):
        return input_list[::-1]

    def sort_two_lists(self, list1, list2):
        if not list1 or not list2 or len(list1) != len(list2):
                return [], []  # Return empty lists if either input list is empty or their lengths don't match

        zipped_lists = zip(list1, list2)
        sorted_lists = sorted(zipped_lists, reverse=True)

        sorted_list1, sorted_list2 = zip(*sorted_lists)
        return list(sorted_list1), list(sorted_list2) 

    def get_color_gradient(self, color1 = (0, 0, 0), color2 = (1, 1, 1), num_colors = 10): # default color1 is black, color2 is white (greyscale)
        # create a list of num_colors colors
        colors = []
        for i in range(num_colors):
            # calculate new color
            new_color = [color1[j] + (color2[j] - color1[j]) * i / (num_colors - 1) for j in range(3)]
            colors.append(new_color)
        return colors

    def get_distinct_colors(self, num_colors = 10):
        # generate a list of distinct colors
        distinct_colors = plt.cm.get_cmap('tab20', num_colors)
        return [distinct_colors(i) for i in range(num_colors)]