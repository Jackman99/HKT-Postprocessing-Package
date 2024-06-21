import math

class turbine:
    """
    A class used to represent a turbine

    Attributes
    ----------
    length : float
        length of the turbine [m]
    width : float
        turbine width [m]
    blade_radius : float
        radius of the turbine blade [m]
    axis_of_rotation : str
        axis of roation of the turbine (can be 'horizontal', 'vertical', or 'other')
    tip_speed_ratio : float
        tip speed ratio of the turbine
    hub_height : float
        height of the turbine hub [m]
    
    Methods
    -------
    swept_area()
        Calculates the swept of the turbine
    """

    def __init__(self, length, width, blade_radius, axis_of_rotation, tip_speed_ratio, hub_height):
        """
        Parameters
        ----------
        length : float
            length of the turbine [m]
        width : float
            turbine width [m]
        blade_radius : float 
            radius of the turbine blade [m]
        axis_of_rotation : str
            axis of roation of the turbine (can be 'horizontal', 'vertical', or 'other')
        tip_speed_ratio : float
            tip speed ratio of the turbine
        hub_height : float
            height of the turbine hub [m]
        """

        self.length = length
        self.width = width
        self.blade_radius = blade_radius
        self.axis_of_rotation = axis_of_rotation
        self.tip_speed_ratio = tip_speed_ratio
        self.hub_height = hub_height

    def swept_area(self):
        """Calculates the swept area of the turbine

        Parameters
        ----------
        None

        Returns
        -------
        float
            The swept area of the turbine [m^2]

        """
        if self.axis_of_rotation == 'horizontal':
            return self.blade_radius*math.pi


