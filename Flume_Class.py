class flume:
    """
    A class used to represent a Flume

    ...

    Attributes
    ----------
    depth : float
        flume water depth [m]
    width : float
        flume width [m]
    length : float
        flume length [m]

    Methods
    -------
    area()
        Calculates the cross sectional area of the flume
    """

    def __init__(self, depth, width, length):
        """
        Parameters
        ----------
        depth : float
            flume water depth [m]
        width : float
            flume width [m]
        length : float
            flume length [m]
        """
        
        self.depth = depth
        self.width = width
        self.length = length

    def area(self):
        """Calculates the cross sectional area of the flume

        Parameters
        ----------
        None

        """
        return self.width*self.depth


