import math
import statistics as stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels import robust
import dask.dataframe as dd
from scipy.interpolate import CubicSpline
from scipy.signal import welch
import matplotlib.patches as patches


class ADV_Point(object):
    """
    A class used to represent an ADV point
    ...

    Attributes
    ----------
    Vel : float 4 x number of samples
        The velocity time series

    Amp : float 4 x number of samples
        The amplitude time series

    Snr : float 4 x number of samples
        The signal to noise ratio time series

    Cor : float 4 x number of samples
        The correlation time series

    point_name : str
        The name of the ADV point Ex: 'A1_06cm_40cm'

    freq : float
        The frequency sampled by the Vectrino

    temp : float
        The tempurature, measured by the Vectrino

    z_data : float
        The height of the point, measured by the Vectrino

    Z : float
        The height of the point, comes from its name

    Methods
    -------

        Despiking Methods
        -------
            Zhong_Despike()
                TODO

            Robust_Estimation()
                TODO

            Phase_Space_Thresholding_Single_Component()
                TODO

            Phase_Space_Thresholding()
                TODO

            mPST()
                TODO

            Velocity_Correlation_Filter()
                TODO

            PS_Hybrid_Method()
                TODO

            Kernel_Density(u, hx=0.01, hy=0.01):
                TODO

        Despiking Helper Methods
        -------
            in_Ellipse(x, y, majorAxis, minorAxis, center=[0, 0], angle=0)
                TODO

            getVelocity_Correlation_FilterAxis(U, V)
                TODO

            getDel2U(U, method="CD_O1")
                TODO

            getDelU(U, method="CD_O1")
                TODO

            convergence_test(var, confidence)
                TODO

            calc_Vel_fluc()
                TODO

            cutoff1(dp, uf, c1, c2, f, Ip)
                TODO

        Time Series Checking Methods
        -------
            check_height(threshold)
                TODO

            height_difference()
                TODO

            check_length(expected)
                TODO

        Reader Methods
        -------
            read_freq_z_temp(self, info_indexes = [9, 112, 117])
                TODO

            dataframe_to_list(dataframe)
                TODO

            extract(list, num_col=1)
                TODO

    """

    def __init__(self, dat_file, hdr_file):
        """
        Arguments
        ----------
        dat_file : str
            The .dat file.
        hdr_file : str
            The The .hdr file.
        """
        self.dat_file = dat_file
        self.hdr_file = hdr_file

        # These Methods read the relevant data from the .dat and .hdr files and sets it to the attributes
        self.read_vel_amp_snr_cor()
        self.read_freq_z_temp()  # This method reads the  sampling frequency, measured height from the bed (z), and temp from the .hdr file
        self.read_point_name()  # This method can be modified for the specific naming convention of the data
        self.read_xyz_position()

        # These methods calculate parameters of interest
        self.calc_mean()
        self.calc_rms()
        self.calc_turb_intensity()
        self.calc_reynolds_stresses()
        self.calc_tke()
        self.calc_auto_corr()
        self.calc_integral_time_scale()

        # This parameter tracks the number of points that were replaced during the despiking process
        self.numReplaced = np.zeros(len(self.Vel))

    """Reader Methods"""

    def read_freq_z_temp(self, info_indexes=[9, 112, 117]):
        """Reads the frequency, height from the bed, and temperature from the .hdr file

        Arguments
        ----------
        info_indexes: 1x3 int (optional)
            These are the indexes of the rows in the .hdr file where the sampling frequency, height, and temperature are contained. default is [9, 112, 117]
        """
        # Make sure the .hdr file is good
        try:
            file_inf = dd.read_csv(
                self.hdr_file,
                header=None,
                assume_missing=True,
                skiprows=[i for i in range(max(info_indexes)) if i not in info_indexes],
            ).to_dask_array(lengths=True)
            fr_i = int(file_inf.blocks[0][0].compute()[0].split(" ")[-2])
            z_data = round(
                float(file_inf.blocks[0][1].compute()[0].split(" ")[-2]) - 0.05, 5
            )  # Substract 5cm from ADV position
            temp = float(file_inf.blocks[0][2].compute()[0].split(" ")[-3])
            print("Succuessfully read {}".format(self.hdr_file))
        except ValueError:
            print("Could not read {}".format(self.hdr_file))
            fr_i = None
            z_data = None
            temp = None

        self.freq, self.z_data, self.temp = fr_i, z_data, temp

    def read_point_name(self):
        """Reads the point name from the data file. It can be modified for the specific naming convention of the data"""
        try:
            self.point_name = self.dat_file.split("\\")[-1][:-4]# this works for the naming convention that ends like '*\-2D26_21cm_40cm.dat' only
            print("Successfully read point name from {}".format(self.dat_file))
        except:
            print("Could not read point name from {}. Change the way the file name is assigned to the point name.".format(self.dat_file))
            self.point_name = None

    def read_xyz_position(self):
        """Reads the x, y, and z position of the point from the point name.
        Can be modified for the specific naming convention of the data
        """
        try:
            self.X = float(self.point_name.split("D")[0])
            self.Y = float(self.point_name.split("D")[1].split("_")[0])
            self.Z = (
                float(self.point_name.split("_")[1].split("c")[0]) - 5
            )  # Substract 5cm from ADV position, same rules as the line above original, Note this is different from the z_data attribute, which is the measured height from the bed
            print("Successfully read x, y, and z position from {}".format(self.point_name))
        except:
            print("Could not read x, y, and z position from {}. Need to change naming convention".format(self.point_name))
            self.X = None
            self.Y = None
            self.Z = None

    def read_vel_amp_snr_cor(self):
        """Reads the velocity, amplitude, signal to noise ratio, and correlations from the .dat file"""
        try:
            X = pd.read_csv(self.dat_file, sep="\s+", header=None)
            self.Vel = [X.iloc[:, 2], X.iloc[:, 3], X.iloc[:, 4], X.iloc[:, 5]]
            self.Amp = [X.iloc[:, 6], X.iloc[:, 7], X.iloc[:, 8], X.iloc[:, 9]]
            self.Snr = [X.iloc[:, 10], X.iloc[:, 11], X.iloc[:, 12], X.iloc[:, 13]]
            self.Cor = [X.iloc[:, 14], X.iloc[:, 15], X.iloc[:, 16], X.iloc[:, 17]]
            print("Successfully read {}".format(self.dat_file))
        except:
            print("Could not read {}".format(self.dat_file))
            self.Vel = None
            self.Amp = None
            self.Snr = None
            self.Cor = None

    """Checks on Data"""

    def check_height(self, allowableHeightDifference):
        """Compare actual height from ADV (z_data) with the desired height (Z)

        Arguments
        ----------
        allowableHeightDifference: float
            the allowable difference between the actual height from ADV (z_data) with the desired height (Z)

        Returns Boolean: true if difference is less than allowableHeightDifference and false if it's greater

        """
        height_difference = abs(
            self.Z - (100 * self.z_data)
        )  # the 100 is to convert from meters to cm

        if height_difference > allowableHeightDifference:
            print(
                f"{self.point_name} has height = {self.z_data}; expected = {self.Z}"
            )
            return False

        return True

    def check_length(self, expectedNumPoints):
        """Compare actual number of data points in the time series with the expected number of data points (expectedNumPoints)

        Arguments
        ----------
        expectedNumPoints: float
            the expected number of data points in the ADV time series

        Returns Boolean: true if actual number of points is greater than expected and false if it's less

        """

        if len(self.Vel[0]) < expectedNumPoints:
            print(
                f"{self.point_name}: Expected length: {expectedNumPoints} Actual Length: {len(self.Vel[0])}"
            )
            return False

        return True

    """Graphing Methods (single point only)"""

    def graphTimeSeries(
        self,
        attributes=["Vel"],
        save=False,
        save_path="",
        fontsize=20,
        linewidth=1.5,
        figsize=(16.5, 10.0)
    ):
        """
        Graphs the time series data for the specified attributes. Attributes can be 'Vel', 'Amp', 'Snr', or 'Cor'.

        Args:
            attributes (list, optional): List of attributes to graph. Defaults to ['Vel'].
            save (bool, optional): Flag indicating whether to save the graph. Defaults to False.
            save_path (str, optional): Path to save the graph. Defaults to ''.
            fontsize (int, optional): Font size for the graph. Defaults to 20.
            linewidth (float, optional): Line width for the graph. Defaults to 1.5.
            figsize (tuple, optional): Figure size for the graph. Defaults to (16.5, 10.0).
        """
        if attributes == ["Vel"]:
            ylabels = ["U (m/s)", "V (m/s)", "W1 (m/s)", "W2 (m/s)"]
        elif attributes == ["Amp"]:
            ylabels = ["X Amp", "Y Amp", "Z1 Amp", "Z2 Amp"]
        elif attributes == ["Snr"]:
            ylabels = ["X Snr", "Y Snr", "Z1 Snr", "Z2 Snr"]
        elif attributes == ["Cor"]:
            ylabels = ["X Cor", "Y Cor", "Z1 Cor", "Z2 Cor"]
        else:
            print(
                "One or more of the attributes you entered does not exist in this object"
            )
            return

        fig, axs = plt.subplots(
            len(getattr(self, attributes[0])),
            sharex=True,
            figsize=figsize,
            constrained_layout=True,
        )
        fig.suptitle(f"{self.point_name} Time Series", fontsize=fontsize + 2)

        for i, attr in enumerate(getattr(self, attributes[0])):
            axs[i].plot(attr, linewidth=linewidth)
            axs[i].set_ylabel(ylabels[i], fontsize=fontsize)
            axs[i].tick_params(axis="x", labelsize=fontsize - 2)
            axs[i].tick_params(axis="y", labelsize=fontsize - 2)

        axs[i].set_xlabel("t (counts)", fontsize=fontsize)

        if save:
            plt.savefig(f"{save_path}{self.point_name}_TimeSeries.png")

    def graphFreqSpec(
        self, divNperseg=30, figsize=(16.5, 10.0), fontsize=20, linewidth=1.5
    ):
        """
        Graphs the frequency spectrum of velocity components.

        Args:
        - divNperseg (int): nperseg is length of each segment; divNperseg divides the length of the time series. Default is 30.
        - figsize (tuple): Figure size in inches (width, height). Default is (16.5, 10.0).
        - fontsize (int): Font size for the title and axis labels. Default is 20.
        - linewidth (float): Line width for the plotted lines. Default is 1.5.
        """
        fig, axs = plt.subplots(
            4, sharex=True, figsize=figsize, constrained_layout=True
        )
        fig.suptitle(f"{self.point_name} Frequency Spectrum", fontsize=fontsize + 2)
        for i, vel_comp in enumerate(self.Vel):
            f, S = welch(
                vel_comp,
                fs=self.freq,
                nperseg=len(vel_comp) / divNperseg,
                scaling="density",
            )
            axs[i].loglog(f, S, linewidth=linewidth)
            axs[i].set_ylabel(r"$\Phi$", fontsize=fontsize)
            axs[i].tick_params(axis="x", labelsize=fontsize - 2)
            axs[i].tick_params(axis="y", labelsize=fontsize - 2)
        axs[i].set_xlabel("f (Hz)", fontsize=fontsize)

    def graphPhaseSpace(
        self,
        U,
        delU,
        del2U,
        center1,
        center2,
        center3,
        MajorAxisDelU_vs_U,
        MinorAxisDelU_vs_U,
        MajorAxisDel2U_vs_DelU,
        MinorAxisDel2U_vs_DelU,
        MajorAxisDel2U_vs_U,
        MinorAxisDel2U_vs_U,
        angle,
        replacementIndexes,
        numIterations,
        replacementMethod,
        number_of_replaced_points,
        save=False,
        save_path="",
        fontsize=20,
        linewidth=5,
        figsize=(12, 8),
        edgecolor="green",
        pointColor="red",
        spikecolor="blue",
        pointSize=5,
    ):
        '''This method graphs the phase space of the velocity data. It plots the velocity data against the first and second derivatives of the velocity data. It also plots the ellipses that are used in the Phase Space Thresholding method to identify spikes in the data.
        
        Arguments
        ----------
        U : float
            The velocity time series
        delU : 1 x n array
            The time series of the first derivative of the velocity
        del2U : 1 x n array
            The time series of the second derivative of the velocity
        center1 : tuple
            The center of the first ellipse
        center2 : tuple  
            The center of the second ellipse
        center3 : tuple
            The center of the third ellipse
        MajorAxisDelU_vs_U : float
            The major axis of the ellipse for the first plot
        MinorAxisDelU_vs_U : float
            The minor axis of the ellipse for the first plot
        MajorAxisDel2U_vs_DelU : float
            The major axis of the ellipse for the second plot
        MinorAxisDel2U_vs_DelU : float
            The minor axis of the ellipse for the second plot
        MajorAxisDel2U_vs_U : float
            The major axis of the ellipse for the third plot
        MinorAxisDel2U_vs_U : float
            The minor axis of the ellipse for the third plot
        angle : 3 x 1 array
            [0, 0, The angle of rotation of the U vs del2U ellipse]
        replacementIndexes : list
            The indexes of the points that were replaced
        numIterations : int
            The number of iterations
        replacementMethod : str
            The method used to replace the spikes
        number_of_replaced_points : int
            The number of points that were replaced
        save : bool, optional
            Flag indicating whether to save the graph. Default is False
        save_path : str, optional
            The path to save the graph. Default is ''
        fontsize : int, optional
            The font size for the graph. Default is 20
        linewidth : float, optional
            The line width for the ellipse. Default is 5
        figsize : tuple, optional
            The figure size in inches (width, height). Default is (12, 8)
        edgecolor : str, optional
            The color of the ellipse edge. Default is 'green'
        pointColor : str, optional
            The color of the points. Default is 'red'
        spikecolor : str, optional
            The color of the points that were replaced. Default is 'blue'
        pointSize : int, optional
            The size of the points. Default is 5
            '''
        fig, axs = plt.subplots(1, 3, sharey=True, sharex = True, gridspec_kw={'wspace':0.1}, figsize=figsize, constrained_layout=True)

        # patches takes the angles in degrees. Convert from rad to deg here
        angle = math.degrees(angle)

        names = [[r'$u_i$', r'$\Delta u_i$'], [r'$\Delta u_i$', r'$\Delta^2 u_i$'], [r'$u_i$', r'$\Delta^2 u_i$']]

        fig.suptitle(f"Iteration #{numIterations}, Replacement Method: {replacementMethod}, Number Replaced Points: {number_of_replaced_points}", fontsize=fontsize + 2)
        for ax, center, angle, points, axis, names in zip(axs, [center1, center2, center3], [0, 0, angle], [[U, delU], [delU, del2U], [U, del2U]], [[MajorAxisDelU_vs_U, MinorAxisDelU_vs_U], [MajorAxisDel2U_vs_DelU, MinorAxisDel2U_vs_DelU], [MajorAxisDel2U_vs_U, MinorAxisDel2U_vs_U]], names):
            ax.autoscale(enable=True, axis="both", tight=None)
            ax.xaxis.set_tick_params(labelsize=fontsize - 2)
            ax.yaxis.set_tick_params(labelsize=fontsize - 2)
            ax.scatter(points[0], points[1], c=pointColor, s=pointSize)
            ax.scatter(points[0][replacementIndexes], points[1][replacementIndexes], c=spikecolor, s=pointSize)
            # Plot the ellipse
            ellipse = patches.Ellipse(xy=(center[0], center[1]), width=2 * axis[0], height=2 * axis[1], angle=angle, edgecolor=edgecolor, fill=False, linewidth=linewidth)
            ax.add_patch(ellipse)
            ax.set_xlabel(names[0], fontsize=fontsize)
            ax.set_ylabel(names[1], fontsize=fontsize)

        if save:
            plt.savefig(f"{save_path}{self.point_name}_PhaseSpace.png")

        plt.show()
        return

    """Despiking Methods"""

    def Phase_Space_Thresholding_SingleComponent(
        self,
        U,
        replacementMethod="CubicSpline",
        scaleEstimator="SD",
        safeIndexes=[],
        maxNumIterations=20,
        derivativeMethod="CD_O2",
        showPhaseSpace=False,
    ):
        """This is the Phase Space Thesholding Method to despike ADV data.
        It was first proposed by Nikora and Goring (2002), then improved by Wahl (2003)

        Arguments
        ----------
        U : float
            The velocity time series
        replacementMethod : str, optional
            The method used to replace the spikes. Default is 'CubicSpline'
        scaleEstimator : str, optional
            The method used to estimate the scale. Default is 'SD' for standard deviation. Other options are 'MAD' for median absolute deviation
        safeIndexes : list, optional
            The indexes of the points that should not be replaced. Default is []
        maxNumIterations : int, optional
            The maximum number of iterations. Default is 20
        derivativeMethod : str, optional
            The method used to calculate the derivative. Default is 'CD_O2' for Central Difference Order 2. Other options are 'CD_O1' for Central Difference Order 1 and 'FBD' for Forward and Backward Difference
        showPhaseSpace : bool, optional
            Flag indicating whether to show the phase space plot. Default is False

        Returns
        -------
        U : float
            The despiked velocity time series
        len(overallReplacementIndexes) : int
            The number of points that were replaced
        """

        numIterations = 0
        overallReplacementIndexes = [] # this tracks the indices of all the points that were replaced

        while numIterations < maxNumIterations:
            replaceIndexes = []
            # Calculate Delta U and Delta squared U by taking the average of the point in front and behind the point of interest
            # Note: others have suggested better ways to calculate the derivative and second derivative of U
            delU = self.getDelU(U, method=derivativeMethod)
            del2U = self.getDel2U(U, method=derivativeMethod)

            # Calculate Universal Threshold. Parsheh (2010) multiplied this value by a constant
            lamdabda = math.sqrt(2 * (np.log(len(U))))

            # Scale Estimator. This is the number that will multiply lamdabda to create the major
            # and minor axis of the ellipses.  Nikora and Goring (2002) used the standard deviation for this value.
            # Conversely, Wahl (2003) and Parsheh (2010) used the median absolute deviation (theta sub u) multiplied by a constant.

            # Use the Standard Deviation of U, delU, del2U as the scale estimator
            if scaleEstimator == "SD":
                SEU = stats.stdev(U)
                SEdelU = stats.stdev(delU)
                SEdel2U = stats.stdev(del2U)

            # Use the Median absolute deviation of U, delU, del2U as the scale estimator
            if scaleEstimator == "MAD":
                constant2 = 1.483  # Wahl suggested 1.483 for this value, while Parsheh suggested anything within the range of 1.25 to 1.45
                SEU = constant2 * robust.mad(U)
                SEdelU = constant2 * robust.mad(delU)
                SEdel2U = constant2 * robust.mad(del2U)

            # rotation angle of the principal axis of del2U versus U using the cross correlation
            # this is the rotation of the del2U vs U matrix
            theta_rad = np.arctan(sum(np.multiply(U, del2U)) / sum(np.multiply(U, U)))

            # delta ui vs u
            MajorAxisDelU_vs_U = lamdabda * SEU
            MinorAxisDelU_vs_U = lamdabda * SEdelU

            # delta squared ui vs delta ui
            MajorAxisDel2U_vs_DelU = lamdabda * SEdelU
            MinorAxisDel2U_vs_DelU = lamdabda * SEdel2U

            # delta squared ui vs ui
            # this one has to be calculated differently than the other two, solve
            # system of linear equ
            a = [
                [np.cos(theta_rad) ** 2, np.sin(theta_rad) ** 2],
                [np.sin(theta_rad) ** 2, np.cos(theta_rad) ** 2],
            ]
            b = [(SEU * lamdabda) ** 2], [(SEdel2U * lamdabda) ** 2]
            sol = np.linalg.solve(a, b)
            MajorAxisDel2U_vs_U = ((sol[0]) ** 0.5)[0]
            MinorAxisDel2U_vs_U = ((sol[1]) ** 0.5)[0]

            # Each ellipse has a different center
            center1 = (stats.median(U), 0)
            center2 = (0, 0)
            center3 = (stats.median(U), 0)

            U = np.array(U)

            # Check if each point is in the ellipses
            contains1 = self.in_Ellipse(
                U, delU, MajorAxisDelU_vs_U, MinorAxisDelU_vs_U, center1
            )
            contains2 = self.in_Ellipse(
                del2U, delU, MajorAxisDel2U_vs_DelU, MinorAxisDel2U_vs_DelU, center2
            )
            contains3 = self.in_Ellipse(
                U, del2U, MajorAxisDel2U_vs_U, MinorAxisDel2U_vs_U, center3, theta_rad
            )

            # this is an array of indexes (count) that contain a spike, if the point is outside any of the ellipses then it is marked as a spike
            replaceIndexes = np.where((contains1 == False) | (contains2 == False) | (contains3 == False))[0]

            # Remove safeIndexes from the replaceIndexes array
            if len(safeIndexes) != 0:
                for safeIndex in safeIndexes:
                    index_to_delete = np.where(replaceIndexes == safeIndex)[0]
                    if len(index_to_delete) > 0:
                        replaceIndexes = np.delete(replaceIndexes, index_to_delete)

            # keep track of all the points that were replaced
            overallReplacementIndexes = np.append(overallReplacementIndexes, replaceIndexes)
            # remove duplicates from the overallReplacementIndexes array
            overallReplacementIndexes = np.unique(overallReplacementIndexes)

            if showPhaseSpace:
                self.graphPhaseSpace(
                    U,
                    delU,
                    del2U,
                    center1,
                    center2,
                    center3,
                    MajorAxisDelU_vs_U,
                    MinorAxisDelU_vs_U,
                    MajorAxisDel2U_vs_DelU,
                    MinorAxisDel2U_vs_DelU,
                    MajorAxisDel2U_vs_U,
                    MinorAxisDel2U_vs_U,
                    theta_rad,
                    replaceIndexes, 
                    numIterations,
                    replacementMethod, 
                    number_of_replaced_points=len(overallReplacementIndexes)

                )

            # replace the points with desired replacement method. Default is Cublic Spline Interpolation
            U = self.replace(U, replaceIndexes, method=replacementMethod)

            # if the number of points to be replaced is constant, then break the loop and return the despiked data
            if len(replaceIndexes) == 0:
                break
            numIterations += 1
        
        return U, len(overallReplacementIndexes)

    def Phase_Space_Thresholding(
        self,
        replacementMethod="CubicSpline",
        scaleEstimator="SD",
        safeIndexes=[],
        maxNumIterations=20,
        derivativeMethod="CD_O2",
        showPhaseSpace=False,
    ):
        """This method applies the Phase Space Thresholding to each component of the velocity individually.

        Arguments: These will all be passed to the Phase_Space_Thresholding_SingleComponent method

        - replacementMethod (str): The method used to replace the spikes. Default is 'CubicSpline'
        - scaleEstimator (str): The method used to estimate the scale. Default is 'SD'. Other options are 'MAD' for median absolute deviation
        - safeIndexes (list): The indexes of the points that should not be replaced. Default is []
        - maxNumIterations (int): The maximum number of iterations. Default is 20
        - derivativeMethod (str): The method used to calculate the derivative. Default is 'CD_O2'. Other options are 'CD_O1' for Central Difference Order 1 and 'FBD' for Forward and Backward Difference
        - showPhaseSpace (bool): Flag indicating whether to show the phase space plot. Default is False
        """

        for i, vel_comp in enumerate(self.Vel):
   
            self.Vel[i], self.numReplaced[i] = self.Phase_Space_Thresholding_SingleComponent(
                vel_comp,
                replacementMethod=replacementMethod,
                scaleEstimator=scaleEstimator,
                safeIndexes=safeIndexes,
                maxNumIterations=maxNumIterations,
                derivativeMethod=derivativeMethod,
                showPhaseSpace=showPhaseSpace,
            )

    def mPST(self, threshold=1.8, constant2=1.483, replacementmethod="LastValid", maxNumIterations=20, derivativeMethod="CD_O2", showPhaseSpace=False):
        """Threshold (called C1 in the Parsheh) 'is an arbitrary parameter whose value
        has been selected because it allows for optimal results by
        avoiding elimination by the PST technique of a large
        fraction of the points in the middle part of the PDF, as shown
        in Fig. 4. It should be noted that the optimal value of this
        parameter varies based on the type of velocity PDF' (Quoted from Parsheh et al, 2010)

        'Wahl (2003) suggests that C2*theta, where C2= 1.483, makes the scale estimator
        based on median absolute deviation, analogous to the standard deviation.
        However, our study shows that the optimal value can vary in the range 1.25<C2<1.45
        based on the standard deviation of the PDF. The results presented in the next
        section correspond to a value of C2= 1.35.' (Quoted from Parsheh et al, 2010)
        [1] M. Parsheh, F. Sotiropoulos, and F. Porté-Agel, “Estimation of Power Spectra of Acoustic-Doppler Velocimetry Data Contaminated with Intermittent Spikes,” J. Hydraul. Eng., vol. 136, no. 6, pp. 368–378, Jun. 2010, doi: 10.1061/(ASCE)HY.1943-7900.0000202.

        Arguments
        ----------
        threshold : float, optional
            The threshold value. Default is 1.8
        constant2 : float, optional
            The constant used to calculate the scale estimator. Default is 1.483
        replacementmethod : str, optional
            The method used to replace the spikes. Default is 'LastValid'
        maxNumIterations : int, optional
            The maximum number of iterations. Default is 20
        derivativeMethod : str, optional
            The method used to calculate the derivative. Default is 'CD_O2' 
        showPhaseSpace : bool, optional
            Flag indicating whether to show the phase space plot. Default is False

        """

        # replace the data with the last valid data point
        for i, vel_comp in enumerate(
            self.Vel
        ):  # despike each component of the velocity individually
            Umean = stats.mean(vel_comp)  # calculates the mean of the data
            Ufluc = vel_comp - Umean  # calculates the fluctuations of the data
            Umad = robust.mad(vel_comp)  # median absolute deviation (MAD)

            # These are the indices that will be marked as unchangeable
            safeIndexArray = (
                np.where(Ufluc <= threshold * Umad)
                and np.where(Ufluc >= -1 * threshold * Umad)[0]
            )

            # Remove 'obvious' "spikes characterized by a large magnitude of u compared to the rest of the data set before using
            # the PST ellipsoid to identify the rest of the spikes." (Parsheh et al, 2010)
            Uemax = (
                constant2 * Umad * math.sqrt(2 * (np.log(len(vel_comp))))
            )  # expected maximum value of u
            removeIndexes = (np.where(abs(Ufluc) > Uemax))[
                0
            ]  # these indexes will be removed before PST

            # replace excluded points with the last valid data point
            self.Vel[i] = self.replace(
                vel_comp, removeIndexes, method=replacementmethod
            )

            # finally apply the PST method for one single component using the safe indexes
            self.Vel[i], self.numReplaced[i] = self.Phase_Space_Thresholding_SingleComponent(
                self.Vel[i],
                replacementMethod=replacementmethod,
                scaleEstimator="MAD",
                safeIndexes=safeIndexArray,
                maxNumIterations=maxNumIterations,
                derivativeMethod=derivativeMethod,
                showPhaseSpace=showPhaseSpace,
            )

        # recalculates the mean, rms, tke, and reynolds stresses with the new despiked data
        self.update()

    """Despiking Method Helpers"""

    def in_Ellipse(self, x, y, majorAxis, minorAxis, center=[0, 0], angle=0):
        """This depspiking helper method checks if the point (x,y) is inside the ellipse defined by the major and minor axis,
        the center and the angle of rotation. It returns an array of booleans where True indicates that the point
        is inside the ellipse and False indicates that the point is outside the ellipse.

        Arguements:
        x (float): The x coordinate of the point
        y (float): The y coordinate of the point
        majorAxis (float): The major axis of the ellipse
        minorAxis (float): The minor axis of the ellipse
        center (list, optional): The center of the ellipse. Default is [0, 0]
        angle (float, optional): The angle of rotation of the ellipse. Default is 0

        Returns:
        contains_array (array): An array of booleans where True indicates that the point is inside the ellipse and False indicates that the point is outside the ellipse
        """

        # Shift the point to the ellipse's local coordinate system
        term1 = (((x - center[0])*np.cos(angle) + (y - center[1])*np.sin(angle))**2)/ majorAxis**2
        term2 = (((x - center[0])*np.sin(angle) - (y - center[1])*np.cos(angle))**2)/ minorAxis**2

        return term1 + term2 <= 1

    def getDel2U(self, U, method="CD_O2"):
        """This helper method calculates the second derivative of the velocity data using the central difference method or the forward and backward difference method.
        It returns the second derivative of the velocity data.

        Arguements:
        U (list): The velocity data
        method (str, optional): The method used to calculate the second derivative. Default is "CD_O2" (Central Difference Order 2). Other options are "CD_O1" (Central Difference Order 1) and "FBD" (Forward and Backward Difference)

        Returns:
        del2U (list): The second derivative of the velocity data
        """

        # Create Zero Vector for Delta U (delU)
        del2U = np.zeros(len(U))

        # central difference (CD_O1) Method
        if method == "CD_O1":
            del2U = self.getDelU(self.getDelU(U))

        # Forward and Backward Difference Methods (FBD)
        if method == "FBD":
            del2U = self.getDelU(self.getDelU(U, method="FBD"), method="FBD")

        # Central Difference (CD_O2) Method
        if method == "CD_O2":
            del2U = np.gradient(np.gradient(U))

        return del2U

    def getDelU(self, U, method="CD_O2"):
        """This helper method calculates the first derivative of the velocity data using the central difference method, the forward and backward difference methods.
        "CD_O2" is recommended because it is the fastest method.

        Arguements:
        U (list): The velocity data
        method (str, optional): The method used to calculate the first derivative. Default is "CD_O2" (Central Difference Order 2). Other options are "CD_O1" (Central Difference Order 1) and "FBD" (Forward and Backward Difference)

        Returns:
        delU (list): The first derivative of the velocity data
        """

        # Create Zero Vector for Delta U (delU)
        delU = np.zeros_like(U)

        # Central Difference (CD_O1) method
        if method == "CD_O1":
            # Calculate Delta U by taking the average of the point in front and behind the point of interest
            for i, Vel in enumerate(U):
                if i == len(U) - 1:
                    delU[i] = stats.median(delU)
                    break
                if i > 0:
                    delU[i] = round((U[i + 1] - U[i - 1]) / 2, 5)

            # Set first value to the median
            delU[0] = stats.median(delU)

        # Forward and Backward Difference Methods (FBD)
        if method == "FBD":
            """This method was proposed by Islam and Zhu to solve the problem of overidentification of spikes
            "both the forward and backward difference methods, choosing that one that has a smaller
            absolute value. It was observed that this definition removes fewer
            outliers compared to the central difference method." (Islam and Zhu, 2013)
            """
            for i, Vel in enumerate(U):
                if i == len(U) - 1:
                    delU[i] = 0
                    break
                if i > 0:
                    db = U[i] - U[i - 1]
                    df = U[i + 1] - U[i]
                    if abs(db) > abs(df):
                        delU[i] = df
                    else:
                        delU[i] = db
            # Set first value to 0
            delU[0] = 0

        if method == "CD_O2":
            delU = np.gradient(U)

        return delU

   
        """This helper method calculates the cutoff values for the kernel density method. It returns the lower and upper cutoff values."""

        lf = len(f)
        dk = (np.concatenate(([0], np.diff(f)[0])) * 256) / dp
        i1 = None
        i2 = None

        for i in range(Ip - 1, 1, -1):
            if (f[i] / f[Ip] <= c1) and (abs(dk[i]) <= c2):
                i1 = i
                break

        for i in range(Ip + 1, lf + 1):
            if (f[i] / f[Ip] <= c1) and (abs(dk[i]) <= c2):
                i2 = i
                break

        ul = uf[i1]
        uu = uf[i2]

        return ul, uu

    """Replacement Methods"""
    def replace(self, data, replacementIndexes, method="CublicSpline", neighborhood_size = 13):
        """This method replaces the points in the data that are in the replacementIndexes array with the desired replacement method.

        Arguments:
        data (list): The data to be replaced
        replacementIndexes (list): The indexes of the points to be replaced
        method (str, optional): The method used to replace the points. Default is 'CublicSpline'. Other options are 'LastValid', 'PrecedingTwoExtrapolation', 'PrecedingTwoAverage', 'OverallMean', 'LinearInterpolation'
        neighborhood_size (int, optional): The number of points to interpolate for cubic spline. Default is 13
        
        Returns:
        data (list): The data with the replaced points
        """

        # if there are no indexes to be replaced, then return the original data
        if len(replacementIndexes) == 0:
            return data

        for p in replacementIndexes:
            # p is the index of the point to be replaced

            # Cubic Spline replacement. First suggested in Nikora and Goring (2002).
            if method == "CubicSpline":
                if neighborhood_size <= p < len(data) - neighborhood_size:
                    start_index = p - neighborhood_size
                    end_index = p + neighborhood_size

                    xx = list(range(start_index, end_index + 1))
                    xx.remove(p)
                    uu = data[xx]

                    cs = CubicSpline(xx, uu)
                    data[p] = cs(p)

            # Last Valid Point replacement. This replacement method was suggested in Nikora and Goring (2002) and Parsheh (2010)
            # sample-and-hold technique Adrian and Yao 1987; Nobach et al. 1998
            if method == "LastValid":
                if p == 0:
                    data[0] = stats.median(
                        data
                    )  # if the first data point needed to be replaced, then replace it with the median
                else:
                    # then replace the others with the previous point.
                    # By starting at the beginning this ensures that the point will always be replaced by the last valid point
                    data[p] = data[p - 1]

            # Preceding Two Replacement. This method was suggested in Nikora and Goring (2002) and used in Zhong et al. (2020)
            if method == "PrecedingTwoExtrapolation":
                if p == 0:
                    data[0] = stats.median(
                        data
                    )  # if the first data point needed to be replaced, then replace it with the median
                if p == 1:
                    data[1] = stats.median(
                        data
                    )  # if the second data point needed to be replaced, then replace it with the median
                else:
                    # then replace the others with the previous point.
                    # By starting at the beginning this ensures that the point will always be replaced by the last valid point
                    data[p] = 2 * data[p - 1] - data[p - 2]

            # Replace with the Average of the two proceeding.
            if method == "PrecedingTwoAverage":
                if p == 0:
                    data[0] = stats.median(
                        data
                    )  # if the first data point needed to be replaced, then replace it with the median
                if p == 1:
                    data[1] = stats.median(
                        data
                    )  # if the second data point needed to be replaced, then replace it with the median
                else:
                    data[p] = 0.5 * (data[p - 2] + data[p - 1])

            # Overall Mean replacement. #This replacement method was suggested in Nikora and Goring (2002)
            if method == "OverallMean":
                data[p] = stats.mean(data)

            # Linear Interpolation replacement. #This replacement method was suggested in Birjandi and Bibeau (2011)
            if method == "LinearInterpolation":
                if p > 0 and p < len(data) - 1:
                    data[p] = 0.5 * (data[p - 1] + data[p + 1])

        return data

    """Getter Method"""
    def getParameter(self, parameterName):
        """This method returns the value of the parameter specified by the parameterName.

        Arguments:
        parameterName: The name of the parameter to be returned.
            Options are 'umean', 'vmean', 'wmean', 'tke', 'urms', 'vrms', 'wrms', 'turb_intensity', 'reynolds_stressesuv', 'reynolds_stressesuw', 'reynolds_stressesvw'
        """
        parameterMap = {
            "umean": self.meanVel[0],
            "vmean": self.meanVel[1],
            "wmean": self.meanVel[2],
            "tke": self.tke,
            "urms": self.rms[0],
            "vrms": self.rms[1],
            "wrms": self.rms[2],
            "turb_intensity": self.turb_intense,
            "reynolds_stressesuv": self.reynolds_stresses[0],
            "reynolds_stressesuw": self.reynolds_stresses[1],
            "reynolds_stressesvw": self.reynolds_stresses[2],
            "uauto_corr": self.auto_corr[0],
            "vauto_corr": self.auto_corr[1],
            "wauto_corr": self.auto_corr[2],
            "u_int_time_scale": self.int_time_scale[0],
            "v_int_time_scale": self.int_time_scale[1],
            "w_int_time_scale": self.int_time_scale[2],
        }
        return parameterMap.get(parameterName)

    """Update Method"""
    def update(self):
        """This method updates the mean velocity, rms velocity, turbulent kinetic energy, turbulence intensity, and Reynolds stresses of the velocity data."""
        self.calc_mean()
        self.calc_rms()
        self.calc_turb_intensity()
        self.calc_reynolds_stresses()
        self.calc_tke()
        self.calc_auto_corr()
        self.calc_integral_time_scale()

    """Turbulence Calculation Methods"""
    def calc_auto_corr(self): 
        '''This function calculates the autocorrelation of the u, v, w1, and w2 velocities.
        The output is an array of the autocorrelation of the u, v, and w1 and w2 velocities.'''
        self.auto_corr = []
        for i in range(len(self.Vel)):
            vel_diff = self.Vel[i] - self.meanVel[i]
            R = np.correlate(vel_diff, vel_diff, mode='full')[len(vel_diff)-1:] / (len(vel_diff)-1) / np.mean(self.rms[i])**2
            self.auto_corr.append(R)
        
    def calc_integral_time_scale(self):
        '''This function calculates the integral time scale of the u, v, w1, and w2 velocities. 
        The output is an array of the integral time scale [s] of the u, v, and w1 and w2 velocities.'''
        self.int_time_scale =[]
        
        for i in range(len(self.Vel)):
            try:
                zero_crossing = np.where(self.auto_corr[i] < 0)[0][0] - 1 # find were the autocorrelation crosses zero
                ITS = np.trapz(self.auto_corr[i][:zero_crossing], np.arange(0, zero_crossing,1)/self.freq)
                self.int_time_scale.append(ITS)
            except:
                print(f'No zero crossing found for {self.point_name} {self.Vel[i]}')
                self.int_time_scale.append(np.nan)
    
    def power_spectral_density(vel, fr_i, N):  # TODO impliment this method
        #     #vel: velocity fluctuations (array) (u', v', w1', w2')
        #     #N: number of segments to average spectra
        #     #Output frecuencies and power spectral density (array) (Su', Sv', Sw1', Sw2')
        #     vel = np.array(vel); fr_i = np.array(fr_i)
        #     if vel.ndim == 1:
        #         S = np.zeros([vel.shape[0]//N//2+1, 2])
        #         S[:,0], S[:,1] = welch(vel, fs=fr_i, nperseg=N, scaling='density')
        #     else:
        #         S = np.zeros([vel.shape[0]//N//2+1, vel.shape[1]*2])
        #         for i in range(0,vel.shape[1]*2,2):
        #             S[:,i], S[:,i+1] = welch(vel[:,i//2], fs=fr_i, nperseg=vel.shape[0]/N, scaling='density')
        return None  # S

    def calc_tke(self):
        """This method calculates the turbulent kinetic energy (TKE) of the velocity data.
        The TKE is calculated as the sum of the means of the squares of the velocity fluctuations."""
        u_prime_squared = []
        for velcomp in self.Vel[0]:
            u_prime_squared.append((velcomp - self.meanVel[0]) ** 2)
        v_prime_squared = []
        for velcomp in self.Vel[1]:
            v_prime_squared.append((velcomp - self.meanVel[1]) ** 2)
        
        # Following results of A Standard Criterion for Measuring Turbulence Quantities Using the Four-Receiver Acoustic 
        # Doppler Velocimetry by Park and Hwang (2021), both w1 and w2 are used to calculate the TKE 
        w_prime_squared = []
        for velcomp1, velcomp2 in zip(self.Vel[2], self.Vel[3]):
            w_prime_squared.append((velcomp1 - self.meanVel[2]) * (velcomp2 - self.meanVel[3]))

        self.tke = 0.5 * (np.mean(u_prime_squared) + np.mean(v_prime_squared) + np.mean(w_prime_squared))

    def calc_turb_intensity(self):  # TODO verify this method
        """This method calculates the turbulence intensity of the velocity data."""
        self.turb_intense = (1 / 3 * np.sum(self.rms**2)) ** 0.5 / np.sum(
            self.meanVel**2
        ) ** 0.5

    def calc_reynolds_stresses(self):
        """This method calculates the Reynolds stresses (u'v', "u'w', v'w') of the velocity data."""
        self.reynolds_stresses = np.zeros((3, len(self.Vel[0])))  # 3xn array

        u_prime = []
        for velcomp in self.Vel[0]:
            u_prime.append((velcomp - self.meanVel[0]))
        v_prime = []
        for velcomp in self.Vel[1]:
            v_prime.append((velcomp - self.meanVel[1]))

        w1_prime = []
        for velcomp in self.Vel[2]:
            w1_prime.append((velcomp - self.meanVel[2]))

        w2_prime = []
        for velcomp in self.Vel[3]:
            w2_prime.append((velcomp - self.meanVel[3]))

        # Following results of A Standard Criterion for Measuring Turbulence Quantities Using the Four-Receiver Acoustic 
        # Doppler Velocimetry by Park and Hwang (2021), w2 was used to calculate the u'w' and w1 was used to calculate the v'w'  

        # multiply each element then take the mean of that list
        self.reynolds_stresses = [
            np.mean([a*b for a,b in zip(u_prime,v_prime)]),
            np.mean([a*b for a,b in zip(u_prime,w2_prime)]),
            np.mean([a*b for a,b in zip(v_prime,w1_prime)]),
        ]

    def calc_mean(self):
        """This method calculates the mean of each of the components of the velocity data."""
        self.meanVel = np.zeros(len(self.Vel))
        for i, comp in enumerate(self.Vel):
            self.meanVel[i] = np.mean(comp)

    def calc_rms(self):
        """This method calculates the root mean square (RMS) ("u_rms", "v_rms", "w1_rms", "w2_rms") of the velocity data."""
        self.rms = np.zeros((4, len(self.Vel[0])))  # 3xn array
        vel_fluc = np.zeros((4, len(self.Vel[0])))  # 3xn array
        for i, vel_comp in enumerate(self.Vel):
            vel_fluc[i] = vel_comp - stats.mean(vel_comp)
            self.rms[i] = np.sqrt(np.mean(vel_fluc[i] ** 2))
                    #    #Datagrame column order ['u','v','z1','z2']
            #Output mean, root mean square velocity (RMS), and turbulence intensity
            #Mean_rms ["u_mean", "v_mean", "w1_mean", "w2_mean", "u_rms", "v_rms", "w1_rms", "w2_rms", "Ti"]
            # dataframe = np.array(dataframe)
            # mean_rms = np.zeros([1, 9])
            # mean_rms[0,:4] = np.mean(dataframe[:,:4], axis=0)
            # mean_rms[0,4:8] = np.std(dataframe[:,:4], axis=0)
            # mean_rms[0,8] = (1/3*np.sum(mean_rms[0,4:7]**2))**.5/np.sum(mean_rms[0,:3]**2)**.5
            # return mean_rms
            # ##Turbulence calculations
            # def turbulence_i(dataframe, fr_i, w=1): 
            #     #Datagrame column order ['u','v','z1','z2','SNR1','SNR2','SNR3','SNR4','R1','R2','R3','R4']
            #     # w: vertical velocity for the calculations
            #     #Output mean, turbulence statistics, root mean square velocity (RMS), and turbulence intensity
            #     #Turbulence (u', v', w1', w2', k, u'v', u'w', v'w', 't')
            #     mean = np.mean(dataframe, axis=0)
            #     turb_i = np.zeros([len(dataframe), 9])
            #     turb_i[:,:4] = dataframe[:,:4] - mean [:4]
            #     turb_i[:,5] = turb_i[:,0]*turb_i[:,1]
            #     if w==1:
            #         turb_i[:,4] = 1/2*(turb_i[:,0]**2+turb_i[:,1]**2+turb_i[:,2]**2)
            #         turb_i[:,6] = turb_i[:,0]*turb_i[:,2]
            #         turb_i[:,7] = turb_i[:,1]*turb_i[:,2]
            #     else :
            #         turb_i[:,4] = 1/2*(turb_i[:,0]**2+turb_i[:,1]**2+turb_i[:,3]**2)
            #         turb_i[:,6] = turb_i[:,0]*turb_i[:,3]
            #         turb_i[:,7] = turb_i[:,1]*turb_i[:,3]    
            #     turb_i[:,8] = np.arange(1,len(dataframe)+1,1)/fr_i
            #     vel_rms_i = np.std(turb_i[:,:4], axis=0)
            #     turb_intensity_i = (1/3*np.sum(vel_rms_i[:3]**2))**.5/np.sum(mean[:3]**2)**.5
            #     return mean, turb_i, vel_rms_i, turb_intensity_i