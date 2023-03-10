from StabilityClass import *
import math


class GaussianPlume(Sigma_y, Sigma_z):
    """
    Implements the basic equation for a Gaussian plume given some constant release rate Q.
    Acts as a base class for the more complex plume models, gets the multiple inheritance part with
    the dispersion coefficients out of the way. Need to specify the parameters needed to use Turner's
    method such as angle of incidence (degrees), cloud cover (%), time of day (used to check day/night),
    ceiling_height (currently in feet but might change to meters), and the wind speed. Stack height is
     required for plume center-line. Also take x', y', and z' (plume coordinates) as parameters,
     (x', y', z') is the location we want to get concentration at in plume coordinates.
    """
    def __init__(self, incidence, cloud_cover, ceiling_height, time, wind_speed, stack_height,
                 distance_downwind, distance_crosswind, height, emission_rate):
        Sigma_y.__init__(self, incidence, cloud_cover, ceiling_height, time, wind_speed, distance_downwind)
        Sigma_z.__init__(self, incidence, cloud_cover, ceiling_height, time, wind_speed, distance_downwind)
        self._x = distance_downwind
        self._y = distance_crosswind
        self._z = height
        self._h = stack_height
        self._Q = emission_rate
        self._U = wind_speed
        self._sigmaY = Sigma_y.calculate_sigma_y(self)
        self._sigmaZ = Sigma_z.calculate_sigma_z(self)

    def concentration(self):
        return (self._Q / self._U) * pow(2 * math.pi * self._sigmaY * self._sigmaZ, -1) * \
            math.exp(-0.5 * pow(self._y / self._sigmaY, 2)) * \
            math.exp(-0.5 * pow((self._z - self._h) / self._sigmaZ, 2))


class PlumeWithReflectionAndDeposition(GaussianPlume):
    """
    Adds reflection and depostion to the basic plume model. Takes an additional parameter alpha
    that describes the fraction of pollutant that is reflected off of the ground.
    Class defines functions concentration and absorbed, which describe the concentration at an
    (x', y', z') above ground and the fraction of the plume deposited instantaneously.
    """
    def __init__(self, incidence, cloud_cover, ceiling_height, time, wind_speed, stack_height,
                 distance_downwind, distance_crosswind, height, emission_rate, alpha):
        super().__init__(incidence, cloud_cover, ceiling_height, time, wind_speed, stack_height,
                         distance_downwind, distance_crosswind, height,  emission_rate)
        self.reflected = alpha

    def concentration(self):
        return (self._Q / self._U) * pow(2 * math.pi * self._sigmaY * self._sigmaZ, -1) * \
            math.exp(-0.5 * pow(self._y / self._sigmaY, 2)) * \
            (math.exp(-0.5 * pow((self._z - self._h) / self._sigmaZ, 2)) +
             self.reflected * math.exp(-0.5 * pow((self._z + self._h) / self._sigmaZ, 2)))

    def absorbed(self):
        return (1 - self.reflected) * math.exp(-0.5 * pow((self._z + self._h) / self._sigmaZ, 2))


class PlumeWithGravitationalSettling(GaussianPlume):
    """"This class adds gravitational settling along with deposition and reflection. This amounts
    to allowing the plume centerline vary with downwind distance.
    Need to specify density of the pollutant and the fluid (air in this case), particle radius,
    and viscosity. """
    def __init__(self, incidence, cloud_cover, ceiling_height, time, wind_speed, initial_stack_height,
                 distance_downwind, distance_crosswind, height,  emission_rate, alpha,
                 particle_density, fluid_density, viscosity, particle_radius):
        super().__init__(incidence, cloud_cover, ceiling_height, time, wind_speed, initial_stack_height,
                         distance_downwind, distance_crosswind, height,
                         emission_rate)
        self.reflected = alpha
        self._rhoP = particle_density
        self._rhoF = fluid_density
        self._R = particle_radius
        self._mu = viscosity
        self._H = self.create_tilt()

    def vg(self):
        return 2 * 9.8 * (self._rhoP - self._rhoF) * pow(self._R, 2) / (9 * self._mu)

    def create_tilt(self):
        def tilted_center_line(x):
            return self._h + self.vg() * x / self._U
        return tilted_center_line

    def concentration(self):
        return (self._Q / self._U) * pow(2 * math.pi * self._sigmaY * self._sigmaZ, -1) * \
            math.exp(-0.5 * pow(self._y / self._sigmaY, 2)) * \
            (math.exp(-0.5 * pow((self._z - self._H(self._x)) / self._sigmaZ, 2)) +
             self.reflected * math.exp(-0.5 * pow((self._z + self._H(self._x)) / self._sigmaZ, 2)))

    def absorbed(self):
        return (1 - self.reflected) * math.exp(-0.5 * pow((self._z + self._H(self._x)) / self._sigmaZ, 2))


class PlumeCoordinateTransform:
    """
    Might need to edit this later on but this is the general idea of what we'll have to do with the map
    coordinates that describe representative locations for our populations of interest.
        Might be 180 instead of 90 that we subtract wind_dir from,
        have to confirm how wind direction is given in data

    """
    def __init__(self, wind_direction, plume_origin, map_origin):
        self._wind_dir = wind_direction
        self._plume_origin = plume_origin
        self._map_origin = map_origin
        self._dx = self._plume_origin[0] - self._map_origin[0]
        self._dy = self._plume_origin[1] - self._map_origin[1]
        self._theta = math.radians(90 - self._wind_dir)

    def map_to_plume(self, map_coords):
        x_map, y_map = map_coords
        x_plume = self._dx + x_map * math.cos(self._theta) - y_map * math.sin(self._theta)
        y_plume = self._dy + x_map * math.sin(self._theta) + y_map * math.cos(self._theta)
        return (x_plume, y_plume)

    def plume_to_map(self, plume_coords):
        x_plume, y_plume = plume_coords
        x_map = self._map_origin[0] + (x_plume - self._dx) * math.cos(self._theta) + \
            (y_plume - self._dy) * math.sin(self._theta)
        y_map = self._map_origin[1] + (y_plume - self._dy) * math.cos(self._theta) - \
            (x_plume - self._dx) * math.sin(self._theta)
        return (x_map, y_map)
