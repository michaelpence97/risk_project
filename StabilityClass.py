class InsolationIndex:
    """Takes angle of solar incidence relative to the horizontal (ground) in degrees
     and outputs Insolation Index. First requirement for Turner's method."""
    def __init__(self, incidence):
        self.beta = incidence

    def calculate_insolation_index(self):
        if self.beta < 15:
            return 1
        elif 15 <= self.beta < 35:
            return 2
        elif 35 <= self.beta < 60:
            return 3
        else:
            return 4


class NRI(InsolationIndex):
    """
    Class to retrieve the Normalized Response Index
    Uses angle of solar incidence to calculate Insolation Index using parent class.
    Adjusts Insolation Index according to cloud cover expressed as a percent, the ceiling height
    in feet.
    The time of day is compared to the time at which the sun rises and sets to determine whether
    it is nighttime. Sunrise and sunset are currently just placeholders, and the specific times will need
    to be input later when I actually look them up.
    """
    def __init__(self, incidence, cloud_cover, ceiling_height, time):
        super().__init__(incidence)
        self.CC = cloud_cover
        self.time = time
        self.H = ceiling_height

    def nighttime(self):
        if sunrise + 1 < self.time < sunset - 1:
            return False
        else:
            return True

    def calculate_nri(self):
        insolation_index = super().calculate_insolation_index()
        if self.CC == 100 and self.H < 7000:
            return 0
        else:
            if self.nighttime():
                if self.CC <= 40:
                    return -2
                else:
                    return -1
            else:
                if self.CC <= 50:
                    return insolation_index
                else:
                    if self.H < 7000:
                        insolation_index += (-2)
                    else:
                        if self.H < 16000:
                            insolation_index += (-1)
                            if self.CC == 100:
                                insolation_index += (-1)
                    if insolation_index < 1:
                        return 1
                    else:
                        return insolation_index


class Stability(NRI):
    def __init__(self, incidence, cloud_cover, ceiling_height, time, wind_speed):
        super().__init__(incidence, cloud_cover, ceiling_height, time)
        self._U = wind_speed

    def calculate_stability(self):
        nri = super().calculate_nri()
        speed = self._U
        if 0 <= speed < 0.8:
            if nri == -2:
                return "G"
            elif nri == -1:
                return "F"
            elif nri == 0:
                return "D"
            elif nri == 1:
                return "C"
            elif nri == 2:
                return "B"
            elif nri >= 3:
                return "A"
        elif 0.8 <= speed < 1.9:
            if nri == -2:
                return "G"
            elif nri == -1:
                return "F"
            elif nri == 0:
                return "D"
            elif nri == 1:
                return "C"
            elif nri == 2:
                return "B"
            elif nri == 3:
                return "B"
            elif nri == 4:
                return "A"
        elif 1.9 <= speed < 2.9:
            if nri == -2:
                return "F"
            elif nri == -1:
                return "E"
            elif nri == 0:
                return "D"
            elif nri == 1:
                return "D"
            elif nri == 2:
                return "C"
            elif nri == 3:
                return "B"
            elif nri == 4:
                return "A"
        elif 2.9 <= speed < 3.4:
            if nri == -2:
                return "F"
            elif nri == -1:
                return "E"
            elif nri == 0:
                return "D"
            elif nri == 1:
                return "D"
            elif nri == 2:
                return "C"
            elif nri == 3:
                return "B"
            elif nri == 4:
                return "B"
        elif 3.4 <= speed < 3.9:
            if nri == -2:
                return "E"
            elif nri == -1:
                return "D"
            elif nri == 0:
                return "D"
            elif nri == 1:
                return "D"
            elif nri == 2:
                return "C"
            elif nri == 3:
                return "B"
            elif nri == 4:
                return "B"
        elif 3.9 <= speed < 4.9:
            if nri == -2:
                return "E"
            elif nri == -1:
                return "D"
            elif nri == 0:
                return "D"
            elif nri == 1:
                return "D"
            elif nri == 2:
                return "C"
            elif nri == 3:
                return "C"
            elif nri == 4:
                return "B"
        elif 4.9 <= speed < 5.5:
            if nri == -2:
                return "E"
            elif nri == -1:
                return "D"
            elif nri == 0:
                return "D"
            elif nri == 1:
                return "D"
            elif nri == 2:
                return "D"
            elif nri == 3:
                return "C"
            elif nri == 4:
                return "C"
        elif 5.5 <= speed < 6:
            if nri == -2:
                return "D"
            elif nri == -1:
                return "D"
            elif nri == 0:
                return "D"
            elif nri == 1:
                return "D"
            elif nri == 2:
                return "C"
            elif nri == 3:
                return "C"
            elif nri == 4:
                return "C"
        else:
            if nri == -2:
                return "D"
            elif nri == -1:
                return "D"
            elif nri == 0:
                return "D"
            elif nri == 1:
                return "D"
            elif nri == 2:
                return "D"
            elif nri == 3:
                return "D"
            elif nri == 4:
                return "C"


class Sigma_y(Stability):
    def __init__(self, incidence, cloud_cover, ceiling_height, time, wind_speed, distance):
        super().__init__(incidence, cloud_cover, ceiling_height, time, wind_speed)
        self.sigmay = self.create_function()
        self.x = distance

    def create_function(self):
        sc = super().calculate_stability()
        if sc == 'A':
            def sigmay(x):
                return .22 * x * pow((1 + 0.0001 * x), -0.5)
        elif sc == 'B':
            def sigmay(x):
                return .16 * x * pow((1 + 0.0001 * x), -0.5)
        elif sc == 'C':
            def sigmay(x):
                return .11 * x * pow((1 + 0.0001 * x), -0.5)
        elif sc == 'D':
            def sigmay(x):
                return .08 * x * pow((1 + 0.0001 * x), -0.5)
        elif sc == 'E':
            def sigmay(x):
                return .06 * x * pow((1 + 0.0001 * x), -0.5)
        else:
            def sigmay(x):
                return .04 * x * pow((1 + 0.0001 * x), -0.5)
        return sigmay

    def calculate_sigma_y(self):
        function = self.sigmay
        return function(self.x)


class Sigma_z(Stability):
    def __init__(self, incidence, cloud_cover, ceiling_height, time, wind_speed, distance):
        super().__init__(incidence, cloud_cover, ceiling_height, time, wind_speed)
        self.sigmaz = self.create_function()
        self.x = distance

    def create_function(self):
        sc = super().calculate_stability()
        if sc == 'A':
            def sigmaz(x):
                return 0.2 * x
        elif sc == 'B':
            def sigmaz(x):
                return 0.12 * x
        elif sc == 'C':
            def sigmaz(x):
                return .08 * x * pow((1 + 0.0002 * x), -0.5)
        elif sc == 'D':
            def sigmaz(x):
                return .06 * x * pow((1 + 0.0015 * x), -0.5)
        elif sc == 'E':
            def sigmaz(x):
                return .03 * x * pow((1 + 0.0003 * x), -1)
        else:
            def sigmaz(x):
                return .016 * x * pow((1 + 0.0003 * x), -1)
        return sigmaz

    def calculate_sigma_z(self):
        function = self.sigmaz
        return function(self.x)