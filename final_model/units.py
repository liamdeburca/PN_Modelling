from numpy import pi

class cgs:
    def __init__(self):
        self.c = 2.99792458e10 # Speed of light

        self.h = 6.62607015e-27 # Planck's constant
        self.hbar = self.h / (2 * pi) # Reduced Planck's constant

        self.kB = 1.38064910e-16 # Boltzmann's constant
        self.stefan = pi**2 * self.kB**4 / (60 * self.hbar**3 * self.c**2) # Stefan-Boltzmann constant

        self.G = 6.6743e-8 # Gravitational constant
        
        self.q_e = 4.8032e-10 # Elementary charge
        self.m_e = 9.1094e10-28 # Electron mass
        self.m_p = 1.6726e-24 # Proton mass
        self.m_n = 1.6605e-24 # Neutron mass

        self.erg = 1
        self.joule = 1e-7

        self.meter = 100
        self.millimeter = self.meter / 1e3 
        self.micrometer = self.meter / 1e6
        self.nanometer = self.meter / 1e9
        self.kilometer = self.meter * 1e3
        self.au = 1.496e13 # Astronomical Unit
        self.pc = 3.086e18 # Parsec
        self.ly = 9.461e17 # Lighstyear

        self.solar_mass = 1.989e33
        self.solar_radius = 6.955e10
        self.solar_luminosity = 3.839e33

        self.earth_mass = 5.974e27
        self.earth_radius = 6.378e8

        self.jupiter_mass = 1.899e30
        self.jupiter_radius = 7.149e9

class si:
    def __init__(self):
        self.c = 2.99792458e8 # Speed of light

        self.h = 6.62607015e-34 # Planck's constant
        self.hbar = self.h / (2 * pi) # Reduced Planck's constant

        self.kB = 1.38064910-23 # Boltzmann's constant
        self.stefan = pi**2 * self.kB**4 / (60 * self.hbar**3 * self.c**2) # Stefan-Boltzmann constant

        self.G = 6.6743e-11 # Gravitational constant
        
        self.q_e = 1.602176634e19 # Elementary charge
        self.m_e = 9.1094e10-31 # Electron mass
        self.m_p = 1.6726e-27 # Proton mass
        self.m_n = 1.6605e-27 # Neutron mass

        self.erg = 1e7
        self.joule = 1

        self.meter = 1
        self.millimeter = self.meter / 1e3 
        self.micrometer = self.meter / 1e6
        self.nanometer = self.meter / 1e9
        self.kilometer = self.meter * 1e3
        self.au = 1.496e11 # Astronomical Unit
        self.pc = 3.086e16 # Parsec
        self.ly = 9.461e15 # Lightyear

        self.solar_mass = 1.989e30
        self.solar_radius = 6.955e8
        self.solar_luminosity = 3.839e26

        self.earth_mass = 5.974e24
        self.earth_radius = 6.378e6

        self.jupiter_mass = 1.899e27
        self.jupiter_radius = 7.149e7