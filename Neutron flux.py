import numpy as np

# equilibrium activity in detector
ea = 8.81223129e+05 # s^-1
eaErr = 3.19415134e+03

radc = 2.3  # cm
radcErr = 0.3  # +-cm
detRad = 7.8/2  # cm
detRadErr = 0.5/2  # +-cm
rad = np.sqrt(radc**2 + detRad**2)
radErr = 1/2 * (np.sqrt((2 * radcErr/radc*radc**2)**2
                      + (2*detRadErr/detRad*detRad**2))
              / (radc**2+detRad**2))*rad

cross = 13.4*10E-24 #cm^2
# Manganese mass from:
# https://physics.nist.gov/cgi-bin/Elements/elInfo.pl?element=25
massMn = 55.938044 #u
massMnErr = 0.000003
gramToUnit = 6.0221366516752E+23 #u
massMnTot = 4.2*gramToUnit
massMnTotErr = 0.02*gramToUnit
mntot = massMnTot/massMn
flux = ea * 4 * rad**2/(detRad**2*mntot*cross) #s^-1 cm^-2
print(flux)
