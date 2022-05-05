#=====================================================================
#
#                      SOLVENT COUPLING SCRIPT
#
#=====================================================================

"""
please see manual for how to run the script.

execute script by:
$ python3 ./solvent_coupling.py
"""

#=====================================================================
#
#                  PACKAGES AND CONVERSION FACTORS
#
#=====================================================================

import numpy as np
import re
import os

# conversion factors
Hz_to_atomic_unit = 1.51983 * 10 ** -16
second_to_atomic_unit = 4.1341374575751 * 10 ** 16
angstrom_to_atomic_units = 1.8897259886

# =====================================================================
#
#                              FUNCTIONS
#
# =====================================================================

def dielectric_constant(freq, lifetime, epsilon_s, epsilon_infinity):

    """
    this function calculates the real part of the dielectric constant using the Debye equation
    """

    return epsilon_infinity + (epsilon_s - epsilon_infinity) / (1 + (freq * lifetime) ** 2)

def g_weight(radius_of_cav, freq, lifetime, l, epsilon_s, epsilon_infinity):

    """
    this function calculates the weight, g_lm, from the DCM
    """

    return - radius_of_cav ** (-(2 * l + 1)) * ((dielectric_constant(freq, lifetime, epsilon_s, epsilon_infinity) - 1) * (l + 1 )) / (l + dielectric_constant(freq, lifetime, epsilon_s, epsilon_infinity) * (l + 1))

def coupling_constant(radius_of_cav, freq, lifetime, l, epsilon_s, epsilon_infinity):

    """
    this function calculates the solvent coupling constant
    """

    return np.abs( np.sqrt( - (1/2) * g_weight(radius_of_cav, freq, lifetime, l, epsilon_s, epsilon_infinity) * freq))

# =====================================================================
#
#                         SOLVENT PARAMETERS
#
# =====================================================================

"""
the solvent parameters are manually added to this dictionary. the 
frequencies are listed in [Hz], and the lifetimes are listed in [s].
"""

solvents = {
    "water": {
        'frequency': [2 * (10 ** 9), 400 * (10 ** 9), 1850 * (10 ** 9)],
        'lifetime': [1.25 * (10 ** -12), 1.0 * (10 ** -12), 0.48 * (10 ** -12)],
        'static_dielectric_constant': 78.54,
        'optical_dielectric_constant': 1.7766
    },
    "acetonitrile": {
        'frequency': [50 * (10 ** 9), 440 * (10 ** 9), 2000 * (10 ** 9)],
        'lifetime': [1.0 * (10 ** -12), 0.5 * (10 ** -12), 0.35 * (10 ** -12)],
        'static_dielectric_constant': 35.96,
        'optical_dielectric_constant': 1.7985
    },
    "methanol": {
        'frequency': [3 * (10 ** 9), 20 * (10 ** 9), 200 * (10 ** 9)],
        'lifetime': [14.3 * (10 ** -12), 11.1 * (10 ** -12), 5.0 * (10 ** -12)],
        'static_dielectric_constant': 32.63,
        'optical_dielectric_constant': 1.7583
    },
    "dichloromethane": {
        'frequency': [70 * (10 ** 9), 400 * (10 ** 9), 1830 * (10 ** 9)],
        'lifetime': [5.55 * (10 ** -12), 1.1 * (10 ** -12), 0.41 * (10 ** -12)],
        'static_dielectric_constant': 8.93,
        'optical_dielectric_constant': 2.0244
    }
}

#=====================================================================
#
#                        CALCULATION SETTINGS
#
#=====================================================================

# specify radius of cavity

# average of donor, chromophore and acceptor
radius_of_cav = 5.57 * angstrom_to_atomic_units # [a.u.]

# average of donor, bridge1, chromophore, bridge2 and acceptor
#radius_of_cav = 4.33 * angstrom_to_atomic_units # [a.u.]

# radius of largest fragment (chromophore)
#radius_of_cav = 6.69 * angstrom_to_atomic_units # [a.u.]

#=====================================================================
#
#                            RUN PROGRAM
#
#=====================================================================

if __name__ == '__main__':

    # find solvent output files in the current directory
    files = [filename for filename in os.listdir(".") if "out" in filename]

    # define empty list to append relevant values
    l_values = []
    m_values = []
    spherical_multipoles = []

    # for each file, open and read the content
    for file in files:
        with open(file, "r") as f:
            content = f.read()

        # split the content
        split_content = content.splitlines()

        # create search patterns
        l_and_m = re.compile(" --- l, m :", re.MULTILINE)
        spherical_multipole = re.compile(" T\(", re.MULTILINE)

        # search through lines and append if found
        for i, line in enumerate(split_content):

            if re.match(l_and_m, line):
                l = float(line.split()[4])
                l_values.append(l)
                m = float(line.split()[5])
                m_values.append(m)

            if re.match(spherical_multipole, line):
                T = line.split()[2]
                T_value = float(T.replace('D', 'E'))
                spherical_multipoles.append(T_value)

    # calculate the number of l values for each solvent
    l_values_per_solvent = int( len(l_values) / len(files) )

    # get list of l values
    l_list = l_values[0: l_values_per_solvent]

    # create empty list for the solvent names
    solvent_names = []

    # extract the solvent names from the solvent output files
    for i in files:
        solvent = i.split("_structure")[0]
        solvent_names.append(solvent)

    # define counter to extract spherical multipole moments
    k = 0

    # extract spherical multipole moments and add them to the solvents dictionary
    for solvent in solvent_names:
        solvents['' + solvent + '']['spherical_multipole'] = spherical_multipoles[k: l_values_per_solvent + k]
        k += l_values_per_solvent

    # for each solvent
    for solvent in solvent_names:

        # define list for the solvent coupling constants
        solvent_coupling = []

        # call dictionary
        property = solvents.get(solvent)

        # extract the frequencies, lifetimes, static and optical dielectric constant
        freq_list_Hz = property.get('frequency')
        lifetime_list_s = property.get('lifetime')
        static_dielectric_constant = property.get('static_dielectric_constant')
        optical_dielectric_constant = property.get('optical_dielectric_constant')

        # create empty list for frequencies and lifetimes in atomic units
        freq_list = []
        lifetime_list = []

        # convert frequency unit from [Hz] to [a.u.]
        for freq in freq_list_Hz:
            freq *= Hz_to_atomic_unit
            freq_list.append(freq)

        # convert lifetime unit from [s] to [a.u.]
        for tau in lifetime_list_s:
            tau *= second_to_atomic_unit
            lifetime_list.append(tau)

        # for each l value
        for l in l_list:

            # for each pair of frequency and lifetime
            for freq, lifetime in zip(freq_list, lifetime_list):

                # calculate coupling constant
                alpha = coupling_constant(radius_of_cav, freq, lifetime, l, static_dielectric_constant, optical_dielectric_constant)
                solvent_coupling.append(alpha)

        # extract coupling constants for each frequency
        alpha1_list = solvent_coupling[0::3]
        alpha2_list = solvent_coupling[1::3]
        alpha3_list = solvent_coupling[2::3]

        # extract spherical multipole moments from solvents dictionary
        spherical_multipole_list = property.get('spherical_multipole')

        # write .csv files with l values, spherical multipole moments and coupling constants
        with open(str(solvent) + "_coupling.csv", "w") as file:
            file.write("{} \n".format(",".join(map(str, l_list))))
            file.write("{} \n".format(",".join(map(str, spherical_multipole_list))))
            file.write("{} \n".format(",".join(map(str, alpha1_list))))
            file.write("{} \n".format(",".join(map(str, alpha2_list))))
            file.write("{} \n".format(",".join(map(str, alpha3_list))))

#=====================================================================
#
#                           END OF SCRIPT
#
#=====================================================================