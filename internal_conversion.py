#=====================================================================
#
#                    VIBRATIONAL COUPLING SCRIPT
#
#=====================================================================

"""
please see manual for how to run the script

execute script by:
$ python3 ./internal_conversion.py ./frequency_output
"""

#=====================================================================
#
#         PACKAGES, PHYSICAL CONSTANTS AND CONVERSION FACTORS
#
#=====================================================================

import numpy as np
import re
import os
import sys

# physical constants
hbar = 1.054571628 * 10 ** (-34) # [J*s]
c = 299792458 * 10 ** 2 # [cm/s]

# conversion factors
hartree_to_joule = 4.359744650 * 10 ** (-18) # [J/hartree]
eV_to_J = 1.60217662 * 10 ** (-19) # [J/eV]
bohr_to_m = 5.2918 * 10 ** (-11) # [bohr/m]
A_to_m = 10 ** (-10) # [A/m]
amu_to_kg = 1.66053904*10 ** (-27) # [amu/kg]

# =====================================================================
#
#                              FUNCTIONS
#
# =====================================================================

def read_energies(filename):

    """
    this function extracts the energies from the TD-DFT output files
    """

    # open file and read content
    with open(filename,"r") as file:
        content = file.read()

    # split content by space between characters
    split_content = content.split('\n')

    # create search patterns
    excited = re.compile(" Excited State", re.MULTILINE)
    SCF_done = re.compile(" SCF Done:", re.MULTILINE)

    # create empty list to collect the energies
    energies = []

    for i, line in enumerate(split_content):

        # search for SCF energy
        if re.match(SCF_done, line):

            # convert from atomic units to [eV]
            scf_energy = float(line.split()[4]) * 27.212
            energies.append(scf_energy)

        # search for excitation energy
        if re.match(excited, line):
            ex_energy = float(line.split()[4])
            energies.append(ex_energy)

    # the zero point is chosen to be the ground state
    zeropoint = energies[0]

    # calculate the excited state energies
    energies[1:] = [zeropoint + ex_energy for ex_energy in energies[1:]]

    return np.array(energies)    

def read_normal_modes(filename):

    """
    this function extracts the normal modes from the frequency output file
    """

    with open(filename,"r") as f:
        content = f.read()

    split_content = content.split('\n')

    # create search patterns
    atom_num = re.compile(" NAtoms= ", re.MULTILINE)
    patt = re.compile(" Frequencies --", re.MULTILINE)

    # search for number of atoms and stop when found
    for i, line in enumerate(split_content):
        if re.match(atom_num, line):
            numatoms = line.split()[1]
            break

    # define empty list to collect the line numbers where the frequencies are
    # listed in the frequency output file
    freq_lines = []

    for i, line in enumerate(split_content):
        if re.search(patt, line):

            # append the line number of the output file
            # that contains the frequencies
            freq_lines.append(i)

    # define empty list to collect all the normal modes
    normal_modes = []

    # go through the lines where the frequencies are listed
    for i in freq_lines:

        # three normal modes are listed on each line, and each
        # column is appended to a separate list. mode1 contain those
        # to the left, mode2 contain those in the middle and mode3
        # contain those to the right
        mode1 = []
        mode2 = []
        mode3 = []

        # read the frequencies (three frequencies are extracted)
        frequencies = split_content[i].split()[2:]

        # read the reduced masses, which are listed one line below
        # the frequencies
        reduces_massees = split_content[i + 1].split()[3:]

        # append the frequency and reduced mass in the first column
        # to mode1, second column to mode2 and last column to mode3
        mode1.append(float(frequencies[0]))
        mode1.append(float(reduces_massees[0]))
        mode2.append(float(frequencies[1]))
        mode2.append(float(reduces_massees[1]))
        mode3.append(float(frequencies[2]))
        mode3.append(float(reduces_massees[2]))

        # the displacement matrix coordinates are listed five lines
        # further below, and extends [(number of atoms) + line number]
        for j in np.arange(int(i) + 5, int(i) + int(numatoms) + 5, 1):

            # the content on each line is splitted
            line = split_content[j].split()

            # the x, y and z coordinate for the mode in the first column
            # are character 2, 3 and 4, respectively. for the second column
            # it is 5, 6 and 7. for the third column it is 8, 9 and 10. they
            # are appended to the lists defined for each column
            mode1.append([float(line[2]), float(line[3]), float(line[4])])
            mode2.append([float(line[5]), float(line[6]), float(line[7])])
            mode3.append([float(line[8]), float(line[9]), float(line[10])])

        # the displacement matrix is flatten (converted to a vector) for
        # the vector projection of the forces. at the moment, they are written
        # as [x, y, z] for each atom, and must therefore be flatten. however,
        # the two first elements are the frequency and reduced mass, so only the
        # remaining elements are flatten
        mode1[2:] = np.array(mode1[2:]).flatten()
        mode2[2:] = np.array(mode2[2:]).flatten()
        mode3[2:] = np.array(mode3[2:]).flatten()
        
        # append the columns to the normal mode list. the normal modes are listed
        # in increasing order, that is, normal mode 1, normal mode 2, etc.
        normal_modes.append(mode1)
        normal_modes.append(mode2)
        normal_modes.append(mode3)
    
    return np.array(normal_modes)

def read_forces(filename):

    """
    this function extracts the forces from the TD-DFT output files
    """

    with open(filename,"r") as f:
        content = f.read()

    split_content = content.split('\n')

    # create search patterns
    atom_num = re.compile(" NAtoms= ", re.MULTILINE)
    patt = re.compile("Forces\s\(Hartrees\/Bohr\)", re.MULTILINE)
    # here \s represents an undefined number of whitespaces

    # search for atom number and stop when found
    for i,line in enumerate(split_content):
        if re.match(atom_num, line):
            numatoms = line.split()[1]
            break
            
    # search for forces
    for i, line in enumerate(split_content):
        if re.search(patt, line):

            # define line number in the output file where the
            # forces are listed
            force_line = i

    # define null matrix for the forces of dimension
    # (number of atoms) x (3 = number of Cartesian coordinates)
    force_matrix = np.zeros((int(numatoms), 3))

    # k is the number of lines from force_line to end of the list with forces,
    # j is the actual line number in the file
    for k, j in enumerate(np.arange(int(force_line) + 3, int(force_line) + int(numatoms) + 3)):

        # first two characters of the line is center number and atom number,
        # the three remaining characters are the Cartesian components of the force
        forces = split_content[j].split()[2:]

        # for each line, set the first element in forces to the first column, the second
        # element in the second column and the third element in third column
        force_matrix[k][0] = forces[0]
        force_matrix[k][1] = forces[1]
        force_matrix[k][2] = forces[2]

    # the force matrix is flatten, such that it can be projected onto the normal modes
    return force_matrix.flatten()

def vector_projection(vec_a, vec_b):

    """
    this function calculates the inner product of two vectors, vec_a and vec_b
    """

    return np.dot(vec_a, vec_b) / np.linalg.norm(vec_b)

# =====================================================================
#
#                             RUN PROGRAM
#
# =====================================================================

if __name__ == '__main__':

    # read the normal modes from the frequency output file
    normal_modes = read_normal_modes(sys.argv[1])

    # collect all files with forces, which should contain the word "root" in the name
    files = [filename for filename in os.listdir(".") if "root" in filename]
    files.sort()

    # the frequencies of the normal modes are listed as the first element in each row
    frequencies = normal_modes[:,0]

    # the reduced masses of the normal modes are listed as the second element in each row
    masses = normal_modes[:,1]

    # define empty list for coupling constants (\gamma), Huang-Rhys factors and energies
    coupling_constants = []
    huang_rhys = []
    energies = []

    # for each force file
    for j in range(len(files)):

        # read force file
        fm = read_forces(files[j])

        # for each normal mode
        for i in range(len(normal_modes)):

            # calculate gradient and convert to [J/m]
            grad = vector_projection(fm, normal_modes[i][2:]) * hartree_to_joule / bohr_to_m

            # read corresponding frequency
            freq = float(frequencies[i])

            # calculate angular frequency
            omega = 2 * np.pi * freq

            # read corresponding reduced mass and convert to kg
            mu = float(masses[i]) * amu_to_kg

            # calculate coupling
            g = (1 / np.sqrt((2 * hbar * mu * (omega * c) ** 3))) * grad

            # append the coupling constant
            coupling_constants.append(("mode" + str(i + 1), "state" + str(j + 1), g))

            # append mode, state, frequency and the calculated Huang-Rhys factor
            huang_rhys.append(("mode" + str(i + 1), "state" + str(j + 1), float(freq), float(0.5 * g ** 2)))

    # calculate number of states the gradient is calculated for
    num_states = len(files)

    # extract the Huang-Rhys factors from the huang-rhys list defined above, which
    # is the third element in each row
    huang_rhys = np.array(huang_rhys)
    hr = huang_rhys[:, 3]
    hr = np.array(hr)

    # collect the mode numbers
    modes = huang_rhys[:, 0]
    modes = [int(mode.strip("mode"))-1 for mode in modes]

    # collect the frequencies
    freq = huang_rhys[:, 2]
    freq = [float(fr) for fr in freq]
    freq = np.array(freq)

    # extract the coupling elements
    coupling_constant = np.array(coupling_constants)
    g = np.array(coupling_constant[:, 2], dtype=float)

    # calculate number of normal modes in the molecule
    num_normal = len(normal_modes)

    # define normal mode counter
    k = 0

    # for each excited state
    for i in range(num_states):

        # calculate number of normal modes
        k += num_normal

        # extract frequency
        freq_list = freq[i * num_normal: k + 1]

        # extract coupling element
        g_list = g[i * num_normal: k + 1]

        # construct empty list for frequencies and coupling elements
        frequencies = []
        coupling_elements = []

        # go through all the normal modes
        for j in range(len(freq_list)):

            # append only frequencies and coupling elements for frequencies
            # higher than 200 [cm^-1]
            if freq_list[j] > 200:
                frequencies.append(float(freq_list[j]))
                coupling_elements.append(g_list[j])

        # extract the root numbers from the TD-DFT output files
        root = files[i].split("root")[-1].split(".")[0]

        # write .csv files with the frequencies in [cm^-1] and dimensionless coupling constants
        with open("coupling_constant" + str(root) + ".csv","w") as file:
            file.write("{} \n".format(",".join(map(str, frequencies))))
            file.write("{} \n".format(",".join(map(str, coupling_elements))))

#=====================================================================
#
#                           END OF SCRIPT
#
#=====================================================================
