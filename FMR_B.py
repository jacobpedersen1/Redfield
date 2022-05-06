#=====================================================================
#
#                     ELECTRONIC COUPLING SCRIPT
#
#=====================================================================

"""
please see manual for how to run the script

execute script by:
$ python3 ./FMR_B.py ./extended_gaussian_output
"""

#=====================================================================
#
#                              PACKAGES
#
#=====================================================================

import numpy as np
import math
from scipy import linalg
import sys

# =====================================================================
#
#                              FUNCTIONS
#
# =====================================================================

def getnumberofbasisfunctions(file):
    
    """
    this function extracts the number of basis functions from
    a given gaussian output file
    """

    # define number of basis functions variable
    N_basis_functions = 0

    # open the output file and read lines
    with open(file, 'r') as file:
        line = file.readline()

        while N_basis_functions == 0:

            # the additional requirement is needed to find the right
            # number in the output file, as "basis functions," is written
            # at four different places
            if 'basis functions,' in line and N_basis_functions == 0:

                # read the number of basis functions
                N_basis_functions = int(line.split()[0])

            line = file.readline()
            
    return N_basis_functions

def getnumberofbasisfunctionperfraction(file, N_atoms_per_fraction, N_basis_functions):

    """
    this function extracts the number of basis functions for each fraction of the molecule,
    and outputs the numbers in a list.
    """

    # define empty list to contain the number of basis functions on each fragment
    N_basis_functions_per_fraction = []

    # define number of basis functions on current fragment counter
    N_basis_fraction = 0

    with open(file, 'r') as file:
        line = file.readline()

        while N_basis_fraction == 0:

            if 'Gross orbital populations:' in line:
                line = file.readline()

                # define fragment counter; 0=D, 1=B1, 2=C, 3=B2, 4=A
                j=0

                # N_atoms_per_fraction is a list containing the number of atoms on each
                # fragment. N_atom_fraction is thus the number of atoms on the current fragment
                N_atom_fraction = N_atoms_per_fraction[j]

                # go through the lines with basis functions
                for i in range(N_basis_functions):
                    data = file.readline().split()

                    # the atoms are given an integer number in the output file, and this statement
                    # checks if the current line contains an integer or string.
                    try:
                        data[1] = int(data[1])

                    # if it is a string, pass the value and move on
                    except ValueError:
                        pass

                    # if the current atom number is not equal to the number
                    # of basis functions as specified by the fragment list,
                    # raise the number of basis functions on current fragment
                    # counter by one and start over
                    if data[1] != N_atom_fraction + 1:
                        N_basis_fraction += 1

                    # when the current atom number matches the number of atoms on
                    # the current fragment;
                    else:

                        # append the number of basis functions to the list for all
                        # fragments
                        N_basis_functions_per_fraction.append(N_basis_fraction)

                        # terminate while-loop when exiting for-loop
                        N_basis_fraction = 1

                        # raise fragment counter by one
                        j += 1

                        # define the number of atoms for the current fragment
                        N_atom_fraction += N_atoms_per_fraction[j]

                # in case the if-statement is not true, append 0
                N_basis_functions_per_fraction.append(N_basis_fraction)

            line = file.readline()
    
    return N_basis_functions_per_fraction

def getfockmatrix(file):
    
    """
    this function extracts the Fock matrix in atomic orbitals
    from a Gaussian output file. when multiple Fock matrices
    are printed, only the last one will be returned.
    to print the Fock matrix in atomic orbitals, the keyword iop(5/33=3)
    must be included in the Gaussian input file.
    """

    # conversion factor
    hartree_to_eV = 27.212 # [meV/hartree]

    # define number of basis functions
    N_basis_functions = getnumberofbasisfunctions(file)

    with open(file, 'r') as file:
        line = file.readline()

        # loop through all lines
        while line:

            # search for Fock matrix
            if 'Fock matrix (alpha)' in line:

                # create null matrix with dimension
                # (number of basis functions) x (number of basis functions)
                F_mat = np.zeros((N_basis_functions, N_basis_functions), np.float64)

                # read the lines where the Fock matrix columns are written each time
                # 'Fock matrix (alpha)' is found in the output file. the Fock matrix is
                # partitioned into rows of five columns in the output file
                for i in range(int(math.ceil(N_basis_functions / 5.0))):

                    # identify and skip the line with column number
                    line = file.readline()

                    # for each line with column numbers, read the rows with matrix
                    # elements. note that the Fock matrix is symmetric, and only the
                    # lower triangular part is printed in the output file. thus, the
                    # number of rows to be collected for each "row" reduces with 5,
                    # and the start number in the range-statement should thus be
                    # multiplied with 5
                    for j in range(i * 5, N_basis_functions):

                        # split up the date written on each line
                        data = file.readline().split()

                        # go through the six elements in each line. note that
                        # len(data) = 6, but as python counts from 0; 1 is subtracted
                        # from the argument in the range-statement below
                        for k in range(len(data) - 1):

                            # insert all but the first element from the line in the output
                            # file into the jth row and (i * 5 + k)th column.
                            F_mat[j, i * 5 + k] = np.float(data[k + 1].replace('D', 'E'))

                            # the Fock matrix is symmetric
                            F_mat[i * 5 + k, j] = F_mat[j, i * 5 + k]

            line = file.readline()

    # convert to eV
    F_mat *= hartree_to_eV        

    return F_mat

def getoverlapmatrix(file):
    
    """
    this function extracts the overlap matrix (S) in atomic orbitals
    from a gaussian output file. when multiple overlap matrices are
    printed, only the last one will be returned.
    to print the overlap matrix, the keyword: iop(3/33=4) and pop=full
    must be included in the gaussian input file.
    note that the function works exactly like the getfockmatrix()
    function.
    """

    N_basis_functions = getnumberofbasisfunctions(file)
    
    with open(file, 'r') as file:
        line = file.readline()

        while line:
            if '*** Overlap ***' in line:
                S_mat = np.zeros((N_basis_functions, N_basis_functions), np.float64)
                for i in range(int(math.ceil(N_basis_functions/5.0))):
                    line = file.readline()
                    for j in range(i * 5, N_basis_functions):
                        data = file.readline().split()
                        for k in range(len(data) - 1):
                            S_mat[j, i * 5 + k] = np.float(data[k + 1].replace('D','E'))
                            S_mat[i * 5 + k, j] = S_mat[j, i * 5 + k]
            line = file.readline()

    return S_mat

def lowdin_transformed_S(F_mat, S_mat):
    
    """
    this function performs a Löwdin orthogonalization of the Fock matrix
    """

    # define S^(-1/2)
    S_inv_sqrt = linalg.inv(linalg.sqrtm(S_mat))

    # perform Löwdin orthogonalization
    F_lowdin = np.dot(S_inv_sqrt, np.dot(F_mat, S_inv_sqrt))

    return F_lowdin

def block_diagonalization_with_bridge_new(fock_matrix, DCA_basis):

    """
    this function block diagonalizes the D-B1, B1-C-B2 and B2-A sections of the Fock
    matrix to form the diabatic states.

    DCA_basis; list of number of basis functions on each fragment
    
    output: C_full; matrix containing the coefficients needed for transforming the
    AOs to the diabatic (but non-orthornormal) basis. They constitute the rows of the matrix
    """

    # the number of basis functions on the fragments are summarized
    # recall; 0=D, 1=B1, 2=C, 3=B2, 4=A
    DB1_basis = DCA_basis[0]+DCA_basis[1]
    B1CB2_basis = DCA_basis[1]+DCA_basis[2]+DCA_basis[3]
    B2A_basis = DCA_basis[3]+DCA_basis[4]

    # define null matrices for sub-matrices
    DB1_fock = np.zeros([DB1_basis, DB1_basis])
    B1CB2_fock = np.zeros([B1CB2_basis, B1CB2_basis])
    B2A_fock = np.zeros([B2A_basis, B2A_basis])

    # define sub-Fock matrices by slicing through the bases and inserting
    # in the null matrices
    DB1_fock[:,:] = fock_matrix[:DB1_basis, :DB1_basis]
    B1CB2_fock[:,:] = fock_matrix[DCA_basis[0]:(B1CB2_basis+DCA_basis[0]), DCA_basis[0]:(B1CB2_basis+DCA_basis[0])]
    B2A_fock[:,:] = fock_matrix[(DCA_basis[0]+DCA_basis[1]+DCA_basis[2]):, (DCA_basis[0]+DCA_basis[1]+DCA_basis[2]):]

    # calculate eigenvalues and eigenvectors of the blocks
    eigenvalues_DB1, C_DB1 = np.linalg.eigh(DB1_fock)
    eigenvalues_B1CB2, C_B1CB2 = np.linalg.eigh(B1CB2_fock)
    eigenvalues_B2A, C_B2A = np.linalg.eigh(B2A_fock)

    # construct coefficient matrix
    C_full = np.zeros([sum(DCA_basis), DB1_basis+B1CB2_basis+B2A_basis])
    C_full[:DB1_basis, :DB1_basis] = C_DB1
    C_full[DCA_basis[0]:(DCA_basis[0]+B1CB2_basis), DB1_basis:(DB1_basis+B1CB2_basis)] = C_B1CB2
    C_full[(DCA_basis[0]+DCA_basis[1]+DCA_basis[2]):, (DB1_basis+B1CB2_basis):] = C_B2A

    # calculate diabatic Fock matrix
    f_diabatic = np.dot(C_full.T, np.dot(fock_matrix, C_full))

    return C_full, f_diabatic

def getnumberofoccmos(file):

    """
    this function extract the number of electrons in the molecule, to get
    the number of occupied MOs assuming a ground state configuration
    """

    # define number of electrons variable
    N_electrons = 0

    with open(file, 'r') as file:
        line = file.readline()

        # loop through lines until found
        while N_electrons == 0:
            if 'alpha electrons' in line and N_electrons == 0:

                # find alpha and beta electrons
                N_alpha_electrons = int(line.split()[0])
                N_beta_electrons = int(line.split()[3])

                # calculate total number of electrons
                N_electrons = N_alpha_electrons + N_beta_electrons

            line = file.readline()

    # calculate number of occupied MOs with two electrons per occupied orbital
    N_occ_mos = math.ceil(N_electrons/2)

    return N_occ_mos

def donor_acceptor_orthorgonalize(C_full, f_diabatic, orbital_indexes):

    """
    this function extracts the orbitals indicated in "orbital_indexes" from the
    diabatic matrix, that might not be orthorgonal, and forms the 2 by 2 Fock matrix
    and overlap matrix needed to do the final Löwdin orthogonalization.
    """

    # specify non-orthogonalized fragment orbitals
    initial_orbital = orbital_indexes[0]
    final_orbital = orbital_indexes[1]

    # create 2 x 2 identity overlap matrix
    S_system = np.identity(2)

    # calculate off-diagonal elements of overlap matrix
    S_system[0,1] = np.dot((C_full[:,initial_orbital]).T, C_full[:,final_orbital])
    S_system[1,0] = np.dot((C_full[:,final_orbital]).T, C_full[:,initial_orbital])
    
    # calculate inverse squared overlap matrix
    S_system_inv_sqrt = linalg.inv(linalg.sqrtm(S_system))
    
    # create 2 x 2 null Fock matrix
    f_system = np.zeros([2,2])

    # insert diabatic energies
    f_system[0,0] = f_diabatic[initial_orbital, initial_orbital]
    f_system[1,1] = f_diabatic[final_orbital, final_orbital]

    # insert coupling elements
    f_system[0,1] = f_diabatic[initial_orbital, final_orbital]
    f_system[1,0] = f_diabatic[final_orbital, initial_orbital]
    
    # perform final Löwdin orthogonalization
    f_system_orthorgonal = np.dot(S_system_inv_sqrt, np.dot(f_system, S_system_inv_sqrt))
    
    return f_system_orthorgonal

def getelectronsdonoracceptor(file, N_atoms_per_fragment):

    """
    this function calculates the number of electrons in each fragment based on
    the number of atoms per fragment.
    """

    # define empty list for number of electrons per fragment
    electrons_per_fragment = []

    with open(file, 'r') as file:
        line = file.readline()

        while len(electrons_per_fragment) == 0:
            if 'Standard orientation:' in line:

                # define fragment counter
                # recall; 0=D, 1=B1, 2=C, 3=B2, 4=A
                k=0

                # go through the next five lines
                for i in range(4):
                    line = file.readline()

                # for each fragment
                for j in range(len(N_atoms_per_fragment)):

                    # define number of electrons per fragment variable
                    N_electrons_fragment = 0

                    # go through the lines for the current fragment
                    for l in range(N_atoms_per_fragment[k]):
                        line = file.readline()

                        # add electrons for the current fragment
                        N_electrons_fragment += int(line.split()[1])

                    # append to the list containing the number of electrons
                    electrons_per_fragment.append(N_electrons_fragment)

                    # raise fragment counter with one and repeat for each fragment
                    k += 1

            line = file.readline()
            
    return electrons_per_fragment

# =====================================================================
#
#                             RUN PROGRAM
#
# =====================================================================

if __name__ == '__main__':
      
    # specify the number of atoms per fragment.
    # it must be specified in the order; D, B1, C, B2, A
    N_atoms_per_fragment = [22, 10, 38, 10, 30]

    # read the output file
    file = sys.argv[1]

    # calculate the total number of basis functions
    N_basis_functions = getnumberofbasisfunctions(file)

    # calculate the number of basis functions on each fragment
    N_basis_per_fragment = getnumberofbasisfunctionperfraction(file, N_atoms_per_fragment, N_basis_functions)

    # calculate the number of occupied molecular orbitals
    N_occ_mos = getnumberofoccmos(file)

    # extract the adiabatic Fock matrix from the output file
    fock_mat = getfockmatrix(file)

    # extract the adiabatic overlap matrix from the output file
    S_mat = getoverlapmatrix(file)

    # Löwdin transform the Fock matrix
    fock_mat = lowdin_transformed_S(fock_mat, S_mat)

    # block diagonalize the Fock sub-matrices
    C_full, f_diabatic = block_diagonalization_with_bridge_new(fock_mat, N_basis_per_fragment)

    # calculate number of diabatic eigenvalues
    N_eigenvalues = f_diabatic.shape[0]

    # define list with the diabatic eigenvalues
    diabatic_eigenvalues = [f_diabatic[i, i] for i in range(N_eigenvalues)]

    # calculate the number of electrons per fragment
    electrons_per_fragment = getelectronsdonoracceptor(file, N_atoms_per_fragment)

    # the number of electrons on the three charge sites, donor, chromophore and acceptor
    # are calculated. each charge site is allowed to expand upon the nearest bridge(s), and
    # one electron is withdrawn for each bond that is broken to form the fragment, that is,
    # 1 for the donor and acceptor and 2 for the bridge.
    N_donor_electrons = electrons_per_fragment[0] + electrons_per_fragment[1] - 1
    N_chromophore_electrons = electrons_per_fragment[1] + electrons_per_fragment[2] + electrons_per_fragment[3] - 2
    N_acceptor_electrons = electrons_per_fragment[3] + electrons_per_fragment[4] - 1

    # the HOMO and LUMO orbital numbers for the donor, chromophore and acceptor are calculated
    homo_indexes = [int(N_donor_electrons / 2) - 1, N_basis_per_fragment[0] + N_basis_per_fragment[1] + int(N_chromophore_electrons / 2) - 1, N_basis_per_fragment[0] + 2 * N_basis_per_fragment[1] + N_basis_per_fragment[2] + N_basis_per_fragment[3] + int(N_acceptor_electrons / 2) - 1]
    lumo_indexes = [int(N_donor_electrons / 2), N_basis_per_fragment[0] + N_basis_per_fragment[1] + int(N_chromophore_electrons / 2), N_basis_per_fragment[0] + 2 * N_basis_per_fragment[1] + N_basis_per_fragment[2] + N_basis_per_fragment[3] + int(N_acceptor_electrons / 2)]

    # their diabatic orbital energies are calculated
    homo_energies = [diabatic_eigenvalues[homo_indexes[i]] for i in range(len(homo_indexes))]
    lumo_energies = [diabatic_eigenvalues[lumo_indexes[i]] for i in range(len(lumo_indexes))]

    # the orbital coupling elements are calculated in the following. e.g. choosing "[homo_indexes[0], homo_indexes[1]]"
    # will couple the HOMO of site 1 (donor) with the HOMO of site 2 (chromophore)
    # while "[lumo_indexes[1], lumo_indexes[2]]" couples the LUMO of site 2 (chromophore) with the LUMO of site 3 (acceptor)

    # FORWARD COUPLINGS

    # Fock matrix between donor HOMO and chromophore HOMO
    f_system_orthorgonal_D_HOMO_C_HOMO = donor_acceptor_orthorgonalize(C_full, f_diabatic, [homo_indexes[0], homo_indexes[1]])

    # Fock matrix between chromophoro LUMO and acceptor LUMO
    f_system_orthorgonal_C_LUMO_A_LUMO = donor_acceptor_orthorgonalize(C_full, f_diabatic, [lumo_indexes[1], lumo_indexes[2]])

    # print forward couplings:
    print("forward coupling elements:")
    print("donor HOMO and chromophore HOMO: " + str(np.abs(f_system_orthorgonal_D_HOMO_C_HOMO[0, 1])))
    print("chromophore LUMO and acceptor LUMO: " + str(np.abs(f_system_orthorgonal_C_LUMO_A_LUMO[0, 1])))

    # CHARGE RECOMBINATION COUPLINGS

    # Fock matrix between donor HOMO and chromophore LUMO
    f_system_orthorgonal_D_HOMO_C_LUMO = donor_acceptor_orthorgonalize(C_full, f_diabatic, [homo_indexes[0], lumo_indexes[1]])

    # Fock matrix between chromophore HOMO and acceptor LUMO
    f_system_orthorgonal_C_HOMO_A_LUMO = donor_acceptor_orthorgonalize(C_full, f_diabatic, [homo_indexes[1], lumo_indexes[2]])

    # Fock matrix between donor HOMO and acceptor LUMO
    f_system_orthorgonal_D_HOMO_A_LUMO = donor_acceptor_orthorgonalize(C_full, f_diabatic, [homo_indexes[0], lumo_indexes[2]])

    # print charge recombination couplings:
    print("charge recombination coupling elements:")
    print("donor HOMO and chromophore LUMO: " + str(np.abs(f_system_orthorgonal_D_HOMO_C_LUMO[0, 1])))
    print("chromophore HOMO and acceptor LUMO: " + str(np.abs(f_system_orthorgonal_C_HOMO_A_LUMO[0, 1])))
    print("donor HOMO and acceptor LUMO: " + str(np.abs(f_system_orthorgonal_D_HOMO_A_LUMO[0, 1])))

# =====================================================================
#
#                            END OF SCRIPT
#
# =====================================================================
