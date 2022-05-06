#=====================================================================
#
#                    REDFIELD PROPAGATION SCRIPT
#
#=====================================================================

"""
please see manual for how to run the script

execute script by:
$ python3 ./redfield_propagation_vacuum.py plot_name
"""

#=====================================================================
#
#                  PACKAGES AND PHYSICAL CONSTANTS
#
#=====================================================================

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import os
import sys

hbar = 6.58211899 * 10 ** (-16) # [eV*s]
kb = 8.6173303 * 10 ** (-5) # [eV/K]
c = 299792458 * 10 ** 2 # [cm/s]

#=====================================================================
#
#                        CORRELATION FUNCTION
#
#=====================================================================

def spectral_density(omega, sigma, gamma_k_list, gamma_g_list, freq_list):

    """
    this function calculates the spectral density due to the
    intramolecular vibrations

    omega; incoming frequency
    sigma; broadening of the Lorentzian function
    freq_list; frequencies of normal modes
    gamma_k_list; excited state coupling elements from normal modes
    gamma_g_list; ground state coupling elements from normal modes
    """

    # use the absolute value of the frequency
    if omega < 0:
        omega = - omega

    # define spectral density variable
    spectral = 0

    # for each normal mode frequency
    for i in range(len(freq_list)):

        # calculate Huang-Rhys factor
        HR = (gamma_k_list[i] * gamma_g_list[i])

        # add Lorentzian broadening to spectral structure
        lorentzian = HR * (1 / np.pi) * ((1/2) * sigma) / ((omega - freq_list[i]) ** 2 + ((1/2) * sigma) ** 2)

        # summarize the contributions from the spectral density
        spectral += lorentzian

    return spectral

def correlation_function(omega, sigma, gamma_k_list, gamma_g_list, freq_list, T):

    """
    this function calculates the correlation function

    T; temperature
    """

    # convert the frequency unit from [cm^-1] to [Hz]
    omega *= 2 * np.pi * c

    # there is no correlation at zero frequency
    if omega == 0:
        correlation = 0

    else:

        # calculate spectral density. the unit of omega is converted from [Hz] to [cm^-1] in the argument.
        # the output unit is [cm]
        spectral = spectral_density(omega / (2 * np.pi * c), sigma, gamma_k_list, gamma_g_list, freq_list)

        # convert spectral density unit from [cm] to [s]
        spectral = spectral * (1 / (2 * np.pi * c))

        # calculate Bose-Einstein distribution
        q = 1 / (np.exp(hbar * omega / (kb * T)) - 1)

        # calculate correlation function. the output units is [eV^2*s]
        correlation = np.sign(omega) * float(2 * np.pi * hbar ** 2 * omega ** 2 * (1 + q) * spectral)

    return correlation

#=====================================================================
#
#                       FOURIER TRANSFORMATION
#
#=====================================================================

def C_matrix(correlation_function, W, T, sigma, g_matrix, j, k, freq_list):

    """
    this function calculates the fourier transformation of the correlation function

    W; matrix containing the eigenfrequencies of the Hamiltonian
    g_matrix; matrix containing all vibrational coupling elements
    j; index of correlation function
    k; index of correlation function
    """

    # create null-matrix
    C_matrix = np.zeros((len(W), len(W)))

    # fourier transformation
    for n in range(len(W)):
        for m in range(len(W)):
            C_matrix[n,m] = float(correlation_function(W[m,m] - W[n,n], sigma, g_matrix[j], g_matrix[k], freq_list, T))

    return 0.5 * C_matrix

def CT_matrix(correlation_function, W, T, sigma, g_matrix, j, k, freq_list):

    """
    this function calculates the fourier transformation of the transpose of the correlation function
    """

    CT_matrix = np.zeros((len(W), len(W)))

    for n in range(len(W)):
        for m in range(len(W)):
            CT_matrix[n,m] = float(correlation_function(W[n,n] - W[m,m], sigma, g_matrix[j], g_matrix[k], freq_list, T))

    return 0.5 * CT_matrix

#=====================================================================
#
#                    MATRIX ELEMENTS OF q AND q'
#
#=====================================================================

def q_matrix(W, V, basis, S, correlation_function, T, sigma, g_matrix, freq_list):

    """
    this function calculates the matrix elements of \hat{q}
    the output units are [eV^2*s]

    V; unitary matrix used for the transformation of system operators
    basis; matrix of same size as system Hamiltonian
    S; system operator
    """

    # define number of basis vectors
    num_vector = len(basis)

    # create null-matrix to contain the matrix elements of q.
    q_matrix = np.zeros((len(S), len(S)), dtype=object)

    # for each element
    for j in range(len(S)):
        for k in range(len(S)):

            # the correlation function vanish if j != k
            if k != j:
                sub_matrix = np.zeros((num_vector, num_vector))

            else:

                # calculate fourier transformed correlation function
                C_mat = C_matrix(correlation_function, W, T, sigma, g_matrix, j, k, freq_list)

                # calculate q matrix element
                sub_matrix = (V.conj().T@ S[k] @ V) * C_mat

            # construct q matrix
            q_matrix[j][k] = sub_matrix

    return q_matrix

def q_prime_matrix(W, V, basis, S, correlation_function, T, sigma, g_matrix, freq_list):

    """
    this function calculates the matrix elements of \hat{q'}
    the output units are [eV^2*s]
    """

    num_vector = len(basis)

    q_matrix = np.zeros((len(S), len(S)), dtype=object)

    for j in range(len(S)):
        for k in range(len(S)):

            if k != j:
                sub_matrix = np.zeros((num_vector, num_vector))

            else:
                CT_mat = CT_matrix(correlation_function, W, T, sigma, g_matrix, j, k, freq_list)
                sub_matrix = (V.conj().T @ S[k] @ V) * CT_mat

            q_matrix[j][k] = sub_matrix
    
    return q_matrix

#=====================================================================
#
#                          REDFIELD TENSOR
#
#=====================================================================

def summation(eye, S, q, q_prime, V):

    """
    this function calculates the summation term entering the Redfield tensor

    eye; identity matrix
    q; q-matrix
    q_prime; q'-matrix
    """

    # define summation variable
    summation = 0

    # j and k are the system operator indices
    for j in range(len(S)):
        for k in range(len(S)):

            # calculate summation term entering the Redfield tensor
            element = (-np.kron(eye, S[j] @ V @ q[j][k] @ V.conj().T) + np.kron(S[j].T, V @ q[j][k] @ V.conj().T) - np.kron(S[j].T @ V.conj() @ q_prime[j][k].T @ V.T, eye) + np.kron(V.conj() @ q_prime[j][k].T @ V.T, S[j]))
            summation += element

    return summation

def redfield_tensor(H, S, correlation_function, T, basis, sigma, g_matrix, freq_list):

    """
    this function calculates the Redfield tensor

    H; system Hamiltonian matrix
    """

    # calculate eigenvalues and eigenvectors of system Hamiltonian
    eps, V = np.linalg.eigh(H)

    # convert units of eigenvalues from [eV] to [cm^-1]
    W = np.diag(eps) * 8065.6

    # calculate q and q' matrix elements
    qmatrix = q_matrix(W, V, basis, S, correlation_function, T, sigma, g_matrix, freq_list)
    qprimematrix = q_prime_matrix(W, V, basis, S, correlation_function, T, sigma, g_matrix, freq_list)

    # define identy matrix of same dimension as the system Hamiltonian
    row, _ = H.shape
    eye = np.identity(row)

    # calculate summation term entering the Redfield tensor
    sum_qs = summation(eye, S, qmatrix, qprimematrix, V)

    # calculate Redfield tensor
    redfield_tensor = ( (1.0j) / hbar * (np.kron(H.conj().T, eye) - np.kron(eye, H)) ) + 1 / (hbar ** 2) * sum_qs

    return redfield_tensor

#=====================================================================
#
#                   SITE BASIS AND SYSTEM OPERATOR
#
#=====================================================================

def basis_state(n, m):

    """
    this function creates the matrices representing the diabatic states

    n; dimension of system Hamiltonian
    m; state of basis
    """

    state = np.zeros(n)
    state[m] = 1

    return state

def system_operator(m, n):

    """
    this function sets up the system operators for the states

    m; basis state 1
    n; basis state 2
    """

    # the system operator is a dimension_H x dimension_H
    # matrix. all but one element is zero. e.g. for basis state 0 (GS)
    # the first diagonal element, that is, (0,0), is 1
    S = np.outer(basis[m], basis[n])

    return np.array(S)

#=====================================================================
#
#                          PLOT POPULATION
#
#=====================================================================

def plot_population(N, populations, plot_name):

    """
    this function plots the population transfer

    N; dimension of system Hamiltonian / number of states
    populations; n_times x (dimension_H x dimension_H) matrix containing the populations
                 of the reduced density matrix
    plot_name; name of the .pdf file created with the plot
    """

    # since the density matrix is transformed into the Fock-Liouville space, the population
    # elements must be extracted from the full population matrix. the system operator S will
    # be defined as a list consisting of five 5 x 5 matrices, with only one non-zero
    # element equal to one at the diagonal elements corresponding to the state. the ground
    # state thus has an element at the position (0,0) in the first matrix, which is the 0th
    # column in the entire list. the first excited state has an element at (1,1) in the second
    # matrix, which is at the 5 + 1 = 6th column in the entire list. following this procedure,
    # the position numbers of the charge sites are calculated and appended to this empty list.
    population_list = []

    for i in range(N):

        # calculate positions used for the extraction of populations
        relevant_column_in_populations = i * N + i
        population_list.append(relevant_column_in_populations)

    # label the charge sites
    labels = [r"D-An-Ac", r"D$^+$-An-Ac$^-$", r"D-An$^*$-Ac", r"D$^+$-An$^-$-Ac", r"D-An$^+$-Ac$^-$"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    # define plot environment
    ax = plt.subplot(111)

    # define counter to pick out the population of each charge site
    k = 0

    # plot the population transfer for each charge site
    for i in population_list:

        # the ith column of the population matrix is plotted
        ax.plot(time_list, np.abs(populations[:, i]), '-', label=labels[k], color=colors[k])

        # update counter
        k += 1

    # format tick labels
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)

    # shrink linewidth of frames
    ax.spines["top"].set_linewidth(0.25)
    ax.spines["left"].set_linewidth(0.25)
    ax.spines["right"].set_linewidth(0.25)
    ax.spines["bottom"].set_linewidth(0.25)
    ax.tick_params(axis='both', width=0.25)

    # shrink current axis' height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])

    # put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=3)

    # set tile + labels and save plot
    plt.title("Population Transfer (Vacuum)")
    plt.xlabel("Time / [s]")
    plt.ylabel("Population")
    plt.savefig(plot_name)

# =====================================================================
#
#                          FMR-B HAMILTONIAN
#
# =====================================================================

def fmr_hamiltonian(g_matrix):

    """
    this function constructs the system Hamiltonian.
    the energies and coupling elements are manually added.
    both the energies and couplings are given in [eV]
    """

    # energies
    eps0 = 0.0
    eps1 = 2.97
    eps2 = 3.42
    eps3 = 3.21
    eps4 = 3.74

    # couplings from charge recombination
    g01 = 0.000186 * 10 ** -3
    g03 = 0.396 * 10 ** -3
    g04 = 0.926 * 10 ** -3

    # couplings from forward propagation
    g24 = 0.793 * 10 ** -3
    g23 = 34.9 * 10 ** -3
    g13 = g24
    g14 = g23

    # create null-matrix for the system Hamiltonian
    H = np.zeros((5, 5))

    # insert diabatic state energies
    H[0, 0] = eps0  # ground state
    H[1, 1] = eps1  # d+-an-ac-
    H[2, 2] = eps2  # d-an*-ac
    H[3, 3] = eps3  # d+-an--ac
    H[4, 4] = eps4  # d-an+-ac-

    # insert electronic couplings
    H[0, 1] = H[1, 0] = g01
    H[0, 3] = H[3, 0] = g03
    H[0, 4] = H[4, 0] = g04
    H[2, 4] = H[4, 2] = g24
    H[2, 3] = H[3, 2] = g23
    H[1, 3] = H[3, 1] = g13
    H[1, 4] = H[4, 1] = g14

    return np.array(H), np.array(g_matrix)

#=====================================================================
#
#                        CALCULATION SETTINGS
#
#=====================================================================

# temperature
T = 298.15 # [K]

# width of Lorentzian broadening for the spectral density
sigma = 10 # [cm^-1]

# dimension of system Hamiltonian
dimension_H = 5

# time interval for propagation
time_list = np.arange(1 * 10 ** (-15), 1 * 10 ** (-12), 10 ** (-17)) # picosecond timescale
#time_list = np.arange(1 * 10 ** (-15), 1 * 10 ** (-9), 10 ** (-14)) # nanosecond timescale
#time_list = np.arange(1 * 10 ** (-15), 1 * 10 ** (-6), 10 ** (-11)) # microsecond timescale
#time_list = np.arange(1 * 10 ** (-15), 1 * 10 ** (-3), 10 ** (-8)) # millisecond timescale

# steps
n_steps = len(time_list)

#=====================================================================
#
#                            RUN PROGRAM
#
#=====================================================================

if __name__ == '__main__':

    # generate basis
    basis = [basis_state(dimension_H, 0), basis_state(dimension_H, 1), basis_state(dimension_H, 2), basis_state(dimension_H, 3), basis_state(dimension_H, 4)]
    basis = np.array(basis)
    basis = basis.T

    # define system operator - list of five 5 x 5 matrices.
    # the system operator should not work on the ground state,
    # and the corresponding matrix is thus multiplied with zero
    S = np.array([0 * system_operator(0,0), system_operator(1,1), system_operator(2,2), system_operator(3,3), system_operator(4,4)])

    # find coupling.cvs files with vibrational coupling elements
    files = [filename for filename in os.listdir(".") if filename.startswith("coupling")]
    files.sort()

    # create null-matrix for collection of vibrational coupling elements
    g_matrix = np.zeros(len(basis), dtype=object)

    # write vibrational coupling elements into the g-matrix
    for i, filename in enumerate(files):

        # read the frequencies of the normal modes.
        # the frequencies are listed in all of the
        # coupling.cvs files. they are extracted from
        # the first file
        if i == 0:
            with open(filename, "r") as file:
                freq = file.readline().split(",")
                freq = np.array(freq,dtype=float)

        # read all vibrational coupling elements and
        # define g_matrix. they are listed on the second
        # line in the coupling.cvs files
        with open(filename, "r") as file:
            file.readline()
            g = file.readline().split(",")
            g = np.array(g, dtype=float)
            g_matrix[i+1] = g

    # the ground state vibrational coupling elements are all zero
    g_matrix[0] = np.zeros(len(freq))

    # construct system Hamiltonian, and turn g-matrix into an array
    H, g_matrix = fmr_hamiltonian(g_matrix)

    # get the dimensionality of one row of the Hamiltonian
    row = H.shape[0]

    # define array to hold the reduced electron density in the Fock-Liouville space
    rho_start = np.zeros(row ** 2)

    # the population of charge state D-An^{*}-A is set to 1
    # since the Redfield propagation is initiated from this charge site
    rho_start[12] = 1.0

    # calculate the Redfield tensor
    R = redfield_tensor(H, S, correlation_function, T, basis, sigma, g_matrix, freq)

    # define empty list for the populations
    populations = []

    # for each time step
    for m in range(n_steps):

        # approximate matrix exponential as Padé approximant
        exp_R = la.expm(R * time_list[m])

        # propagate the reduced density matrix in time
        rho = exp_R @ rho_start

        # append value of reduced density matrix
        populations.append(rho)

    # turn the list into an array
    populations = np.array(populations)

    # create filename for the plot
    plot_fmr = str(sys.argv[1]) + ".pdf"

    # plot the population transfer
    plot_population(row, populations, plot_fmr)

#=====================================================================
#
#                           END OF SCRIPT
#
#=====================================================================
