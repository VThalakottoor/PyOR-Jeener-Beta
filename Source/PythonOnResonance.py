# Python On Resonance (PyOR)
# Author: Vineeth Francis Thalakottoor Jose Chacko
# Email: vineethfrancis.physics@gmail.com
# There can be mistakes. If you see anything please write to me.
# PyOR is for the beginners (with basic knowledge of matrices, spin operators, and Python programming).
# "Anbe Sivam": love/compassion is god
"""
Version: Jeener-B-24.08.24
Radiation Damping, Raser/Maser (single and multi-mode) - Removed from the beta version, will reappear in version 1.
"""

# ---------- Package

import numpy as np
from numpy import linalg as lina

import sympy as sp
from sympy import *
from sympy.physics.quantum.cg import CG

import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.cm as cm
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import axes3d

import time
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import os
import sys
#sys.setrecursionlimit(1500)

import numba
from numba import njit, cfunc

from IPython.display import display, Latex, Math

from fractions import Fraction
# ---------- Package

class Numerical_MR:

    def __init__(self,Slist,hbarEQ1):
    
        # Physical Constants
        self.pl = 6.626e-34 # Planck Constant; J s
        self.hbar = 1.054e-34 # Planck Constant; J s
        self.hbarEQ1 = hbarEQ1
        self.ep0 = 8.854e-12 # Permitivity of free space; F m^-1
        self.mu0 = 4 * np.pi * 1.0e-7 # Permeabiltiy of free space; N A^-2
        self.kb = 1.380e-23 # Boltzmann Constant; J K^-1
    
        # 1D list of spin values
        self.Slist = Slist
        self.S = np.asarray(Slist)
    
        # Number of Spins
        self.Nspins = self.S.shape[-1]
        
        # Array of dimensions of individual Hilbert Space    
        Sdim = np.zeros(self.S.shape[-1],dtype='int') 
        for i in range(self.S.shape[-1]): 
            Sdim[i] = np.arange(-self.S[i],self.S[i]+1,1).shape[-1]
        self.Sdim = Sdim    
    
        # Dimension of Hilbert Space
        self.Vdim = np.prod(self.Sdim) 
        
        # Dimenion of Liouville Space
        self.Ldim = (self.Vdim)**2
        
        # Gyromagnetic ratio
        self.gammaE = -1.761e11 # Electron; rad s^-1 T^-1
        self.gammaH1 = 267.522e6 # Proton; rad s^-1 T^-1
        self.gammaH2 = 41.065e6 # Deuterium; rad s^-1 T^-1
        self.gammaC13 = 67.2828e6 # Carbon; rad s^-1 T^-1
        self.gammaN14 = 19.311e6 # Nitrogen 14; rad s^-1 T^-1
        self.gammaN15 = -27.116e6 # Nitrogen 15; rad s^-1 T^-1
        self.gammaO17 = -36.264e6 # Oxygen 17; rad s^-1 T^-1
        self.gammaF19 = 251.815e6 # Flurine 19; rad s^-1 T^-1  
        
        # Plotting Label
        self.PlotLabel_Hilbert = True
        self.UserDefined_Label = False
        
        # Redkite Plot Label
        self.Redkite_Label = False
        self.Redkite_Label_SpinDynamica = False
        
        # Relaxation define spins
        self.spin1 = 0
        self.spin2 = 1
        
        # Print Larmor Frequency
        self.print_Larmor = True
        
        # ODE Methods
        self.ode_method ='RK45'
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Functions to generate the Spin System
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

    def SpinOperatorsSingleSpin(self,X):
        """
		Generate spin operators for a given spin: Sx, Sy and Sz
		INPUT
		-----
		X : Spin quantum number
		
		OUTPUT
		------
		SingleSpin : [Sx,Sy,Sz]
		"""   
		  
        # 1D Array: magnetic qunatum number for spin S (order: S, S-1, ... , -S)
        ms = np.arange(X,-X-1,-1)  
        
        # Initialize Sx, Sy and Sz operators for a spin, S
        SingleSpin = np.zeros((3,ms.shape[-1],ms.shape[-1]),dtype=np.clongdouble)
        
        # Intitialze S+ and S- operators for spin, S
        Sp = np.zeros((ms.shape[-1],ms.shape[-1]),dtype=np.clongdouble)
        Sn = np.zeros((ms.shape[-1],ms.shape[-1]),dtype=np.clongdouble)
        
        # Calculating the <j,m'|S+|j,m> = hbar * sqrt(j(j+1)-m(m+1)) DiracDelta(m',m+1) and
        # <j,m'|S-|j,m> = hbar * sqrt(j(j+1)-m(m-1)) DiracDelta(m',m-1)  
        Id = np.identity((ms.shape[-1])) 
        
        ## Calculate DiracDelta(m',m+1)
        ## Shifter right Identity operator
        Idp = np.roll(Id,1,axis=1) 
        ## Upper triangular martix
        Idp = np.triu(Idp,k=1) 
        
        ## Calculate DiracDelta(m',m-1)
        ## Shifter left Identity operator
        Idn = np.roll(Id,-1,axis=1) 
        ## Lower triangular matrix
        Idn = np.tril(Idn,k=1) 
        
        ## Calculating S+ and S- operators for spin, S
        for i in range(ms.shape[-1]):
            for j in range(ms.shape[-1]):
            
                # Sz operator, Row ordering (top to bottom): |j,S>, |j,S-1>,... , |j,-S> 
                SingleSpin[2][i][j] = self.hbar * ms[j]*Id[i][j] 
                
                # S+ operator, Row ordering (top to bottom): |j,S>, |j,S-1>,... , |j,-S>  
                Sp[i][j] = np.sqrt(X*(X+1) - ms[j]*(ms[j]+1)) * Idp[i][j] 
                # S- operator, Row ordering (top to bottom): |j,S>, |j,S-1>,... , |j,-S> 
                Sn[i][j] = np.sqrt(X*(X+1) - ms[j]*(ms[j]-1)) * Idn[i][j] 
        
        # Sx operator
        SingleSpin[0] = self.hbar * (1/2.0) * (Sp + Sn) 
        # Sy operator
        SingleSpin[1] = self.hbar * (-1j/2.0) * (Sp - Sn)       
        
        if self.hbarEQ1:
            SingleSpin = SingleSpin / self.hbar
        return SingleSpin
        
    def SpinOperator(self):
        """
		Generate spin operators for all spins: Sx, Sy and Sz
		INPUT
		-----
		nill
		
		OUTPUT
		------
		Sx : array [Sx of spin 1, Sx of spin 2, Sx of spin 3, ...]
		Sy : array [Sy of spin 1, Sy of spin 2, Sy of spin 3, ...]
		Sz : array [Sz of spin 1, Sz of spin 2, Sz of spin 3, ...]
		"""
		
        # Sx operator for individual Spin, Sx[i] corresponds to ith spin
        Sx = np.zeros((self.Nspins,self.Vdim,self.Vdim),dtype=np.cdouble) 
        # Sy operator for individual Spin, Sy[i] corresponds to ith spin
        Sy = np.zeros((self.Nspins,self.Vdim,self.Vdim),dtype=np.cdouble)
        # Sz operator for individual Spin, Sz[i] corresponds to ith spin
        Sz = np.zeros((self.Nspins,self.Vdim,self.Vdim),dtype=np.cdouble)
        
        # Calculating Sx, Sy and Sz operators one by one
        for i in range(self.Nspins): 
            VSlist_x = [] 
            VSlist_y = []
            VSlist_z = []
            # Computing the Kronecker product of all sub Hilbert space
            for j in range(self.Nspins):  
                # Making array of identity matrix for corresponding sub vector space
                VSlist_x.append(np.identity(self.Sdim[j])) 
                VSlist_y.append(np.identity(self.Sdim[j]))
                VSlist_z.append(np.identity(self.Sdim[j]))
            
            # Replace ith identity matrix with ith Sx,Sy and Sz operators    
            VSlist_x[i] = self.SpinOperatorsSingleSpin(self.Slist[i])[0]  
            VSlist_y[i] = self.SpinOperatorsSingleSpin(self.Slist[i])[1]
            VSlist_z[i] = self.SpinOperatorsSingleSpin(self.Slist[i])[2]
            
            # Kronecker Product Calculating
            Sx_temp_x = VSlist_x[0]
            Sy_temp_y = VSlist_y[0]
            Sz_temp_z = VSlist_z[0]
            for k in range(1,self.Nspins):
                Sx_temp_x = np.kron(Sx_temp_x,VSlist_x[k])
                Sy_temp_y = np.kron(Sy_temp_y,VSlist_y[k]) 
                Sz_temp_z = np.kron(Sz_temp_z,VSlist_z[k]) 
            Sx[i] = Sx_temp_x
            Sy[i] = Sy_temp_y
            Sz[i] = Sz_temp_z
        return Sx, Sy, Sz
        
    def PMoperators(self,Sx,Sy):
        """
		Generate spin operators for all spins: Sp (Sx + j Sy) and Sm (Sx - j Sy)
		INPUT
		-----
		Sx, Sy
		
		OUTPUT
		------
		Sp : array [Sp of spin 1, Sp of spin 2, Sp of spin 3, ...]
		Sm : array [Sm of spin 1, Sm of spin 2, Sm of spin 3, ...]
		"""
		    
        Sp = np.zeros((self.Nspins,self.Vdim,self.Vdim),dtype=np.cdouble)
        Sm = np.zeros((self.Nspins,self.Vdim,self.Vdim),dtype=np.cdouble)
        for i in range(self.Nspins):
            Sp[i] = Sx[i] + 1j * Sy[i]
            Sm[i] = Sx[i] - 1j * Sy[i]
        return Sp, Sm   
        
    def MagQnu(self,X):
        """
        Magnetic Quantum number of individual spins
        
        INPUT
        -----
        X : Spin Quantum Number, Integer
        
        OUTPUT
        ------
        return Magnetic quantum numbers, X, X-1, X-2,.., -X
        """
        return np.arange(X,-X-1,-1)

    def Basis_Ket_AngularMomentum(self,Sz,formating):
        """
        Magnetic quantum number of each state; Sz | i > = m_i | i > or < i | Sz | i > = m_i
        
        INPUT
        -----
        Sz: Spin operator, Sz
        Formating of output: 'array' or 'list'
        
        OUTPUT
        ------
        Return magnetic quantum number of each state, as 'array' or 'list'
        """
        
        if formating == 'array':
            return (np.sum(Sz,axis=0).real).diagonal()
            
        if formating == 'list':
            array = (np.sum(Sz,axis=0).real).diagonal()
            List = []
            for i in range(array.shape[-1]):
                List.append(str(Fraction(array[i])))
            return(List)         
        
    def Basis_Ket(self):
        """
        Return a list of all the Basis kets
        """
        LABEL = []
        LABEL_temp = []
        for i in range(self.Nspins):
            locals()["Spin_List_"+str(i)] = []
        dummy = 0
        for j in self.Slist:
            for k in self.MagQnu(j):
                locals()["Spin_List_" + str(dummy)].append("|"+str(Fraction(j))+","+str(Fraction(k))+">")
            dummy = dummy + 1    
            
        def Combine(A,B):
            for l in A:
                for m in B:
                    LABEL_temp.append(l+m)
            return LABEL_temp
            
        if self.Nspins == 1:
            LABEL =  locals()["Spin_List_" + str(0)]
            return LABEL
        
        if self.Nspins >= 2:                 
            for n in range(self.Nspins - 1):
                locals()["Spin_List_" + str(n+1)] = Combine(locals()["Spin_List_" + str(n)] ,locals()["Spin_List_" + str(n+1)])   
                LABEL_temp = []        
            LABEL = locals()["Spin_List_" + str(self.Nspins - 1)]
            return LABEL   

    def Basis_Bra(self):
        """
        Return a list of all the Basis Bras
        """
        LABEL = []
        LABEL_temp = []
        for i in range(self.Nspins):
            locals()["Spin_List_"+str(i)] = []
        dummy = 0
        for j in self.Slist:
            for k in self.MagQnu(j):
                locals()["Spin_List_" + str(dummy)].append("<"+str(Fraction(j))+","+str(Fraction(k))+"|")
            dummy = dummy + 1    
            
        def Combine(A,B):
            for l in A:
                for m in B:
                    LABEL_temp.append(l+m)
            return LABEL_temp
            
        if self.Nspins == 1:
            LABEL =  locals()["Spin_List_" + str(0)]
            return LABEL
        
        if self.Nspins >= 2:                 
            for n in range(self.Nspins - 1):
                locals()["Spin_List_" + str(n+1)] = Combine(locals()["Spin_List_" + str(n)] ,locals()["Spin_List_" + str(n+1)])   
                LABEL_temp = []        
            LABEL = locals()["Spin_List_" + str(self.Nspins - 1)]
            return LABEL 
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Halitonian of the Spin System
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
    def Print_Larmor(self,print_Larmor):
        """
        Print Lamor Frequency or not in the function: LarmorFrequency
        By default, True
        """
        self.print_Larmor = print_Larmor
    
    def LarmorFrequency(self,Gamma,B0,Offset):
        """
        Generate Larmor Frequency, Omega0 in Lab Frame
        
        INPUT
        -----
        Gamma: List of Gyromagnetic ratios of individual spins
        B0: Field of the spectrometer in Tesla
        Offset: List of the chemical shifts of individual spins
        
        OUTPUT
        ------
        return array of Larmor frequencies of individual spins in lab frame
        """
        W0 = np.zeros((self.Nspins))
        gamma = np.asarray(Gamma)
        offset = np.asarray(Offset)
        for i in range(self.Nspins):
            W0[i] = -1 * gamma[i] * B0 - 2 * np.pi * offset[i]
        
        if self.print_Larmor:
            print("Larmor Frequency in MHz: ", W0/2.0/np.pi/1.0e6)    
        return W0    
        
    def Zeeman(self, LarmorF, Sz):
        """
        Generating Zeeman Hamiltonian in Lab Frame
        
        INPUT
        ----
        LarmorF: Array of Larmor frequencies of individual spins in lab frame (LarmorF = System.LarmorFrequency(Gamma,B0,Offset))
        Sz: Sz spin operators
        
        OUTPUT
        ------
        HZ: Zeeman hamiltonian in lab Frame (Angluar frequency Units) 
        """

        Hz = np.zeros((self.Vdim,self.Vdim),dtype=np.double)
        for i in range(self.Nspins):
            Hz = Hz + LarmorF[i] * Sz[i]
                
        return Hz

    def Zeeman_RotFrame(self, LarmorF, Sz, OmegaRF):
        """
        Generating Zeeman Hamiltonian in Rotating Frame
        
        INPUTS
        ------
        LarmorF: Array of Larmor frequencies of individual spins in lab frame (LarmorF = System.LarmorFrequency(Gamma,B0,Offset))
        Sz: Sz spin operators
        OmegaRF: List of rotating frame frequencies 
                 Homonuclear case - All frequencies are the same
                 Hetronuclear case - ??

        OUTPUT
        ------
        HZ: Zeeman hamiltonian in rotating Frame  (Angluar frequency Units)       
        """
        
        omegaRF = np.asarray(OmegaRF)
        Hz = np.zeros((self.Vdim,self.Vdim),dtype=np.double)
        for i in range(self.Nspins):
            Hz = Hz + (LarmorF[i]-omegaRF[i]) * Sz[i]
        return Hz
                
    def Zeeman_B1(self,Sx,Sy,Omega1,Omega1Phase):  
        """
        Generating Zeeman Hamiltonian with B1 Hamiltonian
        
        INPUT
        -----
        Sx: Sx spin operators
        Sy: Sy spin operators
        Omega1: List of amplitude of RF signal in Hz (nutation frequency)
        Omega1Phase: List of Phase of RF signal in deg
                
        OUTPUT
        ------
        HzB1: B1 field hamiltonian (Angluar frequency Units) 
        """  
        
        HzB1 = np.zeros((self.Vdim,self.Vdim),dtype=np.double)
        omega1 = 2*np.pi*np.asarray(Omega1)
        Omega1Phase = np.pi*np.asarray(Omega1Phase)/180.0
        for i in range(self.Nspins):
            HzB1 = HzB1 + omega1[i] * (Sx[i]*np.cos(Omega1Phase[i]) + Sy[i]*np.sin(Omega1Phase[i]))
        return HzB1
        
    def Jcoupling(self,J,Sx,Sy,Sz):    
        """
        Generate J coupling Hamiltonian    
        
        INPUT
        -----
        J: J coupling constant (Hz)
        Sx: Sx spin operators
        Sy: Sy spin operators
        Sz: Sz spin operators 
        
        OUTPUT
        ------
        Hj: J coupling Hamiltonian (Angluar frequency Units) 
        """ 
        
        J = 2*np.pi*J    
        Hj = np.zeros((self.Vdim,self.Vdim),dtype=np.double)
        for i in range(self.Nspins):
            for j in range(self.Nspins):
                Hj = Hj + J[i][j] * (np.matmul(Sx[i],Sx[j]) + np.matmul(Sy[i],Sy[j]) + np.matmul(Sz[i],Sz[j]))      
        return Hj        

    def Jcoupling_Weak(self,J,Sz):    
        """
        Generate J coupling Hamiltonian (weak)

        INPUT
        -----
        J: J coupling constant (Hz)
        Sz: Sz spin operators 
        
        OUTPUT
        ------
        Hj: J coupling Hamiltonian (weak)  (Angluar frequency Units)             
        """ 
        
        J = 2*np.pi*J    
        Hj = np.zeros((self.Vdim,self.Vdim),dtype=np.double)
        for i in range(self.Nspins):
            for j in range(self.Nspins):
                Hj = Hj + J[i][j] * np.matmul(Sz[i],Sz[j])      
        return Hj 
                
    def DDcoupling(self,Sx,Sy,Sz,Sp,Sm,bIS,theta,phi,secular):
        """
        Generate Dipole-Dipole coupling Hamiltonian
        INPUT
        -----
        bIS : - mu0 * Gamma(I) * Gamma(S) * hbar /(4 * PI * (rIS)**3)
        mu0 : permiability of free space
        Gamma(I/S) : Gyromagnetic ratio of Spin I/S 
        secular: If True secular approximation is used
                 
        OUTPUT     
        ------
        Hdd : Dipole-Dipole coupling Hamitonian of the system (Angluar frequency Units)         
        """ 
        
        theta = np.pi*theta/180.0
        phi = np.pi*phi/180.0
        
        if secular:
            A = np.matmul(Sz[0],Sz[1]) * (3 * (np.cos(theta))**2 - 1)
            B = C = D = E = F = 0
        else:     
            A = np.matmul(Sz[0],Sz[1]) * (3 * (np.cos(theta))**2 - 1)
            B = (-1/4) * (np.matmul(Sp[0],Sm[1]) + np.matmul(Sm[0],Sp[1])) * (3 * (np.cos(theta))**2 - 1)
            C = (3/2) * (np.matmul(Sp[0],Sz[1]) + np.matmul(Sz[0],Sp[1])) * np.sin(theta) * np.cos(theta) * np.exp(-1j*phi)
            D = (3/2) * (np.matmul(Sm[0],Sz[1]) + np.matmul(Sz[0],Sm[1])) * np.sin(theta) * np.cos(theta) * np.exp(1j*phi)
            E = (3/4) * np.matmul(Sp[0],Sp[1]) * (np.sin(theta))**2 * np.exp(-1j * 2 * phi)
            F = (3/4) * np.matmul(Sm[0],Sm[1]) * (np.sin(theta))**2 * np.exp(1j * 2 * phi)
                            
        Hdd = np.zeros((self.Vdim,self.Vdim),dtype=np.double)

        Hdd = 2.0*np.pi * bIS * (A+B+C+D+E+F)     
        return Hdd 
        
    def Convert_EnergyTOFreqUnits(self,H):
        """
        Convert Hamiltonian from Energy Unit to Frequency Unit

        INPUT
        -----
        H: Hamiltonian (Joules units)
        
        OUTPUT
        ------ 
        H: Hamiltonian (Angular Frequency Units)       
        """
        
        return H/self.hbar    

    def Convert_FreqUnitsTOEnergy(self,H):
        """
        Convert Hamiltonian from Frequency Unit to Energy Unit
        
        INPUT
        -----
        H: Halitonian (Angular Frequency Units)
        
        OUTPUT
        ------
        H: Hamiltonian (Joules)
        
        """
        
        return H * self.hbar 

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Eigen Values and Vectors 
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

    def Eigen(self,H):
        """
        Eigen Values and Vectors
        
        INPUT
        -----
        H: Hamiltonian
        
        OUTPUT
        ------
        return eigenvalues, eigenvectors 
        """
        
        eigenvalues, eigenvectors = lina.eig(H)
        return eigenvalues, eigenvectors    

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Vector Basis Hilbert Space
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
    def Transform_StateBasis(self,old,new):
        """
        Change Basis state from one to other: Function return the transformation matrix
        | new > = U | old >
        O_new = U O_old U_dagger
        INPUT
        -----
        old: array of old Basis state
        New: array of new Baisis state
        
        OUTPUT
        ------
        return transformation matrix
        """
        
        dim = old.shape[0]
        U = np.zeros((dim,dim))
        for i in range(dim):
            for j in range(dim):
                U[i][j] = self.Adjoint(new[i]) @ old[j]
        return U 
        
    def Operator_BasisChange(self,O,U):
        """
        Change the Operator basis
        
        INPUT
        -----
        O: Old operator
        U: Basis Transformation matric
        
        OUTPUT
        ------
        return basis transformed operator
        """
        
        return U @ O @ self.Adjoint(U)

    def SpinOperator_BasisChange(self,Sop,U):
        """
        Change the Spin Operator basis
        
        INPUT
        -----
        O: array of old spin operators
        U: Basis Transformation matric
        
        OUTPUT
        ------
        return transformed spin operator
        """
        
        dim = Sop.shape[0]
        Sop_N = np.zeros(Sop.shape,dtype=complex)
        for i in range(dim):
            Sop_N[i] = U @ Sop[i] @ self.Adjoint(U) 
        return Sop_N           
                
    def ZBasis_H(self,Hz):
        """"
        Zeeman Basis
        INPUT
        -----
        Hz: Zeman Hamiltonian (lab frame)
        
        OUTPUT
        ------
        return BZ (eigen vectors of Zeman Hamiltonian (lab frame): Bz[0] first eigen vector, Bz[1] second eigen vector, ... )
        """
        
        Bz = np.zeros((self.Vdim,self.Vdim,1))
        eigenvalues, eigenvectors = lina.eig(Hz) 
        for i in range(self.Vdim):
            Bz[i] = (eigenvectors[:,i].reshape(-1,1)).real
        if self.Nspins == 1:
            labelx,labely =  self.XYlabel_1spin(self.S[0])
            #display(Latex(r'Basis: $\alpha$, $\beta$'))
            display(Latex(','.join(labelx)))  
        if self.Nspins == 2:
            labelx,labely =  self.XYlabel_2spin(self.S[0],self.S[1])
            #display(Latex(r'Basis: $\alpha \alpha$, $\alpha \beta$, $\beta \alpha$, $\beta \beta$'))
            display(Latex(','.join(labelx)))           
        return Bz 
        
    def STBasis(self,Bz):
        """
        Singlet Triplet Basis (Two Spin Half Only)
        INPUT
        -----
        Bz: Zeeman eigen states (Output of function 'ZBasis_H')
        
        OUTPUT
        ------
        return Singlet Triplet basis      
        """
        
        if ((self.Nspins == 2) and (self.S[0] == 1/2) and (self.S[1] == 1/2)):
            Bst = np.zeros((self.Vdim,self.Vdim,1))
            Bst[0] = Bz[0] # Tm
            Bst[1] = (1/np.sqrt(2)) * (Bz[1] + Bz[2]) 
            Bst[2] = Bz[3]
            Bst[3] = (1/np.sqrt(2)) * (Bz[1] - Bz[2])  
            display(Latex(r'Basis: $T_{-}$, $T_{0}$,$T_{+}$,$S_{0}$'))  
            return Bst
        else:
            print("Two spin half system only")                
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Operator Basis Hilbert Space
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

    def Adjoint(self,A):
        """
        Return adjoint of operator A

        INPUT
        -----
        A: an operator or vector
        
        OUTPUT
        ------  
        return adjoint of operator or vector      
        """
        
        return A.T.conj()

    def OP_InnerProduct(self,A,B):
        """
        Inner Product
        
        INPUT
        -----
        A: Operator or vector
        B: Operator or vector
        
        OUTPUT
        ------
        return iiner product of the operators or vectors        
        """
        return np.trace(np.matmul(A.T.conj(),B))
        
    def DensityMatrix_Components(self,A,rho):
        """
        Components of density matrix in the Basis Operators, inner product of A_i and rho, where A_i is the ith basis operator
        
        INPUT
        -----
        A: Basis operators (array containing basis operators)
        rho: Density matrix
        OUTPUT
        ------        
        return projection or component of density matrix in the Basis Operators         
        """
        no_Basis = A.shape[0]
        components = np.zeros((no_Basis),dtype=np.cdouble)    
        for i in range(no_Basis):
            components[i] = self.OP_InnerProduct(A[i],rho)
        tol = 1.0e-5 # make elements lower than 'tol' into zero 
        components.real[abs(components.real) < tol] = 0.0  
        components.imag[abs(components.real) < tol] = 0.0  
        return np.round(components.real,3)

    def DensityMatrix_Components_Dictionary(self,A,dic,rho):
        """
        Components of density matrix in the Basis Operators, inner product of A_i and rho, where A_i is the ith basis operator
        
        INPUT
        -----
        A: Basis operators (array containing basis operators)
        dic: Dictionary of Spin Operators
        rho: Density matrix
        OUTPUT
        ------        
        return projection or component of density matrix in the Basis Operators         
        """
        no_Basis = A.shape[0]
        components = np.zeros((no_Basis),dtype=np.cdouble)    
        for i in range(no_Basis):
            components[i] = self.OP_InnerProduct(A[i],rho)
        tol = 1.0e-5 # make elements lower than 'tol' into zero 
        components.real[abs(components.real) < tol] = 0.0  
        components.imag[abs(components.real) < tol] = 0.0 
        density_out = ["Density Matrix = "]
        for i in range(no_Basis):
            density_out.append(str(round(components[i].real,3)) + " " + dic[i] + " + ") 
        print(''.join(density_out))
        
    def Matrix_Tol(self,M,tol):
        """
        Make very small values of a matrix to zero
        
        INPUT
        -----
        M: Matrix
        tol: Tolarance, below which matrix element will be zero
        
        OUTPUT
        ------
        return new matrix 
        """       
        
        M.real[abs(M.real) < tol] = 0.0
        M.imag[abs(M.imag) < tol] = 0.0
        return M
        
    def OP_Normalize(self,A):
        """
        Normalize 
        
        INPUT
        -----
        A: Operator
        OUTPUT
        ------        
        return normalized operator, that means: inner product of A and A equals 1
        """
        return A/np.sqrt(self.OP_InnerProduct(A,A))   

    def CG_Coefficient(self,j1,m1,j2,m2,J,M):
        """
        Clebsch-Gordan Coefficients
        
        INPUT
        -----
        j1: spin quantum number particle 1
        m1: magnetic quantum number particle 1
        j2: spin quantum number particle 2
        m2: magnetic quantum number particle 2    
        J: Total spin quantum number
        M: total magnetic quantum number    
        OUTPUT
        ------  
        return Clebsch-Gordan Coefficients        
        """
        
        return float(CG(j1, m1, j2, m2, J, M).doit())
        

    def Spherical_OpBasis(self,S):
        """
        Spherical Operator Basis
        
        INPUT
        -----
        S: spin quantum number
        
        OUTPUT
        ------        
        return spherical operator basis,Coherence order and LM_state
        """
        
        states = int(2 * S + 1)  # Number of spherical operators in Hilbert-Schidth space, states**2
        EYE = np.eye(states)
        std_basis = np.zeros((states,states,1))
        for i in range(states): 
            std_basis[i] = EYE[:,i].reshape(-1,1)
        L = np.arange(0,2*S+1,1,dtype=np.int16)
        m = -1*np.arange(-S,S+1,1,dtype=np.double)
        Pol_Basis = []
        Coherence_order = []
        LM_state = []

        for i in L:
            M = np.arange(-i,i+1,1,dtype=np.int16)
            for j in M:  
                Sum = 0
                for k in range(states):
                    for l in range(states):
                        cg_coeff = float(CG(S, m[l], i, j, S, m[k]).doit())
                        Sum = Sum + cg_coeff * np.outer(std_basis[k],std_basis[l].T.conj())
                Pol_Basis.append(np.sqrt((2*i + 1)/(2*S+1)) * Sum)
                Coherence_order.append(j) 
                LM_state.append(tuple([i,j]))
        
        print("Coherence Order: ",Coherence_order)
        print("LM state: ",LM_state)
        return Pol_Basis,Coherence_order,LM_state                 

    def ProductOperator(self,OP1,CO1,DIC1,OP2,CO2,DIC2,sort,indexing):
        """
        Product of two spherical basis operators (kronecker porduct)
        
        INPUT
        -----
        OP1 and OP2: Individual operator basis of each particles
        CO1 and CO2: Individual coherence order of each particle
        DIC1 and DIC2: Individual labels of basis operators of each particle
        sort: sort coherence order by 'normal' or 'negative to positive' or 'zero to high'
        indexing: if True, index will be added with the labelling of basis operators
        
        OUTPUT
        ------        
        OP: New Operator basis
        CO: New coherence order
        DIC: New labelling
        """
        
        CO = []
        OP = []
        DIC = []
        index = 0
        for i,j,k in zip(OP1,CO1,DIC1):
            for m,n,o in zip(OP2,CO2,DIC2):
                OP.append(np.kron(i,m))
                CO.append(j+n)
                DIC.append(k+o)
                
        if sort == 'normal':
            pass
            
        if sort == 'negative to positive':        
            # Sorting increasing coherence order
            combine = list(zip(CO,OP,DIC))
            combine_sort = sorted(combine, key=lambda x: x[0])
            Sort_CO,Sort_OP,Sort_DIC = zip(*combine_sort)  
            CO = list(Sort_CO)
            OP = list(Sort_OP)
            DIC = list(Sort_DIC)      
            
        if sort == 'zero to high':        
            # Sorting increasing coherence order
            combine = list(zip(list(map(abs, CO)),CO,OP,DIC))
            combine_sort = sorted(combine, key=lambda x: x[0])
            Sort_CO_dumy,Sort_CO,Sort_OP,Sort_DIC = zip(*combine_sort)  
            CO = list(Sort_CO)
            OP = list(Sort_OP)
            DIC = list(Sort_DIC)      
            
        if indexing:                        
            for p in range(len(DIC)):
                DIC[p] = DIC[p] + "[" + str(index) + "]"      
                index = index + 1  
                
        return OP, CO, DIC                                 
        
    def Proj_OP(self,Ba):
        """
        Projection Operators (Alpha Beta Operator Basis)
        
        INPUT
        -----
        Ba: Zeeman basis
        OUTPUT
        ------        
        return projection operators in Zeeman basis
        """
        B = np.zeros((self.Vdim * self.Vdim, self.Vdim, self.Vdim))
        k = 0
        for i in range(self.Vdim):
            for j in range(self.Vdim):
                B[k] = np.outer(Ba[i],self.Adjoint(Ba[j]))
                k = k + 1

        if ((self.Nspins == 1) and (self.S[0] == 1/2)):
            display(Latex(r'\begin{bmatrix} & \alpha &  \beta \\' \
            r'\alpha & B(0) & B(1)\\' \
            r'\beta & B(2) & B(3) \end{bmatrix}')) 
                            
        if ((self.Nspins == 2) and (self.S[0] == 1/2) and (self.S[1] == 1/2)):
            display(Latex(r'\begin{bmatrix} & \alpha \alpha & \alpha \beta & \beta \alpha & \beta \beta \\' \
            r'\alpha \alpha & B_{0}(P) & B_{1}(SQ) & B_{2}(SQ) & B_{3}(DQ) \\' \
            r'\alpha \beta & B_{4}(SQ) & B_{5}(P) & B_{6}(ZQ) & B_{7}(SQ) \\' \
            r'\beta \alpha & B_{8}(SQ) & B_{9}(ZQ) & B_{10}(P) & B_{11}(SQ) \\' \
            r'\beta \beta & B_{12}(DQ) & B_{13}(SQ) & B_{14}(SQ) & B_{15}(P) \end{bmatrix}')) 
   
        return B  
        
    def SingleSpinOP(self,Sx,Sy,Sz,Sp,Sm,Basis):
        """
        Singel Spin Half Operators
        
        INPUT
        -----
        Sx: Spin operator Sx
        Sy: Spin operator Sy
        Sz: Spin operator Sz
        Sp: Spin operator S+
        Sm: Spin operator S-
        Basis: 'Cartesian spin half' or 'PMZ spin half'
        OUTPUT
        ------        
        return operator basis
        """
        
        B = np.zeros((4, self.Vdim, self.Vdim),dtype=np.clongdouble)       
        
        if Basis == 'Cartesian spin half':
            B[0] = (1/np.sqrt(2)) * eye(self.Vdim)
            B[1] = np.sqrt(2) * Sx[0]
            B[2] = np.sqrt(2) * Sy[0]
            B[3] = np.sqrt(2) * Sz[0]
            #display(Latex(r'Basis: $\frac{1}{\sqrt{2}}E$,$\sqrt{2}I_{x}$,$\sqrt{2}I_{y}$,$\sqrt{2}I_{z}$'))
            dic = ['E','Ix','Iy','Iz']
        
        if Basis == 'PMZ spin half':
            B[0] = (1/np.sqrt(2)) * eye(self.Vdim)
            B[1] = Sp[0]
            B[2] = Sm[0]
            B[3] = np.sqrt(2) * Sz[0]
            #display(Latex(r'Basis: $\frac{1}{\sqrt{2}}E$,$I_{+}$,$I_{-}$,$\sqrt{2}I_{z}$'))
            dic = ['E','I+x','I-','Iz']
            
        return B, dic                   
        
    def TwoSpinOP(self,Sx,Sy,Sz,Sp,Sm,Basis):
        """
        Two Spin Half Operators
        
        INPUT
        -----
        Sx: Spin operator Sx
        Sy: Spin operator Sy
        Sz: Spin operator Sz
        Sp: Spin operator S+
        Sm: Spin operator S-
        Basis: 'Cartesian spin half' or 'PMZ spin half'
        OUTPUT
        ------        
        return operator basis
        """
        
        B = np.zeros((16, self.Vdim, self.Vdim),dtype=np.clongdouble)        
        
        if Basis == 'Cartesian spin half':
            B[0] = (1/2) * eye(self.Vdim)
            B[1] = Sx[0] # In-Phase
            B[2] = Sy[0] # In-Phase
            B[3] = Sz[0] # In-Phase
            B[4] = Sx[1] # In-Phase
            B[5] = Sy[1] # In-Phase
            B[6] = Sz[1] # In-Phase
            
            B[7] = 2 * np.matmul(Sx[0],Sz[1]) # Anti-Phase
            B[8] = 2 * np.matmul(Sy[0],Sz[1]) # Anti-Phase 
            B[9] = 2 * np.matmul(Sz[0],Sx[1]) # Anti-Phase  
            B[10] = 2 * np.matmul(Sz[0],Sy[1]) # Anti-Phase 
            B[11] = 2 * np.matmul(Sz[0],Sz[1]) # Anti-Phase  
            
            B[12] = 2 * np.matmul(Sx[0],Sx[1]) # Multiple Quantum Coherence
            B[13] = 2 * np.matmul(Sx[0],Sy[1]) # Multiple Quantum Coherence
            B[14] = 2 * np.matmul(Sy[0],Sx[1]) # Multiple Quantum Coherence
            B[15] = 2 * np.matmul(Sy[0],Sy[1]) # Multiple Quantum Coherence
            #display(Latex(r'Basis: $\frac{1}{2}E$,$I_{x}$,$I_{y}$,$I_{z}$,$S_{x}$,$S_{y}$,$S_{z}$,' \
            #r'2$I_{x}S_{z}$,2$I_{y}S_{z}$,2$I_{z}S_{x}$,2$I_{z}S_{y}$,2$I_{z}S_{z}$,'\
            #r'2$I_{x}S_{x}$,2$I_{x}S_{y}$,2$I_{y}S_{x}$,2$I_{y}S_{y}$'))                                            
            dic = ['E','Sx','Sy','Sz','Ix','Iy','Iz','Sx Iz','Sy Iz','Sz Ix','Sz Iy','Sz Iz','Sx Ix','Sx Iy','Sy Ix','Sy Iy']
            
        if Basis == 'PMZ spin half':
            B[0] = (1/2) * eye(self.Vdim)                                         
            B[1] = (1/np.sqrt(2)) * Sp[0] # In-Phase
            B[2] = (1/np.sqrt(2)) * Sm[0] # In-Phase
            B[3] = Sz[0] # In-Phase
            B[4] = (1/np.sqrt(2)) * Sp[1] # In-Phase
            B[5] = (1/np.sqrt(2)) * Sm[1] # In-Phase
            B[6] = Sz[1] # In-Phase
            
            B[7] = np.sqrt(2) * np.matmul(Sp[0],Sz[1]) # Anti-Phase
            B[8] = np.sqrt(2) * np.matmul(Sm[0],Sz[1]) # Anti-Phase 
            B[9] = np.sqrt(2) * np.matmul(Sz[0],Sp[1]) # Anti-Phase  
            B[10] = np.sqrt(2) * np.matmul(Sz[0],Sm[1]) # Anti-Phase 
            B[11] = 2 * np.matmul(Sz[0],Sz[1]) # Anti-Phase  
            
            B[12] = np.matmul(Sp[0],Sp[1]) # Multiple Quantum Coherence
            B[13] = np.matmul(Sp[0],Sm[1]) # Multiple Quantum Coherence
            B[14] = np.matmul(Sm[0],Sp[1]) # Multiple Quantum Coherence
            B[15] = np.matmul(Sm[0],Sm[1]) # Multiple Quantum Coherence
            #display(Latex(r'Basis: $\frac{1}{2}E$,$\frac{1}{\sqrt{2}}I_{+}$,$\frac{1}{\sqrt{2}}I_{-}$,$\frac{1}{\sqrt{2}}I_{z}$,$\frac{1}{\sqrt{2}}S_{+}$,$\frac{1}{\sqrt{2}}S_{-}$,$S_{z}$,' \
            #r'$\sqrt{2}I_{+}S_{z}$,$\sqrt{2}I_{-}S_{z}$,$\sqrt{2}I_{z}S_{+}$,$\sqrt{2}I_{z}S_{-}$,2$I_{z}S_{z}$,'\
            #r'$I_{+}S_{+}$,$I_{+}S_{-}$,$I_{-}S_{+}$,$I_{-}S_{-}$')) 
            dic = ['E','S+','S-','Sz','I+','I-','Iz','S+ Iz','S- Iz','Sz I+','Sz I-','Sz Iz','S+ I+','S+ I-','S- I+','S- I-']
            
        return B, dic
                
    def OperatorBasis(self,Basis): 
        """
	    Show Operator Basis for single and two spin half particle(s)
	    
	    This function will be removed in PyOR version 1.0
        INPUT
        -----
        Basis: Single spin ('Cartesian') and Two spin ('Cartesian' or 'PMZ')
        
        OUTPUT
        ------	    
	    print the label of Operator basis
        """        
        if self.Nspins == 1:
            if Basis == 'Cartesian':
                display(Latex(r'Basis: $\frac{1}{\sqrt{2}}E$,$\sqrt{2}I_{x}$,$\sqrt{2}I_{y}$,$\sqrt{2}I_{z}$'))
            if Basis == 'PMZ':
                display(Latex(r'Basis: $\frac{1}{\sqrt{2}}E$,$I_{+}$,$I_{-}$,$\sqrt{2}I_{z}$'))
		        
        if self.Nspins == 2:
            if Basis == 'Cartesian':
                display(Latex(r'Basis: B0 = $\frac{1}{2}E$, B1 = $I_{x}$, B2 = $I_{y}$, B3 = $I_{z}$, B4 = $S_{x}$, B5 = $S_{y}$, B6 = $S_{z}$,' \
                r' B7 = 2$I_{x}S_{z}$, B8 = 2$I_{y}S_{z}$, B9 = 2$I_{z}S_{x}$, B10 = 2$I_{z}S_{y}$, B11 = 2$I_{z}S_{z}$,'\
                r' B12 = 2$I_{x}S_{x}$, B13 = 2$I_{x}S_{y}$, B14 = 2$I_{y}S_{x}$, B15 = 2$I_{y}S_{y}$'))
            if Basis == 'PMZ':
                display(Latex(r'Basis: B0 = $\frac{1}{2}E$, B1 = $\frac{1}{\sqrt{2}}I_{+}$, B2 = $\frac{1}{\sqrt{2}}I_{-}$, B3 = $\frac{1}{\sqrt{2}}I_{z}$, B4 = $\frac{1}{\sqrt{2}}S_{+}$, B5 = $\frac{1}{\sqrt{2}}S_{-}$, B6 = $S_{z}$,' \
                r' B7 = $\sqrt{2}I_{+}S_{z}$, B8 = $\sqrt{2}I_{-}S_{z}$, B9 = $\sqrt{2}I_{z}S_{+}$, B10 = $\sqrt{2}I_{z}S_{-}$, B11 = 2$I_{z}S_{z}$,'\
                r' B12 = $I_{+}S_{+}$, B13 = $I_{+}S_{-}$, B14 = $I_{-}S_{+}$, B15 = $I_{-}S_{-}$'))                 
                    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Matrix Visualization
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
    def User_Label(self,Uxlabel,Uylabel):
        """
        User Define the X and Y label
        """
        
        self.Uxlabel = Uxlabel
        self.Uylabel = Uylabel
    
    def XYlabel_1spin(self,X):
        """
        Label X and Y single spin
        
        INPUT
        -----
        X: spin quantum number
        
        OUTPUT
        ------        
        return x and y label for matrix visualization
        """
        labelx = []
        labely = []
        mag = self.MagQnu(X)
        for i in mag:
            labely.append(r"$\langle$" + str(Fraction(X)) + "," + str(Fraction(i)) + r"$|$")
            labelx.append(r"$|$" + str(Fraction(X)) + "," + str(Fraction(i)) + r"$\rangle$")
        return labelx, labely

    def XYlabel_2spin(self,X,Y):
        """
        Label X and Y Two spins
        
        INPUT
        -----
        X and Y spin quantum number of particle 1 and 2
        OUTPUT
        ------        
        return x and y label for matrix visualization
        """
        labelx = []
        labely = []
        magX = self.MagQnu(X)
        magY = self.MagQnu(Y)
        for i in magX:
            for j in magY:
                labely.append(r"$\langle$" + str(Fraction(X)) + "," + str(Fraction(i)) + r"$|$" + r"$\langle$" + str(Fraction(Y)) + "," + str(Fraction(j)) + r"$|$")
                labelx.append(r"$|$" + str(Fraction(X)) + "," + str(Fraction(i)) + r"$\rangle$" + r"$|$" + str(Fraction(Y)) + "," + str(Fraction(j)) + r"$\rangle$")
        return labelx, labely
    
    def MatrixPlot(self,fig_no,M):
        """
        Matrix Plotting
        
        INPUT
        -----
        fig_no: figure number
        M: Matrix
        OUTPUT
        ------        
        return plot matrix
        
        OTHERS (labelling options)
        ------
        PlotLabel_Hilbert: True for Hilbert space (Default: True)
        PlotLabel_Hilbert: False for Liouvielle, option to plot redkite: Redkite_Label (Default: False) or Redkite_Label_SpinDynamica (Default: False) == True or False
        """
        
        cmap = [cm.RdBu, cm.seismic, cm.bwr, cm.RdGy]
        labelx = []
        labely = []
        if ((self.Nspins == 1) and (self.PlotLabel_Hilbert)):
            #label = [r'$\alpha$', r'$ \beta$']  
            labelx,labely =  self.XYlabel_1spin(self.S[0]) 
            
        if ((self.Nspins == 2) and (self.PlotLabel_Hilbert)):
            #label = [r'$\alpha \alpha$', r'$\alpha \beta$', r'$\beta \alpha$', r'$\beta \beta$']
            labelx,labely =  self.XYlabel_2spin(self.S[0],self.S[1])
            
        if self.Redkite_Label: 
            labelx = [r'$E$',r'$S_{z}$',r'$I_{z}$',r'$S_{z}I_{z}$',r'$S_{+}I_{-}$',r'$S_{-}I_{+}$',r'$S_{+}$',r'$S_{+}I_{z}$',r'$I_{+}$',r'$S_{z}I_{+}$',r'$S_{-}$',r'$S_{-}I_{z}$',r'$I_{-}$',r'$S_{z}I_{-}$',r'$S_{+}I_{+}$',r'$S_{-}I_{-}$']
            labely = [r'$E$',r'$S_{z}$',r'$I_{z}$',r'$S_{z}I_{z}$',r'$S_{+}I_{-}$',r'$S_{-}I_{+}$',r'$S_{+}$',r'$S_{+}I_{z}$',r'$I_{+}$',r'$S_{z}I_{+}$',r'$S_{-}$',r'$S_{-}I_{z}$',r'$I_{-}$',r'$S_{z}I_{-}$',r'$S_{+}I_{+}$',r'$S_{-}I_{-}$']    

        if self.Redkite_Label_SpinDynamica: 
            labelx = [r'$S_{-}I_{-}$',r'$S_{-}I_{z}$',r'$S_{z}I_{-}$',r'$S_{-}$',r'$I_{-}$',r'$S_{-}I_{+}$',r'$S_{+}I_{-}$',r'$S_{z}I_{z}$',r'$I_{z}$',r'$S_{z}$',r'$E$',r'$S_{+}I_{z}$',r'$S_{z}I_{+}$',r'$S_{+}$',r'$I_{+}$',r'$S_{+}I_{+}$']
            labely = [r'$S_{-}I_{-}$',r'$S_{-}I_{z}$',r'$S_{z}I_{-}$',r'$S_{-}$',r'$I_{-}$',r'$S_{-}I_{+}$',r'$S_{+}I_{-}$',r'$S_{z}I_{z}$',r'$I_{z}$',r'$S_{z}$',r'$E$',r'$S_{+}I_{z}$',r'$S_{z}I_{+}$',r'$S_{+}$',r'$I_{+}$',r'$S_{+}I_{+}$']
            
        if self.UserDefined_Label:
            labelx = self.Uxlabel
            labely = self.Uylabel    
                                
        plt.rcParams['figure.figsize'] = (8,8)
        plt.rcParams['font.size'] = 8
        
        fig = plt.figure(fig_no)
        ax = fig.add_subplot(111)
        
        cax = ax.matshow(M, interpolation='nearest',cmap=cmap[1],vmax=abs(M).max(), vmin=-abs(M).max())
        fig.colorbar(cax)
        
        ax.set_xticks(np.arange(len(labelx)))
        ax.set_yticks(np.arange(len(labely)))
        ax.set_xticklabels(labelx)
        ax.set_yticklabels(labely) 
        
        plt.show()

    def MatrixPlot_slider(self,fig_no,t,rho_t):
        """
        Matrix Plotting as function of time
        
        INPUT
        -----
        fig_no: figure number
        t: Time array
        rho_t: array of density matrices for each time 
        OUTPUT
        ------        
        Plot matrix with slider, move slider to see density matrix at different time.
        """
        
        cmap = [cm.RdBu, cm.seismic, cm.bwr, cm.RdGy]
        labelx = []
        labely = []
        if ((self.Nspins == 1) and (self.PlotLabel_Hilbert)):
            #label = [r'$\alpha$', r'$ \beta$']  
            labelx,labely =  self.XYlabel_1spin(self.S[0])  
            
        if ((self.Nspins == 2) and (self.PlotLabel_Hilbert)):
            #label = [r'$\alpha \alpha$', r'$\alpha \beta$', r'$\beta \alpha$', r'$\beta \beta$']
            labelx,labely =  self.XYlabel_2spin(self.S[0],self.S[1])
            
        if self.UserDefined_Label:
            labelx = self.Uxlabel
            labely = self.Uylabel            
                    
        plt.rcParams['figure.figsize'] = (8,5)
        plt.rcParams['font.size'] = 14
        plt.rcParams["figure.autolayout"] = True
        
        fig = plt.figure(fig_no)
        ax = fig.add_subplot(111)
        X = ax.matshow(rho_t[0].real, interpolation='nearest',cmap=cmap[1])
        
        cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.65])
        cbar = fig.colorbar(X, cax = cbaxes)
        
        ax.set_title('T=%2.3f'%t[0])
        ax.set_xticklabels([''] + labelx,fontsize=10)
        ax.set_yticklabels([''] + labely,fontsize=10) 
        
        fig.subplots_adjust(left=0.25, bottom=0.25)
        axfreq = fig.add_axes([0.2, 0.001, 0.65, 0.03])
        index_slider = Slider(ax=axfreq,label='index',valmin=0,valmax=t.shape[-1], valinit=0)
        
        def update(val):
            X = ax.matshow(rho_t[int(index_slider.val)].real, interpolation='nearest',cmap=cmap[1])
            ax.set_title('T=%2.3f'%t[int(index_slider.val)])
            cbar.update_normal(X)
            fig.canvas.draw_idle()
            
        index_slider.on_changed(update)
        
        plt.show()
        
    def MatrixPlot3D(self,fig_no,rho):
        """
        Matrix Plot 3D
        
        INPUT
        -----
        fig_no: Figure number
        rho: density matrix
        OUTPUT
        ------        
        Plot 3D matrix
        """
             
        labelx = []
        labely = []
        if ((self.Nspins == 1) and (self.PlotLabel_Hilbert)):
            #label = [r'$\alpha$', r'$ \beta$'] 
            labelx,labely =  self.XYlabel_1spin(self.S[0])   
            
        if ((self.Nspins == 2) and (self.PlotLabel_Hilbert)):
            #label = [r'$\alpha \alpha$', r'$\alpha \beta$', r'$\beta \alpha$', r'$\beta \beta$']
            labelx,labely =  self.XYlabel_2spin(self.S[0],self.S[1])

        if self.UserDefined_Label:
            labelx = self.Uxlabel
            labely = self.Uylabel
                
        rc('font', weight='bold')
        fig = plt.figure(fig_no,constrained_layout=True, figsize=(13, 8))
        ax1 = plt.axes(projection = "3d")
        
        numofCol = rho.shape[-1]
        numofRow = rho.shape[0]
        
        xpos = np.arange(0, numofCol, 1)
        ypos = np.arange(0, numofRow, 1)
        xpos, ypos = np.meshgrid(xpos+0.25, ypos+0.25)
        
        xpos = xpos.flatten()
        ypos = ypos.flatten()
        zpos = np.zeros(numofRow*numofRow)
        
        dx = 0.5 * np.ones_like(zpos)
        dy = dx.copy()
        dz = rho.flatten()
        
        positive = dz.copy()
        negative = dz.copy()
        positive[positive<0] = 0
        negative[negative>=0] = 0
                
        ax1.bar3d(xpos,ypos,zpos, dx, dy, dz, color='b', alpha=0.5)

        #ax1.bar3d(xpos,ypos,zpos, dx, dy, positive, color='b', alpha=0.5)
        #ax1.bar3d(xpos,ypos,zpos, dx, dy, -negative, color='r', alpha=0.5)
        
        ticksx = np.arange(0.5, rho.shape[-1], 1)
        ticksy = np.arange(0.6, rho.shape[-1], 1)
        #ax1.set_xticklabels(label)
        #ax1.set_yticklabels(label)
        plt.xticks(ticksx,labelx,fontsize=5)
        plt.yticks(ticksy,labely,fontsize=5)
        ax1.set_zlim(np.min(rho),np.max(rho))
        #ax1.set_zlim(0,np.max(rho))
        ax1.grid(False)
            
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Equlibrium Density Matrix
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def EqulibriumDensityMatrix(self,H,T,HT_approx):
        """
        Equlibrium Density Matrix
        INPUT
        -----
        H         : Hamiltonian always in energy unit and not in frequency unit.
        T         : Temperature
        HT_approx : if True, high temperature approximation will be considered
        
        OUTPUT     
        ------
        Equlibrium Density Matrix      
        """
        
        rho_T = np.zeros((self.Vdim,self.Vdim))
        if HT_approx: 
            E = np.eye(self.Vdim)   
            rho_T = (E - H/(self.kb*T))/np.trace(E - H/(self.kb*T))
        else:
            rho_T = expm(-H/(self.kb*T))/np.trace(expm(-H/(self.kb*T)))
            
        print("Trace of density metrix = ", (np.trace(rho_T)).real)    

        return rho_T    

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Commutators
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
    def Commutator(self,A,B):
        """
        Commutator
        INPUT
        -----
        A : matrix A
        B : matrix B

        OUTPUT     
        ------
        Commutator [A,B]      
        """     
        return np.matmul(A,B) - np.matmul(B,A)
    
    def DoubleCommutator(self,A,B,rho):
        """
        Double Commutator 
        INPUT
        -----
        A   : matrix A
        B   : matrix B
        rho : matrix rho

        OUTPUT     
        ------
        Double Commutator [A,[B,rho]]      
        """     
        C = self.Commutator(B,rho)
        return self.Commutator(A,C)
    
    def AntiCommutator(self,A,B):
        """
        Anti Commutator
        INPUT
        -----
        A   : matrix A
        B   : matrix B

        OUTPUT     
        ------
        Anti Commutator {A,B}      
        """     
        return np.matmul(A,B) + np.matmul(B,A)      
        
    def CommutationSuperoperator(self,X):
        """
        Commutation Superoperator [H,rho] = left(H) [rho] - right(H) [rho] = H rho - rho H
        
        INPUT
        -----
        X : matrix X

        OUTPUT     
        ------
        Commutation Superoperator   
        """     
        Id = np.identity((X.shape[-1]))
        return np.kron(X,Id) - np.kron(Id,X.T)

    def AntiCommutationSuperoperator(self,X):
        """
        Anti Commutation Superoperator: {H,rho} = left(H) [rho] + right(H) [rho] = H rho + rho H
        
        INPUT
        -----
        X : matrix X

        OUTPUT     
        ------
        anti Commutation Superoperator   
        """     
        Id = np.identity((X.shape[-1]))
        return np.kron(X,Id) + np.kron(Id,X.T)

    def Left_Superoperator(self,X):
        """
        Left Superoperator: left(H) [rho] = H rho
        
        INPUT
        -----
        X : matrix X

        OUTPUT     
        ------
        left Superoperator   
        """     
        Id = np.identity((X.shape[-1]))
        return np.kron(X,Id)
        
    def Right_Superoperator(self,X):
        """
        Right Superoperator: right(H) [rho] = rho H
        
        INPUT
        -----
        X : matrix X

        OUTPUT     
        ------
        right Superoperator   
        """     
        Id = np.identity((X.shape[-1]))
        return np.kron(Id,X.T)        
    
    def DoubleCommutationSuperoperator(self,X,Y):
        """
        Double Commutation Superoperator
        INPUT
        -----
        X : matrix X
        Y : matrix Y

        OUTPUT     
        ------
        Double Commutation Superoperator  
        """     
        Idx = np.identity((X.shape[-1]))
        Idy = np.identity((Y.shape[-1]))
        return np.matmul(np.kron(X,Idx) - np.kron(Idx,X.T), np.kron(Y,Idy) - np.kron(Idy,Y.T) ) 
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Rotation in Hilbert space and Liouville Space
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    
    def Pulse_Phase(self,Sx,Sy,phase):
        """
        Pulse with defined phase; cos(phase) Sx + Sin(phase) Sy
        
        INPUT
        -----
        Sx: Spin operator Sx
        Sy: Spin operator Sy
        Phase: Phase in deg
        
        OUTPUT
        ------        
        return spin operator about which to rotate spin
        """
        phase = np.pi * phase / 180.0
        return np.cos(phase) * np.sum(Sx,axis=0) + np.sin(phase) * np.sum(Sy,axis=0)
    
    def Receiver_Phase(self,Sx,Sy,phase): 
        """
        Detection operator with phase; (Sx + 1j Sy) * exp(1j phase)
        
        INPUT
        -----
        Sx: Spin operator Sx
        Sy: Spin operator Sy
        Phase: Phase in deg
                
        OUTPUT
        ------        
        return detection operator rotated by reciever phase
        """
        phase = np.pi * phase / 180.0
        return (np.sum(Sx,axis=0) + 1j * np.sum(Sy,axis=0)) * np.exp(1j * phase)        
    
    def Rotation_CyclicPermutation(self, A, B, theta):
    
        """
        Rotation of an operator, when the operator and spin operator follw the relation
        [A,B] = j C (Cylic Commutation Relation)
        
        INPUT
        -----
        A      : Operator about which rotation happens
        B      : Operator to rotate
        theta  : angle in radian
        
        OUTPUT
        ------
        EXP(-j A * theta) @ B @ EXP(j A * theta) = B cos(theta) - j [A, B] sin(theta) = B cos(theta) + C sin(theta)
        """
        
        if A == B:
            Bp = B
        else:
            Bp = B * np.cos(np.pi*theta/180.0) - 1j * self.Commutator(A,B) * np.sin(np.pi*theta/180.0)
            
        return Bp
            
    def Rotate_H(self,rho,theta_rad,operator):
        """
        Rotation in Hilbert Space
        INPUT
        -----
        rho       : intial density matrix or operator (eg: hamiltonian)
        theta_rad : Angle to be rotated in degree
        operator  : Spin Operator for rotation

        OUTPUT     
        ------
        rho       : Rotated density matrix or operator (eg: hamiltonian)       
        """     
        theta_rad = np.pi * theta_rad / 180.0
        U = expm(-1j * theta_rad * operator)
        return np.matmul(U,np.matmul(rho,U.T.conj()))  

    def Rotate_L(self,Lrho,theta_rad,operator):
        """
        Rotation in Liouville Space
        INPUT
        -----
        Lrho      : intial state
        theta_rad : Angle to be rotated in degree
        operator  : Spin Superoperator for rotation

        OUTPUT     
        ------
        Lrho       : final state      
        """     
        theta_rad = np.pi * theta_rad / 180.0
        return expm(-1j * theta_rad * self.CommutationSuperoperator(operator)) @ Lrho  
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Liouville Vectors
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def Vector_L(self,X):
        """
        Liouville Vector: Vectorize the operator
        INPUT
        -----
        X : Operator to be vectorized. eg: density matrix

        OUTPUT     
        ------
        Vectorized operator      
        """     
        dim = self.Vdim
        return np.reshape(X,(dim**2,-1))
    
    def Detection_L(self,X):
        """
        Liouville Vector for detection: Vectorize the operator
        INPUT
        -----
        X : Operator to be vectorized. eg: Sz

        OUTPUT     
        ------
        Vectorized operator for detection     
        """     
        X = self.Vector_L(X)
        return X.conj().T
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Probability Desnity Function
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

    def PDFgaussian(self, x, std, mean):  
        """
        Probabilty Distribution Function Gaussian
        
        INPUT
        -----
        x: array of variable for Gaussian Probabilty Distribution Function
        std: standard deviation
        mean: mean
        
        OUTPUT
        ------        
        return normalized Gaussian Probabilty Distribution Function
        """
        gaussian =  (1/np.sqrt(2*np.pi*std**2)) * np.exp(-1*(x-mean)**2/(2*std**2))
        return gaussian/np.sum(gaussian)
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Plotting and Fourier transform
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
    
    def Plotting(self,fig_no,x,y,xlab,ylab,col):
        """
        Plotting the signal
        INPUT
        -----
        fig_no    : figure number
        x         : x array (Horizontal axis)
        y         : y array (Vertical axis)
        xlab      : x label
        ylab      : y label
        col       : colour of the plot

        OUTPUT     
        ------
        plot      
        """     
        rc('font', weight='bold')
        fig = plt.figure(fig_no,constrained_layout=True, figsize=(10, 5))
        spec = fig.add_gridspec(1, 1)

        ax1 = fig.add_subplot(spec[0, 0])

        ax1.plot(x,y,linewidth=3.0,color=col)

        ax1.set_xlabel(xlab, fontsize=25, color='black',fontweight='bold')
        ax1.set_ylabel(ylab, fontsize=25, color='black',fontweight='bold')
        ax1.legend(fontsize=25,frameon=False)
        ax1.tick_params(axis='both',labelsize=14)
        ax1.grid(True, linestyle='-.')
        #ax1.set_xlim(xli,xlf)
        plt.show()

    def PlottingTwin(self,fig_no,x,y1,y2,xlab,ylab1,ylab2,col1,col2):
        """
        Plotting Twin Axis (y)
        
        INPUT
        -----
        fig_no: figure number
        x: x array
        y1: y1 array
        y2: y2 array
        xlabel: x label
        ylab1: y1 label
        ylab2: y2 label
        col1: color for y1
        col2: color for y2
        
        OUTPUT
        ------        
        plot
        """
        
        rc('font', weight='bold')
        fig = plt.figure(fig_no,constrained_layout=True, figsize=(10, 5))
        spec = fig.add_gridspec(1, 1)

        ax1 = fig.add_subplot(spec[0, 0])
	    
        ax1.plot(x,y1,linewidth=3.0,color=col1)

        ax1.set_xlabel(xlab, fontsize=25, color='black',fontweight='bold')
        ax1.set_ylabel(ylab1, fontsize=25, color='black',fontweight='bold')
        ax1.legend(fontsize=25,frameon=False)
        ax1.tick_params(axis='both',labelsize=14)
        ax1.grid(True, linestyle='-.')
        #ax1.set_xlim(xli,xlf)

        ax10 = ax1.twinx()
        ax10.plot(x,y2,linewidth=3.0,color=col2)

        ax10.set_xlabel(xlab, fontsize=25, color='black',fontweight='bold')
        ax10.set_ylabel(ylab2, fontsize=25, color='black',fontweight='bold')
        ax10.legend(fontsize=25,frameon=False)
        ax10.tick_params(axis='both',labelsize=14)
        ax10.grid(True, linestyle='-.')
	    #plt.savefig('figure.pdf',bbox_inches='tight')
        plt.show()

    def PlottingMulti(self,fig_no,x,y,xlab,ylab,col):
        """
        Plotting the signal
        INPUT
        -----
        figure    : figure number
        x         : [x1 array, x1 array, ...] (Horizontal axis)
        y         : [y1 array, y1 array, ... ] (Vertical axis)
        xlab      : x label
        ylab      : y label
        col       : [colour 1, colour 2, ... ] of the plot

        OUTPUT     
        ------
        plot      
        """     
        rc('font', weight='bold')
        fig = plt.figure(fig_no,constrained_layout=True, figsize=(10, 5))
        spec = fig.add_gridspec(1, 1)

        ax1 = fig.add_subplot(spec[0, 0])
        
        for i in range(len(x)):
            ax1.plot(x[i],y[i],linewidth=3.0,color=col[i])

        ax1.set_xlabel(xlab, fontsize=25, color='black',fontweight='bold')
        ax1.set_ylabel(ylab, fontsize=25, color='black',fontweight='bold')
        ax1.legend(fontsize=25,frameon=False)
        ax1.tick_params(axis='both',labelsize=14)
        ax1.grid(True, linestyle='-.')
        #ax1.set_xlim(xli,xlf)
        plt.show()

    def Plotting3DWire(self,fig_no,x,y,z,xlab,ylab,title,upL,loL):
        """
        Plot 3D Surface
        
        INPUT
        -----
        fig_no: Figure number
        x: x data
        y: y data
        z: z data, function of x,y
        xlab: x label
        ylab: y label
        title: Title of the plot
        upL: Upper limit of X and Y axis
        loL: Lower limit of X and Y axis        
        OUTPUT
        ------        
        return the wire plot
        """
        rc('font', weight='bold')
        #ax = plt.figure(fig_no,figsize=(10, 5)).add_subplot(projection='3d')
        fig = plt.figure(fig_no,constrained_layout=True, figsize=(10, 5))
        spec = fig.add_gridspec(1, 1)
        ax1 = fig.add_subplot(spec[0, 0],projection='3d')
        
        x1 = x.copy()
        y1 = y.copy()
        x1[x1>upL] = np.nan
        y1[y1>upL] = np.nan
        x1[x1<loL] = np.nan
        y1[y1<loL] = np.nan        
        
        X,Y = np.meshgrid(x1,y1)
        wire = ax1.plot_wireframe(X, Y, z, lw=0.5, rstride=8, cstride=8) #, alpha=0.3
        # rstride=0 for row stride set to 0
        # ctride=0 for column stride set to 0
        ax1.set_xlabel(xlab)
        ax1.set_ylabel(ylab)
        ax1.set_title(title)
        ax1.set_xlim3d(loL,upL)
        ax1.set_ylim3d(loL,upL)
        plt.show()
                
    def PlottingContour(self, fig_no,x,y,z,xlab,ylab,title):
        """
        Plot Contour
        
        INPUT
        -----
        fig_no: Figure number
        x: x data
        y: y data
        z: z data, function of x,y
        xlab: x label
        ylab: y label
        titile: Title of the plot
        
        OUTPUT
        ------        
        return the contour plot
        """
        cmap = [cm.RdBu, cm.seismic, cm.bwr, cm.RdGy]
        rc('font', weight='bold')
        fig = plt.figure(fig_no,constrained_layout=True, figsize=(10, 5))
        spec = fig.add_gridspec(1, 1)
        
        ax1 = fig.add_subplot(spec[0, 0])
        plotC = ax1.contour(z, 10, extent=[x.min(), x.max(), y.min(), y.max()], cmap=cmap[1], vmax=abs(z).max(), vmin=-abs(z).max()) 
        ax1.set_xlabel(xlab)
        ax1.set_ylabel(ylab)
        ax1.set_title(title)
        cbar = fig.colorbar(plotC)
        plt.show()
        
    def PlottingSphere(self, fig_no,Mx,My,Mz,rho_eq,Sz,plot_vector,scale_datapoints):
        """
        Plotting magnetization evolution in a unit sphere
        
        INPUT
        -----
        fig_no: Figure number
        Mx: Array of Mx
        My: Array of My
        Mz: Array of Mz
        rho_eq: equlibrium density matrix
        Sz: Spin operator Sz
        plot_vector: If True, vector will be plotted
        scale_datapoints: scale points in the Mx, My and Mz; Mx[::scale_datapoints]
        
        OUTPUT
        ------        
        return sphere plot
        """        
        
        sphera_radius = self.OP_InnerProduct(Sz,rho_eq)
        
        # Create a sphere
        phi, theta = np.mgrid[0.0:2.0 * np.pi:100j, 0.0:np.pi:50j]
        x = sphera_radius * np.sin(theta) * np.cos(phi)
        y = sphera_radius * np.sin(theta) * np.sin(phi)
        z = sphera_radius * np.cos(theta)
                
        fig = plt.figure(fig_no,figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, color='c', alpha=0.3, rstride=5, cstride=5, linewidth=0.5, edgecolor='k')
            
        if plot_vector:
            for mx,my,mz in zip(Mx,My,Mz):  
                ax.quiver(0, 0, 0, mx, my, mz, color='r', arrow_length_ratio=0.1) 
                
        ax.plot(Mx[::scale_datapoints], My[::scale_datapoints], Mz[::scale_datapoints], color='b', linewidth=2)  
        ax.view_init(10, 20)
        ax.set_xlabel('Mx')
        ax.set_ylabel('My')
        ax.set_zlabel('Mz')     
        plt.show()  
        
    def PlottingMultimodeAnalyzer(self,t,freq,sig,spec):
        """
        Multimode Analyzer
        
        INPUT
        -----
        t: time
        freq: frequency
        sig: signal or FID
        spec: spectrum
        
        OUTPUT
        ------
        plot 4 figures 
        Figure 1,1 Signal
        Figure 1,2 Spectrum
        Figure 2,1 Signal
        Figure 2,2 Spectrum
        """            
        rc('font', weight='bold')
        fig, ax = plt.subplots(2,2,figsize=(11, 6))


        line1, = ax[0,0].plot(t,sig,"-", color='green')
        ax[0,0].set_xlabel("time [s]")
        ax[0,0].set_ylabel("signal" )
        ax[0,0].grid()

        vline1 = ax[0,1].axvline(color='k', lw=0.8, ls='--')
        vline2 = ax[0,1].axvline(color='k', lw=0.8, ls='--')
        text1 = ax[0,1].text(0.0, 0.0, '', transform=ax[0,1].transAxes)
        line2, = ax[0,1].plot(freq,spec,"-", color='green')
        ax[0,1].set_xlabel("Frequency [Hz]")
        ax[0,1].set_ylabel("spectrum" )
        #ax[0,1].set_xlim(-40,40)
        ax[0,1].grid()

        line3, = ax[1,0].plot(freq,spec,"-", color='green')
        ax[1,0].set_xlabel("Frequency [Hz]")
        ax[1,0].set_ylabel("spectrum" )
        #ax[1,0].set_xlim(-40,40)
        ax[1,0].grid()

        vline3 = ax[1,1].axvline(color='k', lw=0.8, ls='--')
        vline4 = ax[1,1].axvline(color='k', lw=0.8, ls='--')
        text2 = ax[1,1].text(0.0, 0.0, '', transform=ax[1,1].transAxes)
        line4, = ax[1,1].plot(t,sig,"-", color='green')
        ax[1,1].set_xlabel("time [s]")
        ax[1,1].set_ylabel("signal" )
        ax[1,1].grid()
        #plt.savefig(folder + '/pic3.pdf',bbox_inches='tight')

        fourier = Fanalyzer(sig.real,sig.imag,ax,fig,line1,line2,line3,line4,vline1,vline2,vline3,vline4,text1,text2)
        fig.canvas.mpl_connect("button_press_event",fourier.button_press)
        fig.canvas.mpl_connect("button_release_event",fourier.button_release)
        
        return fig,fourier
                
    def WindowFunction(self,t,signal,LB):
        """
        Induce signal decay
        INPUT
        -----
        t      : time array
        signal : signal array
        LB     : decay rate

        OUTPUT     
        ------
        decaying signal      
        """     
        window = np.exp(-LB*t)
        return signal*window
    
    def FourierTransform(self,signal,fs,zeropoints):
        """
        Fourier Transform
        INPUT
        -----
        signal      : signal array
        fs          : sampling rate (half of the bandwidth)
        zeropoints  : zero filling (zeropoints * Npoints)

        OUTPUT     
        ------
        Fourier transform      
        """     
        signal[0] = signal[0]/2
        spectrum = np.fft.fft(signal,zeropoints*signal.shape[-1])
        spectrum = np.fft.fftshift(spectrum)
        freq = np.linspace(-fs/2,fs/2,spectrum.shape[-1])
        return freq, spectrum  
        
    def PhaseAdjust_PH0(self,spectrum,PH0):
        """
        Phase adjust PH0
        
        INPUT
        -----
        spectrum: spectrum to phase
        PH0: Phase
        
        OUTPUT
        ------        
        return phased spectrum
        """
        
        return spectrum * np.exp(1j * 2 * np.pi * PH0 / 180.0)

    def FourierTransform2D(self,signal,fs1,fs2,zeropoints):
        """
        Fourier Transform 2D
        
        INPUT
        -----
        signa: signal array
        fs1: sampling rate (Indirect Dimension)
        fs2: sampling rate (Direct Dimension)
        zeropoints: zero filling (zeropoints * Npoints)
        
        OUTPUT
        ------        
        
        """     
        signal[:,0] = signal[:,0]/2
        spectrum = np.fft.fft2(signal,(zeropoints*signal[:,0].shape[-1],zeropoints*signal[0,:].shape[-1]),(1,0))
        spectrum = np.fft.fftshift(spectrum)
        freq1 = np.linspace(-fs1/2,fs1/2,spectrum.shape[-1])
        freq2 = np.linspace(-fs2/2,fs2/2,spectrum.shape[0])
        return freq1, freq2, spectrum

    def FourierTransform2D_F1(self,signal,fs,zeropoints):
        """
        Fourier Transform 1D - F1 (Indirect Dimension)
        
        INPUT
        -----
        signal      : signal array
        fs          : sampling rate (half of the bandwidth)
        zeropoints  : zero filling (zeropoints * Npoints)

        OUTPUT     
        ------
        return frequency and Fourier transform      
        """
        spectrum = np.zeros((signal.shape[0],signal.shape[-1]),dtype=np.cdouble)
        for i in range(signal.shape[-1]):
            spec = np.fft.fft(signal[:,i])
            spectrum[:,i] = np.fft.fftshift(spec)
        freq = np.linspace(-fs/2,fs/2,spectrum.shape[0])
        return freq, spectrum

    def FourierTransform2D_F2(self,signal,fs,zeropoints):
        """
        Fourier Transform 1D (Direct Dimension)
        INPUT
        -----
        signal      : signal array
        fs          : sampling rate (half of the bandwidth)
        zeropoints  : zero filling (zeropoints * Npoints)

        OUTPUT     
        ------
        return frequency and Fourier transform      
        """     
        signal[0] = signal[0]/2
        spectrum = np.fft.fft(signal,zeropoints*signal.shape[-1],1)
        spectrum = np.fft.fftshift(spectrum)
        freq = np.linspace(-fs/2,fs/2,spectrum.shape[-1])
        return freq, spectrum
                        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Time evolution of Density Matrix in Hilbert Space
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                 
    def ODE_Method(self,method):
        """
        ODE Methods
        method Options: 'RK45' Explicit Runge-Kutta method of order 5(4), Non Stiff Problem, Real and Complex domain (Default)
                        'RK23' Explicit Runge-Kutta method of order 3(2), Non Stiff Problem, Real and Complex domain
                        'DOP853' Explicit Runge-Kutta method of order 8, Non Stiff Problem, Real and Complex domain
                        'Radau' Implicit Runge-Kutta method of the Radau IIA family of order 5, Stiff Problem
                        'BDF' Implicit multi-step variable-order (1 to 5) method based on a backward differentiation formula for the derivative approximation, Stiff Problem, Real and Complex domain
                        'LSODA' Adams/BDF method with automatic stiffness detection and switching, Stiff Problem
        """
        self.ode_method = method
                
    def Evolution_H(self,rhoeq,rho,Sx,Sy,Sz,Sp,Sm,Hamiltonian,dt,Npoints,method,Rprocess):
        """
        Evolution of density matrix
        INPUT
        -----
        rho         : intial state
        Hamiltonian : Hamiltonian of evolution
        detection   : detection operator
        dt          : time step
        Npoints     : number of time points
        method      : "unitary propagator"  Propagate the hamiltonian by unitary matrix (exp(-j H dt))
                    : "solve ivp" solve the Liouville with differential equation solver (radiation damping and relaxation included)
        Rprocess    :  "No Relaxation" 
                       or "Phenomenological"
                       or "Auto-correlated Random Field Fluctuation" 
                       or "Auto-correlated Dipolar Heteronuclear"
                       or "Auto-correlated Dipolar Homonuclear"
        
        OUTPUT     
        ------
        t       : time
        rho     : Array of density matrix      
        """ 
              
        R1 = self.R1
        R2 = self.R2
        
        if method == "Unitary Propagator":    
            rho_t = np.zeros((Npoints,self.Vdim,self.Vdim),dtype=complex)
            t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
            U = expm(-1j * Hamiltonian * dt)
            for i in range(Npoints):
                rho = np.matmul(U,np.matmul(rho,U.T.conj()))
                rho_t[i] = rho 

        if method == "Unitary Propagator Relaxation_Phen": # Under testing; Is it possible??
            """
            Combine Phenominological relaxation and Unitary Propagator
            """  
            rho_t = np.zeros((Npoints,self.Vdim,self.Vdim),dtype=complex)
            t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
            U = expm(-1j * Hamiltonian * dt)
            for i in range(Npoints):
                rho = rhoeq + np.multiply(np.exp(-1 * self.Relaxation_Phenomenological(R1[0],R2[0]) * dt),rho-rhoeq)
                rho = np.matmul(U,np.matmul(rho,U.T.conj()))                
                rho_t[i] = rho 
                
        if method == "ODE Solver":
            """
            Relaxation possible in Hilbert space by using solver for ODE. 
            """
            rho_t = np.zeros((Npoints,self.Vdim,self.Vdim),dtype=complex)                       
            t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
            rhoi = rho.reshape(-1) + 0 * 1j
            def rhoDOT(t,rho,Hamiltonian,Rprocess,R1,R2,Sx,Sy,Sz,Sp,Sm):
                rho = np.reshape(rho,(self.Vdim,self.Vdim))
                rhodot = np.zeros((rhoi.shape[-1]))
                Rso = self.Relaxation_H(rho-rhoeq,Rprocess,R1,R2,Sx,Sy,Sz,Sp,Sm)
                H = Hamiltonian     
                rhodot = (-1j * self.Commutator(H,rho) - Rso).reshape(-1)        
                return rhodot  
            rhoSol = solve_ivp(rhoDOT,[0,dt*Npoints],rhoi,method=self.ode_method,t_eval=t,args=(Hamiltonian,Rprocess,R1,R2,Sx,Sy,Sz,Sp,Sm),atol = 1e-10, rtol = 1e-10)
            t, rho2d = rhoSol.t, rhoSol.y
            for i in range(Npoints):          
                rho = np.reshape(rho2d[:,i],(self.Vdim,self.Vdim))
                rho_t[i] = rho	            
                                            
        return t, rho_t
        
    def Expectation_H(self,rho_t,detection,dt,Npoints):
        """
        Expectation Value
        
        INPUT
        -----
        rho_t: array of 2d matrix, the density matrix
        detection: observable
        dt: dwell time
        Npoints: Acquisition points 
        
        
        OUTPUT
        ------        
        t: array, Time
        signal: array, Expectation values
        """
        
        signal = np.zeros(Npoints,dtype=complex)
        t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
        for i in range(Npoints):
            signal[i] = np.trace(np.matmul(detection,rho_t[i]))
        return t, signal    

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Time evolution of Density Matrix in Liouville Space
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
    
    def Evolution_L(self,Lrhoeq,Lrho,Sx,Sy,LHamiltonian,dt,Npoints,method):
        """
        Evolution of density vector
        INPUT
        -----
        Lrho         : intial state vector
        Lrhoeq       : equlibrium state vector
        LHamiltonian : Hamiltonian of evolution
        Ldetection   : detection operator
        dt          : time step
        Npoints     : number of time points
        method      : "unitary propagator"  Propagate the hamiltonian by unitary matrix (exp(-j H dt))
                      "Relaxation"          Propagate the hamiltonian by unitary matrix with relaxation included
                    : "solve ivp" solve the Liouville with differential equation solver (relaxation included)

        OUTPUT     
        ------
        t       : time
        Lrho     : array of final density state vector     
        """    

        if method == "Unitary Propagator":    
            Lrho_t = np.zeros((Npoints,self.Ldim,1),dtype=complex)
            t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
            U = expm(-1j * LHamiltonian * dt)
            for i in range(Npoints):
                Lrho = np.matmul(U,Lrho)  
                Lrho_t[i] = Lrho  
        
        if method == "Relaxation":    
            Lrho_t = np.zeros((Npoints,self.Ldim,1),dtype=complex)
            t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
            U = expm(-1j * LHamiltonian * dt)
            for i in range(Npoints):
                Lrho = Lrhoeq + np.matmul(U,Lrho - Lrhoeq) 
                Lrho_t[i] = Lrho        
        
        if method == "ODE Solver":
            Lrho_t = np.zeros((Npoints,self.Ldim),dtype=complex) 
            t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
            Lrho = np.reshape(Lrho,Lrho.shape[0]) + 0 * 1j            
            Lrhoeq = np.reshape(Lrhoeq,Lrhoeq.shape[0])
            
            def rhoDOT(t,Lrho,LHamiltonian,Lrhoeq,Sx,Sy):
                LH = LHamiltonian
                rhodot = np.zeros((self.Ldim))
                rhodot = -1j * np.matmul(LH,Lrho-Lrhoeq) # Under testing: is this the right equation, how to add relaxation to equlibrium?
                rhodot = np.reshape(rhodot,rhodot.shape[0])
                return rhodot
            rhoSol = solve_ivp(rhoDOT,[0,dt*Npoints],Lrho,method=self.ode_method,t_eval=t,args=(LHamiltonian,Lrhoeq,Sx,Sy),atol = 1e-10, rtol = 1e-10)   
            t, rho = rhoSol.t, rhoSol.y
            for i in range(Npoints):
                Lrho_t[i] = rho[:,i]
                                     
        return t, Lrho_t       
    
    def Convert_LrhoTO2Drho(self,Lrho):
        """
        Convert a Vector into a 2d Matrix
        
        INPUT
        -----
        Lrho: density matrix, coloumn vector
        OUTPUT
        ------        
        return density matrix, 2d array
        """
        
        return np.reshape(Lrho,(self.Vdim,self.Vdim))
            
    def Expectation_L(self,Lrho_t,Ldetection,dt,Npoints):
        """
        Expectation Value
        
        INPUT
        -----
        Lrho_t: array of coloumn Vectors, the density matrix
        Ldetection: observable
        dt: dwell time
        Npoints: Acquisition points 
        
        
        OUTPUT
        ------        
        t: array, Time
        signal: array, Expectation values
        """
        
        signal = np.zeros(Npoints,dtype=complex)
        t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
        for i in range(Npoints):
            signal[i] = np.matmul(Ldetection,Lrho_t[i])
        return t, signal
               
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Relaxation in Hilbert space and Liouville Space
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
    def Relaxation_Constants(self,R1,R2):
        """
        Relaxation Constants
        R1: Longitudinal Relaxation
        R2: Transverse Relaation
        """
        self.R1 = R1
        self.R2 = R2
        
    def Relaxation_Parameters(self, LarmorF, OmegaRF, tau, bIS):
        """
        Relaxation Parameters for dipolar relaxation case
        
        INPUTS
        ------
        Larmor: Larmor Frequency
        OmegaRF: Rotating frame frequency
        tau: Correlation time
        bIS: Dipolar Coupling constants
        """
        
        self.Woff = LarmorF - np.asarray(OmegaRF)
        self.tau = tau    
        self.bIS = 2*np.pi*bIS
        self.LarmorF = LarmorF

    def Relaxation_Phenomenological_Input(self,R):
        """
        User defined Relaxation Matrix
        """
        self.R_input = R
        
    def Relaxation_Spins(self,spin1,spin2):
        """
        Spin Index: spin1 and spin2
        By default, self.spin1 = 0 and self.spin2 = 1
        User can redefine the value of spin1 and spin2 by calling this function.
        """
        self.spin1 = spin1
        self.spin2 = spin2 
        
    def CSA_Constant(self,Delta,Theta_R):
        """
        CSA Constant
        
        INPUT
        -----
        Delta: Shift Anisotropy 
        Theta_R: Angle between CSA tensor and the dipolar vector  
        """
        self.Delta_CSA = Delta
        self.Theta_R = Theta_R  

    def SpectralDensity(self,W,tau):
        """
        Spectral Density Function

        INPUT
        -----
        W: Eigen frequency
        tau: correlation time
        
        OUTPUT
        ------
        return spectral density
        """
        
        return tau/(1 + W**2 * tau**2) 

    def Temperature(self,T):
        """
        Temperature of the system
        used by function: SpectralDensity_Lb(W,tau)
        
        INPUT
        -----
        T: Temperature
        """
        self.T = T

    def SpectralDensity_Lb(self,W,tau):
        """
        Spectral Density Function with thermal correction.
        For Lindblad Relaxation
        
        INPUT
        -----
        W: Eigen frequency
        tau: correlation time
        
        OUTPUT
        ------
        return spectral density
        """
        
        #return 2 * tau * np.exp(-0.5 * W * (self.hbar/(self.T * self.kb)))  # W * tau << 1
        return (2 * tau/(1 + W**2 * tau**2)) * np.exp(-0.5 * W * (self.hbar/(self.T * self.kb))) 

    def Spherical_Tensor(self,spin,Rank,m,Sx,Sy,Sz,Sp,Sm):
        """
        Spherical rank tensors
        
        INPUT
        -----
        spin: List of spin index, example [0, 1] ( 0 corresponds to index of spin 1 and 0 corresponds to index of spin 2 ) or [1,2]
        Rank: rank of spherical tensor
        m: it takes values from -Rank,...,Rank
        Sx: Spin Operator Sx
        Sy: Spin Operator Sy
        Sz: Spin Operator Sz
        Sp: Spin Operator Sp
        Sm: Spin Operator Sm
        
        OUTPUT
        ------
        Return Value of spherical tensor for corresponding Rank and m value.        
        """
        
        if Rank == 2:
            if m == 0:
                return (4 * np.matmul(Sz[spin[0]],Sz[spin[1]]) - np.matmul(Sp[spin[0]],Sm[spin[1]]) - np.matmul(Sm[spin[0]],Sp[spin[1]]))/(2 * np.sqrt(6))  # T(2,0)
            if m == 1:
                return -0.5 * (np.matmul(Sz[spin[0]],Sp[spin[1]]) + np.matmul(Sp[spin[0]],Sz[spin[1]])) # T(2,+1)
            if m == -1:
                return 0.5 * (np.matmul(Sz[spin[0]],Sm[spin[1]]) + np.matmul(Sm[spin[0]],Sz[spin[1]])) # T(2,-1)
            if m == 2:
                return 0.5 * np.matmul(Sp[spin[0]],Sp[spin[1]]) # T(2,+2)
            if m == -2:
                return 0.5 * np.matmul(Sm[spin[0]],Sm[spin[1]]) # T(2,-2)
                
        if Rank == 1:
            if m == 0:
                return Sz[spin[0]]
            if m == 1:
                return (-1/np.sqrt(2)) * Sp[spin[0]]
            if m == -1:
                return (1/np.sqrt(2)) * Sp[spin[0]]                       

    def H0_comsop_EigFreq(self,Hz_L,opBasis_L):
        """
        Compute the eigen frequency of the eigen (operator) basis of the Hamiltonian commutation superoperator
        
        INPUT
        -----
        Hz_L: Commutation superoperator Hamiltonina
        opBasis_L: Eigen Operator
        
        OUTPUT
        ------
        return eigen frequency
        """
        
        return np.dot(Hz_L @ opBasis_L,opBasis_L)
                    
    def Relaxation_L(self,Rprocess,R,Sx,Sy,Sz,Sp,Sm):
        """
        Redfield Relaxation in Liouville Space
        
        INPUT
        -----
        Rprocess: "No Relaxation" or 
                  "Phenomenological" or 
                  "Auto-correlated Random Field Fluctuation" or 
                  "Auto-correlated Dipolar Homonuclear" or 
                  "Cross Correlated CSA - Dipolar Hetronuclear"
                  
        Sx: Spin Operator Sx
        Sy: Spin Operator Sy
        Sz: Spin Operator Sz
        Sp: Spin Operator Sp
        Sm: Spin Operator Sm
        
        OUTPUT
        ------
        Rso: Relaxation Superoperator 
                
        """
        if Rprocess == "No Relaxation":
            """
            No Relaxation
            """
            Rso = np.zeros((self.Ldim,self.Ldim))
            
        if Rprocess == "Phenomenological":  
            """
            Phenomenological Relaxation
            """
            Rso = np.zeros((self.Ldim,self.Ldim),dtype=np.cdouble)
            np.fill_diagonal(Rso, R)
            
        if Rprocess == "Auto-correlated Random Field Fluctuation":
            """
            Auto-correlated Random Field Fluctuation Relaxation
            """
            omega_R = 1.0e11
            Rso = np.zeros((self.Ldim,self.Ldim),dtype=np.cdouble)
            for i in range(self.Nspins):
                Rso = Rso + omega_R * (self.SpectralDensity(0,self.tau) * self.DoubleCommutationSuperoperator(Sz[i],Sz[i]) + self.SpectralDensity(self.LarmorF[i],self.tau) * (self.DoubleCommutationSuperoperator(Sp[i],Sm[i]) + self.DoubleCommutationSuperoperator(Sm[i],Sp[i])))
                
        if Rprocess == "Auto-correlated Dipolar Homonuclear":
            """
            Auto-correlated Dipolar Homonuclear Relaxation
            """
            Rso = np.zeros((self.Ldim,self.Ldim),dtype=np.cdouble)
            m = [-2,-1,0,1,2]
            for i in m:
                Rso = Rso + (-1)**i * self.SpectralDensity(i * self.LarmorF[0],self.tau) * self.DoubleCommutationSuperoperator(self.Spherical_Tensor([0,1],2,i,Sx,Sy,Sz,Sp,Sm),self.Spherical_Tensor([0,1],2,-i,Sx,Sy,Sz,Sp,Sm))     
            Rso = Rso * (6/5) * self.bIS**2    

        if Rprocess == "Auto-correlated Dipolar Hetronuclear": 
            """
            Auto-correlated Dipolar Hetronuclear Relaxation
            """
            Rso = np.zeros((self.Ldim,self.Ldim),dtype=np.cdouble)
            
            constant1 = (6/5) * (self.bIS)**2 # (4/5) * (self.bIS)**2
            constant2 = (6/5) * (self.bIS)**2 # (4/5) * (self.bIS)**2
            constant3 = (6/5) * (self.bIS)**2 # (3/10.0) * (self.bIS)**2
            constant4 = (6/5) * (self.bIS)**2 # (3/10.0) * (self.bIS)**2
            constant5 = (6/5) * (self.bIS)**2 # (3/10.0) * (self.bIS)**2
            
            # Zero Quantum Coherence
            ZeroQ_0 = [Sz[0] @ Sz[1]] #  Frequency 0
            ZeroQ_1 = [(-1/4)*Sp[0] @ Sm[1],(-1/4)*Sm[0] @ Sp[1]] # Frequency |difference in Larmor Frequencies|
            ZeroQ_W0 = [0.0, self.LarmorF[0] - self.LarmorF[1]]
            for i in ZeroQ_0: 
                for j in ZeroQ_0: 
                    Rso = Rso + constant1 * self.SpectralDensity(ZeroQ_W0[0],self.tau) * self.DoubleCommutationSuperoperator(i,self.Adjoint(j)) 

            for i in ZeroQ_1: 
                for j in ZeroQ_1: 
                    Rso = Rso + constant2 * self.SpectralDensity(ZeroQ_W0[1],self.tau) * self.DoubleCommutationSuperoperator(i,self.Adjoint(j)) 
                    
            # Single Quantum Coherence
            SingleQ_0 = [Sz[0] @ Sp[1],Sz[0] @ Sm[1]] # Frequency: Larmor of Second Spin
            SingleQ_1 = [Sp[0] @ Sz[1],Sm[0] @ Sz[1]]  # Frequency: Larmor of First Spin
            SingleQ_W0 = [self.LarmorF[1], self.LarmorF[0]]      
            for i in SingleQ_0: 
                for j in SingleQ_0: 
                    Rso = Rso + constant3 * self.SpectralDensity(SingleQ_W0[0],self.tau) * self.DoubleCommutationSuperoperator(i,self.Adjoint(j)) 

            for i in SingleQ_1: 
                for j in SingleQ_1: 
                    Rso = Rso + constant4 * self.SpectralDensity(SingleQ_W0[1],self.tau) * self.DoubleCommutationSuperoperator(i,self.Adjoint(j)) 
                    
            # Double Quantum Coherence
            DoubleQ_0 = [Sp[0] @ Sp[1],Sm[0] @ Sm[1]] # Frequency: |sum of Larmor Frequencies|
            DoubleQ_W0 = [self.LarmorF[1] + self.LarmorF[0]]      
            for i in DoubleQ_0: 
                for j in DoubleQ_0: 
                    Rso = Rso + constant5 * self.SpectralDensity(DoubleQ_W0[0],self.tau) * self.DoubleCommutationSuperoperator(i,self.Adjoint(j)) 
            
        if Rprocess == "Cross Correlated CSA - Dipolar Hetronuclear": 
            """
            Cross Correlated CSA - Dipolar Hetronuclear Relaxation
            
            ATTENTION
            ---------
            I never tested this options, may contain mistakes
            """
            Rso = np.zeros((self.Ldim,self.Ldim),dtype=np.cdouble)
            
            constant1 = (-4/(3 * np.sqrt(5))) * self.bIS * self.Delta_CSA * self.LarmorF[0] * 0.5 * (3 * np.cos(self.Theta_R) - 1)
            constant2 = (4/5) * (self.bIS)**2
            constant3 = (3/10.0) * (self.bIS)**2
            constant4 = (-1/(2 * np.sqrt(5))) * self.bIS * self.Delta_CSA * self.LarmorF[0] * 0.5 * (3 * np.cos(self.Theta_R) - 1)
            constant5 = (3/10.0) * (self.bIS)**2
            
            # Zero Quantum Coherence
            ZeroQ_0 = [Sz[0] @ Sz[1], Sz[0]] #  Frequency 0
            ZeroQ_1 = [(-1/4)*Sp[0] @ Sm[1],(-1/4)*Sm[0] @ Sp[1]] # Frequency |difference in Larmor Frequencies|
            ZeroQ_W0 = [0.0, self.LarmorF[0] - self.LarmorF[1]]
            for i in ZeroQ_0: 
                for j in ZeroQ_0: 
                    Rso = Rso + constant1 * self.SpectralDensity(ZeroQ_W0[0],self.tau) * self.DoubleCommutationSuperoperator(i,self.Adjoint(j)) 

            for i in ZeroQ_1: 
                for j in ZeroQ_1: 
                    Rso = Rso + constant2 * self.SpectralDensity(ZeroQ_W0[1],self.tau) * self.DoubleCommutationSuperoperator(i,self.Adjoint(j)) 
                    
            # Single Quantum Coherence
            SingleQ_0 = [Sz[0] @ Sp[1],Sz[0] @ Sm[1]] # Frequency: Larmor of Second Spin
            SingleQ_1 = [Sp[0] @ Sz[1],Sm[0] @ Sz[1],Sp[0],Sm[0]]  # Frequency: Larmor of First Spin
            SingleQ_W0 = [self.LarmorF[1], self.LarmorF[0]]      
            for i in SingleQ_0: 
                for j in SingleQ_0: 
                    Rso = Rso + constant3 * self.SpectralDensity(SingleQ_W0[0],self.tau) * self.DoubleCommutationSuperoperator(i,self.Adjoint(j)) 

            for i in SingleQ_1: 
                for j in SingleQ_1: 
                    Rso = Rso + constant4 * self.SpectralDensity(SingleQ_W0[1],self.tau) * self.DoubleCommutationSuperoperator(i,self.Adjoint(j)) 
                    
            # Double Quantum Coherence
            DoubleQ_0 = [Sp[0] @ Sp[1],Sm[0] @ Sm[1]] # Frequency: |sum of Larmor Frequencies|
            DoubleQ_W0 = [self.LarmorF[1] + self.LarmorF[0]]      
            for i in DoubleQ_0: 
                for j in DoubleQ_0: 
                    Rso = Rso + constant5 * self.SpectralDensity(DoubleQ_W0[0],self.tau) * self.DoubleCommutationSuperoperator(i,self.Adjoint(j))                    
                                   
        return Rso

    def Lindblad_Dissipator(self,A,B):
        """
        Lindbald Dissipator
        
        INPUT
        -----
        A:
        B:
        
        OUTPUT
        ------
        return Lindblad Dissipator
        """

        #return np.kron(A,B.T) - 0.5 * ( np.kron(np.matmul(B,A), np.eye(self.Vdim)) + np.kron(np.eye(self.Vdim), np.matmul(A.T,B.T)) ) 
        return np.kron(A,B.T) - 0.5 * self.AntiCommutationSuperoperator(B @ A)
        
    def Relaxation_Lindblad(self,Rprocess,Sx,Sy,Sz,Sp,Sm):
        """
        Lindblad Relaxation in Liouville Space
        
        INPUT
        -----
        Rprocess: "No Relaxation" or "Auto-correlated Dipolar Homonuclear" or "Auto-correlated Dipolar Hetronuclear"
        Sx: Spin Operator Sx
        Sy: Spin Operator Sy
        Sz: Spin Operator Sz
        Sp: Spin Operator Sp
        Sm: Spin Operator Sm
        
        OUTPUT
        ------
        Rso: Relaxation Superoperator 
        
        """
        if Rprocess == "No Relaxation":
            """
            No Relaxation
            """
            Rso = np.zeros((self.Ldim,self.Ldim))

        if Rprocess == "Auto-correlated Dipolar Homonuclear":
            """
            Auto-correlated Dipolar Homonuclear Relaxation
            Extreme Narrowing
            """
            Rso = np.zeros((self.Ldim,self.Ldim),dtype=np.cdouble)
            m = [-2,-1,0,1,2]
            for i in m:
                Rso = Rso + (-1)**i * self.SpectralDensity_Lb(i * self.LarmorF[0],self.tau) * self.Lindblad_Dissipator(self.Spherical_Tensor([0,1],2,i,Sx,Sy,Sz,Sp,Sm),self.Spherical_Tensor([0,1],2,-i,Sx,Sy,Sz,Sp,Sm))     
            Rso = Rso * (-6/5) * self.bIS**2 

        if Rprocess == "Auto-correlated Dipolar Hetronuclear": 
            """
            Auto-correlated Dipolar Hetronuclear Relaxation
            """
            Rso = np.zeros((self.Ldim,self.Ldim),dtype=np.cdouble)
            
            constant1 = (6/5) * (self.bIS)**2
            constant2 = (6/5) * (self.bIS)**2
            constant3 = (6/5) * (self.bIS)**2
            constant4 = (6/5) * (self.bIS)**2
            constant5 = (6/5) * (self.bIS)**2
            
            # Zero Quantum Coherence
            ZeroQ_0 = Sz[0] @ Sz[1] #  Frequency 0
            ZeroQ_1 = (-1/4)*Sp[0] @ Sm[1] # Frequency |difference in Larmor Frequencies|
            ZeroQ_2 = (-1/4)*Sm[0] @ Sp[1] # Frequency |difference in Larmor Frequencies|
            ZeroQ_W0 = [0.0, -self.LarmorF[0] + self.LarmorF[1]]

            Rso = Rso + constant1 * self.SpectralDensity_Lb(ZeroQ_W0[0],self.tau) * self.Lindblad_Dissipator(ZeroQ_0,self.Adjoint(ZeroQ_0)) 
            Rso = Rso + constant2 * self.SpectralDensity_Lb(-1 * ZeroQ_W0[1],self.tau) * self.Lindblad_Dissipator(ZeroQ_1,self.Adjoint(ZeroQ_1)) 
            Rso = Rso + constant2 * self.SpectralDensity_Lb(1 * ZeroQ_W0[1],self.tau) * self.Lindblad_Dissipator(ZeroQ_2,self.Adjoint(ZeroQ_2)) 
                                        
            # Single Quantum Coherence
            SingleQ_0 = Sz[0] @ Sp[1] # Frequency: Larmor of Second Spin
            SingleQ_1 = Sz[0] @ Sm[1]  # Frequency: Larmor of Second Spin
            SingleQ_2 = Sp[0] @ Sz[1]  # Frequency: Larmor of First Spin
            SingleQ_3 = Sm[0] @ Sz[1]  # Frequency: Larmor of First Spin
            SingleQ_W0 = [self.LarmorF[1], self.LarmorF[0]]      

            Rso = Rso + constant3 * self.SpectralDensity_Lb(1 * SingleQ_W0[0],self.tau) * self.Lindblad_Dissipator(SingleQ_0,self.Adjoint(SingleQ_0))  
            Rso = Rso + constant3 * self.SpectralDensity_Lb(-1 * SingleQ_W0[0],self.tau) * self.Lindblad_Dissipator(SingleQ_1,self.Adjoint(SingleQ_1))  
            Rso = Rso + constant4 * self.SpectralDensity_Lb(1 * SingleQ_W0[1],self.tau) * self.Lindblad_Dissipator(SingleQ_2,self.Adjoint(SingleQ_2)) 
            Rso = Rso + constant4 * self.SpectralDensity_Lb(-1 * SingleQ_W0[1],self.tau) * self.Lindblad_Dissipator(SingleQ_3,self.Adjoint(SingleQ_3)) 
                                        
            # Double Quantum Coherence
            DoubleQ_0 = Sp[0] @ Sp[1] # Frequency: |sum of Larmor Frequencies|
            DoubleQ_1 = Sm[0] @ Sm[1] # Frequency: |sum of Larmor Frequencies|
            DoubleQ_W0 = [self.LarmorF[1] + self.LarmorF[0]]      

            Rso = Rso + constant5 * self.SpectralDensity_Lb(1 * DoubleQ_W0[0],self.tau) * self.Lindblad_Dissipator(DoubleQ_0,self.Adjoint(DoubleQ_0))
            Rso = Rso + constant5 * self.SpectralDensity_Lb(-1 * DoubleQ_W0[0],self.tau) * self.Lindblad_Dissipator(DoubleQ_1,self.Adjoint(DoubleQ_1))
                                        
            Rso = -1 * Rso
                                
        return Rso    

    def Relaxation_Phenomenological(self,R1,R2):
        """
        Phenomenological Relaxation for using unitary propagator
        This function may be scrapped or modified in PyOR version 1
        
        INPUT
        -----
        R1: Longitudinal Relaxation
        R2: Transverse Relaxation
        
        OUTPUT
        ------
        Rso: Relaxation Matrix
        """
        dim = self.Vdim
        Rso = R2 * np.ones((dim,dim),dtype=np.cdouble)
        np.fill_diagonal(Rso, R1)
        return Rso
            
    def Relaxation_H(self,rho,Rprocess,R1,R2,Sx,Sy,Sz,Sp,Sm):
        """
        Redfield Relaxation in Hilbert space
        
        ATTENTION
        ---------
        This function is called by Evolution_H(self,rhoeq,rho,Sx,Sy,Sz,Sp,Sm,Hamiltonian,dt,Npoints,method,Rprocess), check it for more informations.
        """
        
        if Rprocess == "No Relaxation":
            """
            No Relaxation
            """
            dim = self.Vdim
            Rso = np.zeros((dim,dim))
            
        if Rprocess == "Phenomenological":
            """
            Phenomenological Relaxation
            """
            dim = self.Vdim 
            Rso = R2 * np.ones((dim,dim))
            np.fill_diagonal(Rso, R1) 
            Rso = np.multiply(Rso,rho)               

        if Rprocess == "Phenomenological Input":
            """
            Phenomenological Relaxation
            Relaxation Matrix is given as input
            see function, Relaxation_Phenomenological_Input(R)
            """
            Rso = np.multiply(self.R_input,rho) 
            
        if Rprocess == "Auto-correlated Random Field Fluctuation":
            """
            Auto-correlated
            Random Field Fluctuation
            """
            omega_R = 1.0e11 # Default: 1.0e11
            dim = self.Vdim
            Rso = np.zeros((dim,dim))
            for i in range(self.Nspins):
                Rso = Rso + omega_R * (self.SpectralDensity(0,self.tau) * self.DoubleCommutator(Sz[i],Sz[i],rho) + self.SpectralDensity(self.LarmorF[i],self.tau) * self.DoubleCommutator(Sp[i],Sm[i],rho) + self.SpectralDensity(-1 * self.LarmorF[i],self.tau) * self.DoubleCommutator(Sm[i],Sp[i],rho))
                
        if Rprocess == "Auto-correlated Dipolar Heteronuclear":
            """ 
            Heteronuclear, Auto-correlated
            Dipolar Relaxation
            Double Commutator Select relaxation pathways
            Spectral density function: Determined strength of relaxation pathway
            Auto relaxation: Decay of coherence/population, ie, diagonal elements in relaxation superoperator 
            Cross relaxation: Transfer of coherence/population to another, ie, off-diagonal elements in the relaxation superoperator
            
            ATTENTION
            ---------
            By default, self.spin1 = 0 and self.spin2 = 1
            see function: Relaxation_Spins(self,spin1,spin2)
            """   
            
            # Second Rank Tensor
            T0 = np.matmul(Sz[self.spin1],Sz[self.spin2]) - (1/4) * np.matmul(Sp[self.spin1],Sm[self.spin2]) - (1/4) * np.matmul(Sm[self.spin1],Sp[self.spin2]) # T(2,0)
            T1 = np.matmul(Sz[self.spin1],Sp[self.spin2]) + np.matmul(Sp[self.spin1],Sz[self.spin2]) # T(2,+1)
            T2 = np.matmul(Sz[self.spin1],Sm[self.spin2]) + np.matmul(Sm[self.spin1],Sz[self.spin2]) # T(2,-1)
            T3 = np.matmul(Sp[self.spin1],Sp[self.spin2]) # T(2,+2)
            T4 = np.matmul(Sm[self.spin1],Sm[self.spin2]) # T(2,-2)
            
            # Spacial Constants
            J0 = (4/5) * (self.bIS)**2 
            J1 = (3/10) * (self.bIS)**2
            J2 = J1
            J3 = J1
            J4 = J1
            
            # Eigen Operators (Woff1 I1z + Woff2 I2z) and Eigen Frequencies
            V1 = np.matmul(Sz[self.spin1],Sz[self.spin2]) ; W1 = 0.0
            V2 = (-1/4) * np.matmul(Sp[self.spin1],Sm[self.spin2]); W2 = -self.LarmorF[self.spin1] + self.LarmorF[self.spin2]
            V3 = (-1/4) * np.matmul(Sm[self.spin1],Sp[self.spin2]); W3 = self.LarmorF[self.spin1] - self.LarmorF[self.spin2]
            V4 = np.matmul(Sz[self.spin1],Sp[self.spin2]); W4 = self.LarmorF[self.spin2]
            V5 = np.matmul(Sz[self.spin1],Sm[self.spin2]); W5 = -self.LarmorF[self.spin2]
            V6 = np.matmul(Sp[self.spin1],Sz[self.spin2]); W6 = self.LarmorF[self.spin1]
            V7 = np.matmul(Sm[self.spin1],Sz[self.spin2]); W7 = -self.LarmorF[self.spin1]
            V8 = np.matmul(Sp[self.spin1],Sp[self.spin2]); W8 = -self.LarmorF[self.spin1] - self.LarmorF[self.spin2]
            V9 = np.matmul(Sm[self.spin1],Sm[self.spin2]); W9 = self.LarmorF[self.spin1] + self.LarmorF[self.spin2]
            
            # Relaxation Matrix
            dim = self.Vdim
            Rso = np.zeros((dim,dim))
            
            Rso = Rso + self.DoubleCommutator(T0,self.Adjoint(V1),rho) * J0 * self.SpectralDensity(W1,self.tau)
            Rso = Rso + self.DoubleCommutator(T0,self.Adjoint(V2),rho) * J0 * self.SpectralDensity(W2,self.tau)
            Rso = Rso + self.DoubleCommutator(T0,self.Adjoint(V3),rho) * J0 * self.SpectralDensity(W3,self.tau)
            
            Rso = Rso + self.DoubleCommutator(T1,self.Adjoint(V4),rho) * J1 * self.SpectralDensity(W4,self.tau)
            Rso = Rso + self.DoubleCommutator(T1,self.Adjoint(V6),rho) * J1 * self.SpectralDensity(W6,self.tau)
            
            Rso = Rso + self.DoubleCommutator(T2,self.Adjoint(V5),rho) * J2 * self.SpectralDensity(W5,self.tau)
            Rso = Rso + self.DoubleCommutator(T2,self.Adjoint(V7),rho) * J2 * self.SpectralDensity(W7,self.tau)
            
            Rso = Rso + self.DoubleCommutator(T3,self.Adjoint(V8),rho) * J3 * self.SpectralDensity(W8,self.tau)
            
            Rso = Rso + self.DoubleCommutator(T4,self.Adjoint(V9),rho) * J4 * self.SpectralDensity(W9,self.tau)

        if Rprocess == "Auto-correlated Dipolar Homonuclear":
            """
            Homonuclear Auto-correlated
            Dipolar Relaxation
            Extreme Narrowing
            """
            Rso = np.zeros((self.Vdim,self.Vdim),dtype=np.cdouble)
            m = [-2,-1,0,1,2]
            for i in m:
                Rso = Rso + (-1)**i * self.SpectralDensity(i * self.LarmorF[0],self.tau) * self.DoubleCommutator(self.Spherical_Tensor([0,1],2,i,Sx,Sy,Sz,Sp,Sm),self.Spherical_Tensor([0,1],2,-i,Sx,Sy,Sz,Sp,Sm),rho) 
            Rso = Rso * (6/5) * self.bIS**2    
                
        return Rso      
        
    def Transform_Redkite(self,R_L,SP,SM,SZ,Coherenceorder):
        """
        Transform Relaxation Matric to Redkite form (Liouvillian only). Two spin half only.
        This section will be modified in PyOR version 1
        
        INPUT
        -----
        R_L: Relaxation Superoperator
        SP: Spin Operator S+
        SM: Spin Operator S-
        SZ: Spin Operator SZ
        Coherenceorder: Options "0,1,-1,2,-2" or "-2,-1,0,1,2"
        
        OUTPUT
        ------
        R_red: RedKite matrix
        B: Basis Operators (Defined inside this function). In PyOR version 1, the Basis operator will be given as input
        
        ATTENTION
        ---------
        While plotting RedKite make following option:
        System.PlotLabel_Hilbert = False
        System.Redkite_Label_SpinDynamica = True        
        
        """
        Sp = SP.real
        Sm = SM.real
        Sz = SZ.real
        R_red = np.zeros((self.Ldim, self.Ldim),dtype=np.double)
        B = np.zeros((self.Ldim, self.Ldim),dtype=np.double)
        
        Two_spin_half = False
        Redkite_SpinDynamica = True
        
        if self.Nspins == 2:
            if Coherenceorder == "0,1,-1,2,-2":
            
                B[:,0] = self.Vector_L((1/2) * eye(self.Vdim)).T # Identity
                B[:,1] = self.Vector_L(Sz[0]).T # Sz
                B[:,2] = self.Vector_L(Sz[1]).T # Iz       
                B[:,3] = self.Vector_L(2 * np.matmul(Sz[0],Sz[1])).T # SzIz
                B[:,4] = self.Vector_L(np.matmul(Sp[0],Sm[1])).T # SpIm       
                B[:,5] = self.Vector_L(np.matmul(Sm[0],Sp[1])).T # SmIp       
                B[:,6] = self.Vector_L((1/np.sqrt(2)) * Sp[0]).T # Sp       
                B[:,7] = self.Vector_L(np.sqrt(2) * np.matmul(Sp[0],Sz[1])).T # SpIz       
                B[:,8] = self.Vector_L((1/np.sqrt(2)) * Sp[1]).T # Ip
                B[:,9] = self.Vector_L(np.sqrt(2) * np.matmul(Sz[0],Sp[1])).T # SzIp       
                B[:,10] = self.Vector_L((1/np.sqrt(2)) * Sm[0]).T # Sm       
                B[:,11] = self.Vector_L(np.sqrt(2) * np.matmul(Sm[0],Sz[1])).T # SmIz        
                B[:,12] = self.Vector_L((1/np.sqrt(2)) * Sm[1]).T # Im         
                B[:,13] = self.Vector_L(np.sqrt(2) * np.matmul(Sz[0],Sm[1])).T # SzIm      
                B[:,14] = self.Vector_L(np.matmul(Sp[0],Sp[1])).T # SpIp       
                B[:,15] = self.Vector_L(np.matmul(Sm[0],Sm[1])).T # SmIm       

                display(Latex(r'$E$,$S_{z}$,$I_{z}$,$S_{z}I_{z}$,$S_{+}I_{-}$,$S_{-}I_{+}$,$S_{+}$,$S_{+}I_{z}$,$I_{+}$,$S_{z}I_{+}$,$S_{-}$,$S_{-}I_{z}$,$I_{-}$,$S_{z}I_{-}$,$S_{+}I_{+}$,$S_{-}I_{-}$')) 
            
            if Coherenceorder == "-2,-1,0,1,2":
            
                B[:,0] = self.Vector_L(np.matmul(Sm[0],Sm[1])).T # SmIm 
                B[:,1] = self.Vector_L(np.sqrt(2) * np.matmul(Sm[0],Sz[1])).T # SmIz 
                B[:,2] = self.Vector_L(np.sqrt(2) * np.matmul(Sz[0],Sm[1])).T # SzIm         
                B[:,3] = self.Vector_L((1/np.sqrt(2)) * Sm[0]).T # Sm 
                B[:,4] = self.Vector_L((1/np.sqrt(2)) * Sm[1]).T # Im         
                B[:,5] = self.Vector_L(np.matmul(Sm[0],Sp[1])).T # SmIp       
                B[:,6] = self.Vector_L(np.matmul(Sp[0],Sm[1])).T # SpIm       
                B[:,7] = self.Vector_L(2 * np.matmul(Sz[0],Sz[1])).T # SzIz      
                B[:,8] = self.Vector_L(Sz[0]).T # Sz
                B[:,9] = self.Vector_L(Sz[1]).T # Iz             
                B[:,10] = self.Vector_L((1/2) * eye(self.Vdim)).T 
                B[:,11] = self.Vector_L(np.sqrt(2) * np.matmul(Sp[0],Sz[1])).T # SpIz        
                B[:,12] = self.Vector_L(np.sqrt(2) * np.matmul(Sz[0],Sp[1])).T # SzIp       
                B[:,13] = self.Vector_L((1/np.sqrt(2)) * Sp[0]).T # Sp    
                B[:,14] = self.Vector_L((1/np.sqrt(2)) * Sp[1]).T # Ip      
                B[:,15] = self.Vector_L(np.matmul(Sp[0],Sp[1])).T # SpIp 

                display(Latex(r'$S_{-}I_{-}$,$S_{-}I_{z}$,$S_{z}I_{-}$,$S_{-}$,$I_{-}$,$S_{-}I_{+}$,$S_{+}I_{-}$,$S_{z}I_{z}$,$I_{z}$,$S_{z}$,$E$,$S_{+}I_{z}$,$S_{z}I_{+}$,$S_{+}$,$I_{+}$,$S_{+}I_{+}$'))         
        
        method1 = True
        if method1:
            for i in range(self.Ldim):
                for j in range(self.Ldim):
                    R_red[i][j] = B[:,i].T @ R_L.real @ B[:,j] / (B[:,i].T @ B[:,i])
            #R_red[R_red < 0] = 0 # Make negative values zero
            return R_red, B          
        
        orthogonality_check = False
        if orthogonality_check:
            for i in range(self.Ldim):
                for j in range(self.Ldim):
                    R_red[i,j] = np.dot(B[:,i].T , B[:,j])   
            return R_red, B           
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Frequency Analyzer
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
            
class Fanalyzer:
    def __init__(self,Mx,My,ax,fig,line1,line2,line3,line4,vline1,vline2,vline3,vline4,text1,text2):
        """
        Multi-mode Maser Analyzer
        Time domain (Figure 1,1) to Frequency domain (Figure 1,2)
        Frequency domain (Figure 2,1) to Time domain (Figure 2,2)
        
        Select region of FID (Figure 1,1), its Fourier Transform (in red) will be shown in Figure 1,2. (Blue for entire Fourier Transform)
        Select region of frequencies (Figure 2,1), corresponding time signal (in red) will be shown in Figure 2,2. (Blue for total FID) 
        """
        
        self.x1, self.y1 = line1.get_data()
        self.x2, self.y2 = line2.get_data()
        self.x3, self.y3 = line3.get_data()
        self.x4, self.y4 = line4.get_data()
        self.dt = self.x1[1] - self.x1[0]
        self.fs = 1.0/self.dt
        self.ax = ax
        self.fig = fig
        self.vline1 = vline1
        self.vline2 = vline2
        self.text1 = text1
        self.vline3 = vline3
        self.vline4 = vline4 
        self.text2 = text2
        self.Mx = Mx
        self.My = My
        self.Mt = Mx + 1j * My

    def button_press(self,event):
        if event.inaxes is self.ax[0,0]:
            x1, y1 = event.xdata, event.ydata
            global x1in
            x1in = min(np.searchsorted(self.x1, x1), len(self.x1) - 1)
	    
        if event.inaxes is self.ax[1,0]:
            x3, y3 = event.xdata, event.ydata
            global x3in
            x3in = min(np.searchsorted(self.x3, x3), len(self.x3) - 1)

        if event.inaxes is self.ax[0,1]:
            x2, y2 = event.xdata, event.ydata
            global x2in
            x2in = x2
            self.vline1.set_xdata([x2in])
            plt.draw()

        if event.inaxes is self.ax[1,1]:
            x4, y4 = event.xdata, event.ydata
            global x4in
            x4in = x4
            self.vline3.set_xdata([x4in])
            plt.draw()

    def button_release(self,event):
        if event.inaxes is self.ax[0,0]:
            x1, y1 = event.xdata, event.ydata
            global x1fi
            x1fi = min(np.searchsorted(self.x1, x1), len(self.x1) - 1)
	        
            spectrum = np.fft.fft(self.Mt[x1in:x1fi])
            spectrum = np.fft.fftshift(spectrum)
            freq = np.linspace(-self.fs/2,self.fs/2,spectrum.shape[-1])
            la = self.ax[0,1].get_lines()
            la[-1].remove()
            line2, = self.ax[0,1].plot(self.x2,np.absolute(self.y2),"-", color='blue')
            #line, = self.ax[0,1].plot(freq,spectrum,"-", color='red')
            line, = self.ax[0,1].plot(freq,np.absolute(spectrum),"-", color='red')
            plt.draw()

        if event.inaxes is self.ax[1,0]:
            x3, y3 = event.xdata, event.ydata
            global x3fi
            x3fi = min(np.searchsorted(self.x3, x3), len(self.x3) - 1)
            y3 = self.y3
            print(y3.shape)
            window = np.zeros((y3.shape[-1]))
            window[x3in:x3fi] = 1.0
            sig = np.fft.ifftshift(y3*window)
            sig = np.fft.ifft(sig)
            t = np.linspace(0,self.dt*y3.shape[-1],y3.shape[-1])
            lb = self.ax[1,1].get_lines()
            lb[-1].remove()
            line4, = self.ax[1,1].plot(self.x4,self.y4,'-', color='blue')
            line, = self.ax[1,1].plot(t,sig,"-", color='red')
            plt.draw()

        if event.inaxes is self.ax[0,1]:
            x2, y2 = event.xdata, event.ydata
            global x2fi
            x2fi = x2
            self.vline2.set_xdata([x2fi])
            self.text1.set_text(f'Freq={abs(x2fi-x2in):1.5f} Hz')
            plt.draw()

        if event.inaxes is self.ax[1,1]:
            x4, y4 = event.xdata, event.ydata
            global x4fi
            x4fi = x4
            self.vline4.set_xdata([x4fi])
            self.text2.set_text(f'Time={abs(x4fi-x4in):1.5f} s')
            plt.draw()        
                         
