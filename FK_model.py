"""Calculator for the Frenkel-Kontorova Model"""

# fk.py
# Frenkel-Kontorova Model  
# These routines integrate with the ASE simulation environment
# Pai Li (Aug 2021)
# lipai@mail.ustc.edu.cn

import numba as nb
import numpy as np
import time
import os
from ase.neighborlist import NeighborList
from ase.calculators.calculator import Calculator, all_changes

#from fk_ef import fk_ef

mRy2eV=13.6057/1000
V_0= -8.025*mRy2eV
V_R=  1.483*mRy2eV
V_I= -0.087*mRy2eV
V_2= -0.146*mRy2eV
a=2.884
b=2.7744
k=90*mRy2eV

amplitude=1.0   # 1 Angstrom

class FK(Calculator):
    r"""
    Phys. Rev. Lett. 69, 2455 (1992)
    """

    implemented_properties = ['energy', 'energies','forces']

    def __init__(self, restart=None,atoms=None,recons=None,
                 ignore_bad_restart_file=Calculator._deprecated,
                 label=os.curdir, **kwargs):

        Calculator.__init__(self, restart, ignore_bad_restart_file,
                            label, atoms, **kwargs)

        if(atoms==None):
            raise ValueError('atoms cannot be None')
        if(recons==None):
            raise ValueError('recons cannot be None. 1 for 1st recons, 2 for 2nd recons')
  
   # def get_nb_list(self,atoms):     # We need a fixed neighbor_list in all calc
        cutoffs = 4.7/2.884* np.ones(len(atoms))
        self.neighbors = NeighborList(cutoffs,
                                      self_interaction=False,
                                      bothways=True)
        self.neighbors.update(atoms)
        self.nb_list=[]
        self.nb_list_np=np.zeros((len(atoms),7))-1
        for i in range(len(atoms)):
            # print(self.neighbors.get_neighbors(i))
            indices, offsets = self.neighbors.get_neighbors(i)
            if(len(indices)==7 or len(indices)==5): print(atoms[i].x,len(indices))
            #print("indices: ",len(indices))
            #print(len(indices))
            #print("offsets: ",offsets)
            self.nb_list.append(indices.copy())
            for ind,j in enumerate(indices):
                self.nb_list_np[i,ind]=j

        self.energies = np.empty(len(atoms))
        self.forces = np.empty((len(atoms), 3))

        pi=np.pi
        self.g2=np.zeros((3,2))
        self.g3=np.zeros((6,2))
        if(recons==1):
            for i in range(3):
                self.g2[i,0]=np.cos(i*2*pi/3)   # 0 2*pi/3 4*pi/3
                self.g2[i,1]=np.sin(i*2*pi/3)
            for i in range(6):
                self.g3[i,0]=np.cos(pi/6+i*pi/3)   # pi/6 3*pi/6 ...
                self.g3[i,1]=np.sin(pi/6+i*pi/3)
        elif(recons==2):
            for i in range(3):
                self.g2[i,0]=np.cos(i*2*pi/3-pi/6)   # 0 2*pi/3 4*pi/3
                self.g2[i,1]=np.sin(i*2*pi/3-pi/6)
            for i in range(6):
                self.g3[i,0]=np.cos(i*pi/3)   # pi/6 3*pi/6 ...
                self.g3[i,1]=np.sin(i*pi/3)
        else:
            raise ValueError('recons value error')

        self.g2=self.g2*4*np.pi/a/np.sqrt(3)
        self.g3=self.g3*4*np.pi/a
        #print("g2: ",self.g2)
        #print("g3: ",self.g3)

        self.cell=np.array([atoms.cell[0],atoms.cell[1],atoms.cell[2]])

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.energy = 0.0
        self.energies[:] = 0
        self.forces[:] = 0.0

        #self.calc(atoms,natoms,Fvi,Fbi)

        #fk_ef(self.cell,natoms,self.nb_list_np,atoms.positions,self.energies,self.forces,self.g2,self.g3) 
        #aa,bb=fk_ef(atoms.positions,self.energies,self.forces,self.g2,self.g3,natoms) 
        #print("1: ",time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
        #self.energies,self.forces,=fk_ef(atoms.positions,self.g2,self.g3,natoms) 
        #print("2: ",time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))

        #print("1: ",time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
        self.calc(atoms)
        #print("1: ",time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))

        self.energy = self.energies.sum()
        self.results['energy'] = self.energy
        self.results['energies'] = self.energies
        self.results['forces'] = self.forces

    @nb.jit()
    def calc(self,atoms):
        Fvi=np.empty(3)
        Fbi=np.empty(3)
        for i in range(len(atoms)):
             x=atoms[i].position[0]
             y=atoms[i].position[1]
             ri=np.array([x,y-amplitude*np.sin(0.05*2*np.pi+x*2*np.pi/311.472)])
             Evi=V_0
             Fvi[:]=0
             for j in range(3):    # g2
                 dot=np.dot(self.g2[j],ri)
                 cos=np.cos(dot)
                 sin=np.sin(dot)
                 Evi+=2*(V_R*cos-V_I*sin)
                 Fvi[:2]+=-2*(V_R*sin+V_I*cos)*self.g2[j]  # forces on only x and y direction
             for j in range(6):    # g3
                 dot=np.dot(self.g3[j],ri)
                 Evi+=V_2*np.cos(dot)
                 Fvi[:2]+=-V_2*np.sin(dot)*self.g3[j]      # forces on only x and y direction

             Ebi=0
             Fbi[:]=0
             dists=atoms.get_distances(i,self.nb_list[i],mic=True,vector=True)
             #for j in self.nb_list[i]:  # neighbor list of atom_i
             for dist in dists:
                 #l_j= np.linalg.norm(ri-atoms[j].position[:2])
                 l_j=np.linalg.norm(dist)
                 #print(l_j)
                 Ebi+=(l_j-b)**2
                 #Fbi[:2]+=(l_j-b)*(ri-atoms[j].position[:2])
                 Fbi+=(l_j-b)*(-dist)/l_j
             Ebi=Ebi*k*0.5*0.5 # one 0.5 is in the equation, one 0.5 is to cancel the double counting
             Fbi=Fbi*k         # no 0.5 in equation, no 0.5 for cancel
             #self.energies[i]=Evi+Ebi
             #print("Ei: ",i,Evi,Ebi)
             #self.forces[i]=-(Fvi+Fbi)
             self.energies[i]=Evi+Ebi
             self.forces[i]=-Fvi-Fbi

    def get_potential_energy_surface(self,N=5):
        for x in np.linspace(0, 311.472, num=N*312, endpoint=True):
            for y in np.linspace(0, 74.928, num=N*75, endpoint=True):
                ri=np.array([x,y-amplitude*np.sin(0.05*2*np.pi+x*2*np.pi/311.472)])
                e=V_0
                for j in range(3):    # g2
                    dot=np.dot(self.g2[j],ri)
                    cos=np.cos(dot)
                    sin=np.sin(dot)
                    e+=2*(V_R*cos-V_I*sin)
                for j in range(6):    # g3
                    dot=np.dot(self.g3[j],ri)
                    e+=V_2*np.cos(dot)
                print(x,y,e)
