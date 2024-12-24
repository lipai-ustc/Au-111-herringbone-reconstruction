Extra info for the following work:  
[_Science advances_ __2022__, 8, eabq2900](https://doi.org/10.1126/sciadv.abq2900)  
Origin of the herringbone reconstruction of Au(111) surface at the atomic scale  

The file 311.4499.dump is the original symmetric surface model.  
All structural files can be visualized via OVITO.  
The training set can be found in [https://figshare.com/articles/dataset/Au_111_herringbone_reconstruction/25909786]

### Strategies for generating asymmetric reconstructed Au(111) surface model:  
1. DFT opt 22*sqrt(3) structure (CONTCAR)  
2. Get x/y, _i.e._ the size of model box   
> get_xy.py  
```
import numpy as np
import os
N=9
a=np.arange(0,N+1) # cut into 10 parts
bl=2.883795478 # Au-Au bond length
l0=bl*54*2 # l0 should be divided exactly by 3*bl
la=l0/2+a*(l0/2/N)
La=la*2

#yl=bl*22*2/np.sqrt(3)
yl=[bl*14*np.sqrt(3),bl*15*np.sqrt(3)]

np.savetxt("x",La,fmt='%.4f')
np.savetxt("y",yl,fmt='%.4f')

for y in yl:
    for ni,x in enumerate(La):
        os.system("mkdir -p %.4f/%.4f"%(y,x))
        f=open("%.4f/%.4f/box"%(y,x),'w')
        d=2*ni*(l0/2/N)/np.sqrt(3)
        f.write("%.6f    %.6f    %.6f\n"%(l0,d,0))
        f.write("%.6f    %.6f    %.6f\n"%(0,y,0))
        f.write("%.6f    %.6f    %.6f\n"%(0,0, 21.41839981))
        f.close()
```
3. Get twin structure and fit into the model box  
> get_str.sh  
```
for y in `cat y`
do
    for x in `cat x`
    do

        yy=`echo $y*4|bc`
        echo box $x $yy  0 > polyX.txt
        echo node 0 0 0 0° 0° -60° >> polyX.txt 
        cp CONTCAR polyX.txt $y/$x
        cp get_twins.py $y/$x

        cd $y/$x
            atomsk --polycrystal CONTCAR polyX.txt -shift 0.45*box 0.45*box 0 -wrap POSCAR
            # atomsk build polycrystal with center at (0,0,0), we shift it to (0.45,0.45,0) 
            # so that boundaries are outside of the area that will be fitted to the final box.
            rm POSCAR_*
            mv POSCAR CONTCAR
            a=`grep F CONTCAR | awk '{if($1<5) print}' |head -1|awk '{printf("%.8f\n",-$1)}'`
            shift_atoms CONTCAR POSCAR $a 0 0
            #shift so that x=0 and therefore x=cell[0,0]/2 is the mirror symmetry center
            rm POSCAR_* CONTCAR
            xx=`echo $x/2 |bc -l`
            sed -i s/lipai/$xx/ get_twins.py
            python get_twins.py 
            # in get_twins.py, only surf part did the mirror symmetry operation
            mv POSCAR POSCAR-bk

            for j in surf.vasp subs.vasp
            do
              mv $j CONTCAR
              atomsk CONTCAR -shift -0.2 -0.2 0 -wrap POSCAR
              mv POSCAR $j
            done
            # shift both surf.vasp and subs.vasp
            # so that there is (almost) no atom locates very close to the boundary. 
            # Otherwise, it might lost atoms in the final box after fit_box operation.

            echo "Au" > surf.pos
            echo "1.00" >>surf.pos
            cat box >> surf.pos
            sed '1,5d' surf.vasp >> surf.pos

            echo "Au" > subs.pos
            echo "1.00" >>subs.pos
            cat box >> subs.pos
            sed '1,5d' subs.vasp >> subs.pos

            python ../../fit_box.py surf.pos surf.pos1
            python ../../fit_box.py subs.pos subs.pos1
            
            for j in surf.pos1 subs.pos1
            do
                mv $j CONTCAR
                atomsk CONTCAR -shift 0.2*box 0 0 -wrap -remove-doubles 0.3 POSCAR
                mv POSCAR $j
            done

        cd ../..
    done
done
```
> get_twins.py
```
from ase.io import read,write
from ase import Atom
a=read("POSCAR")
b=a.copy()
del b[[atom.index for atom in a if atom.z>15]]
write("subs.vasp",b)

xlen=lipai # symmetry axis
del a[[atom.index for atom in a if atom.z<15]]
del a[[atom.index for atom in a if (atom.x>xlen+0.8 and atom.x<a.cell[0,0]-0.8)]]
# we do not delete atoms that are very close to the mirror symmetry center
# other wise it might have overlap

for i in a:
  b=Atom('Au',position=i.position)
  b.x=2*xlen-b.x
  if(b.x>xlen+0.8 and b.x<a.cell[0,0]-0.8):
    a.append(b)

write("surf.vasp",a)
```
> fit_box.py
```
from ase.io import read, write
import sys
a=read(sys.argv[1],format='vasp')
mask=[i.index for i in a if i.scaled_position[0]<0 or i.scaled_position[0]>1 or i.scaled_position[1]<0 or i.scaled_position[1]>1]  
del a[mask]
write(sys.argv[2],a,format='vasp',direct=True)
```
4. Box periodic vector problem
The box2 we defined is in the form of upper triangular matrix, which becomes a lower triangular matrix after converting the structure to cif and then back to POSCAR format for surf.pos1 (surface part, _i.e._ the top layer). We therefore need to do the same transitions for subs.pos1 (substrat part, _i.e._ layers except the top layer).  
> up2down.py
```
from ase.io import read, write
import sys
for  i in names:
    a=read(i,format='vasp')
    write(i+".cif",a)
    a=read(i+".cif")
    write(i,a,format='vasp')
```
5. Combine surf.pos1 and subs.pos1   
The reason why we split the structure into surf and subs parts is that Materials Studio is very slow to handle very large systems and meanwhile only the surface part (the first layer) requres further adjustment in MS. Now we have these two parts and can merge them into one model.
```
mkdir combine 
cp MS_cif_out/*.surf combine/
cp subs_pos/*.subs combine/
cd combine
for  i in 311.4499  346.0555  380.6610  415.2665  449.8721  484.4776  519.0832  553.6887  588.2943  622.8998
do
    awk '{if(NR==7) print "lipai"; else print }' $i.subs >$i.vasp
    awk '{if(NR>8) print $0 }' $i.cif.surf >>$i.vasp
    a=`awk '{if(NR==7) print $1}' $i.subs`
    b=`awk '{if(NR==7) print $1}' $i.cif.surf`
    c=`echo $a+$b |bc`
    sed -i s/lipai/$c/ $i.vasp
done
```
6. Other things need to pay attention
Keep the surface density constant, equaling to the full occupation of n*sqrt(3) structure  
pos2lmp: We use ASE to convert POSCAR file to lammps structure file.  
> in.lammps
```
units            metal
boundary         p p p
atom_style       atomic
#pair_style      table spline 8000
#pair_coeff      * * Au.tb_sma Au_TB-SMA 14

read_data       in.lmp
mass 1 196.96655
mass 1 196.96655
thermo 100
region 1 block INF INF INF INF INF 6
group fixed region 1
#group fixed type 1
fix force fixed setforce 0.0 0.0 0.0
pair_style      eam/alloy
pair_coeff      * * ../../Au-Grochola-JCP05.eam.alloy Au
dump		myDump1 all xyz 1000 1.xyz
dump_modify	myDump1 element Au
minimize        0.0 1e-6 10000 100000
###timestep        0.005

compute         myStress all stress/atom NULL
#compute         myVol all voronoi/atom
dump            myDump2 all custom 1 1.stress type x y z c_myStress[1] c_myStress[2] c_myStress[3]
dump            myDump3 all custom 1 1.force id fx fy fz
run             1

#dump		 mydump all xyz 100 1.xyz
#dump_modify		mydump element Au

#dump		1 all xyz 1 2.xyz
#dump_modify		1 element Cu
```
Deal with lammps results:
> conv1.sh
```
mkdir xyz
for i in 311.4499  346.0555  380.6610  415.2665  449.8721  484.4776  519.0832  553.6887  588.2943  622.8998
do
  cd $i
    python ../1to2.py
    filename="POSCAR"
    a=`awk '{if(NR==3) print $1}' $filename`
    b1=`awk '{if(NR==4) print $1}' $filename`
    b2=`awk '{if(NR==4) print $2}' $filename`
    c=`awk '{if(NR==5) print $3}' $filename`
    awk -v a=$a -v b1=$b1 -v b2=$b2 -v c=$c '{
    if(NR==2) 
      printf("Properties=species:S:1:pos:R:3 Lattice=\"%f 0 0 %f %f 0 0 0 %f\" pbc=\"T T T\"\n",a,b1,b2,c);
    else print }' 2.xyz >3.xyz
    cp 3.xyz ../xyz/$i.xyz
  cd ..
done
```
> 1to2.py
```
from ase.io import read,write
a=read("1.xyz",index=":")
write("2.xyz",a[-1])
GAP calculation
from ase.io import read,write
from ase.optimize import MDMin
from ase.optimize import BFGS
from quippy.potential import Potential
from ase.constraints import FixAtoms

Au=read("3.xyz")
calc=Potential(param_filename='../../../../potential/Au.xml')
Au.set_calculator(calc)
Au.set_constraint([FixAtoms(mask=[a.z<6 for a in Au])])
write('CONTCAR0.vasp',Au)

#dyn=MDMin(Au,trajectory='opt.traj')
dyn=MDMin(Au)
dyn.run(fmax=0.02)
write('opt1.xyz',Au)
write('CONTCAR1.vasp',Au)

dyn.run(fmax=0.005)
write('opt2.xyz',Au)
write('CONTCAR2.vasp',Au)
```
