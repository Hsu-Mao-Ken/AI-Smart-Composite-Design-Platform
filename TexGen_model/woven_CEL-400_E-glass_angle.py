#!/usr/bin/env python
# coding: utf-8

# In[ ]:

#Epoxy(CEL-400)
#E-glass

import os
from TexGen.Core import *
from math import *
from random import randint
from random import uniform
import csv

name = 'CEL-400_E-glass'
file_geometry_path = 'D:/TexGen_table/'+name+'/'+name+'_geometry_2.csv'
file_structure_path = 'D:/TexGen_table/'+name+'/'+name+'_structure_2.csv'

with open(file_structure_path, 'wb') as f2:
    
    #geometry
    angle_data = ['angle']
    yarnwidth_data = ['yarnwidth']
    yarnheight_data = ['yarnheight']
    space_data = ['space']
    
    #epoxy
    epoxy_e_data = ['epoxy_e']
    epoxy_v_data = ['epoxy_v']
    epoxy_CLTE_data = ['epoxy_CLTE']
    
    #fibre
    fibre_D_data = ['fibre_density (kg/m^3)']
    fibre_LD_data = ['fibre_lineardensity (kg/m)']
    fibre_A_data = ['fibre_area (mm^2)']
    fibre_diameter_data = ['fibre_diameter (mm)']
    fibre_num_data = ['fibre_num']
    fibre_e1_data = ['fibre_e1']
    fibre_e2_data = ['fibre_e2']
    fibre_e3_data = ['fibre_e3']
    fibre_g12_data = ['fibre_g12']
    fibre_g23_data = ['fibre_g23']
    fibre_g13_data = ['fibre_g13']
    fibre_v1_data = ['fibre_v1']
    fibre_v2_data = ['fibre_v2']
    fibre_v3_data = ['fibre_v3']
    fibre_CLTE1_data = ['fibre_CLTE1']
    fibre_CLTE2_data = ['fibre_CLTE2']
    fibre_CLTE3_data = ['fibre_CLTE3']
     
    for step in range(301, 501):
# Create a 2D weave textile
        YarnWidth = uniform(0.2, 1)
        YarnHeight = uniform(0.1, 0.4)
        nwarp = 5   #Number of weft yarns in the unit cell
        nweft = 5   #Number of warp yarns in the unit cell
    
        a = (step-1)//100
        rotation = a*15
        angle = rotation*pi/180
        s = YarnWidth+(uniform(0.5, 1)*(YarnWidth))+(sin(angle)*YarnWidth)
        t = YarnHeight*2      #Thickness of the fabric (sum of two yarn heights)
        ref = False  #Refine model (True/False)
        
        
        modelname = 'Model_'+name+'_'+str(rotation)+'_'+str(step)
        inputname = name+'_'+str(rotation)+'_'+str(step)
    
# Create a 2D weave textile
        weave = CShearedTextileWeave2D(nwarp, nweft, s, t, angle, ref)

        weave.SetGapSize(0)
    
# Set the yarn section
        weave.SetYarnWidths(YarnWidth)
        weave.SetYarnHeights(YarnHeight)

# Set the weave pattern     
    
        structure = []
        for x in range(0, nweft):
            for y in range(0, nwarp):
                a = randint(0,1)
                if a == 1:                 # weft on top
                    weave.SwapPosition(x, y)
                    structure.append(1)
                elif a == 0:               # warp on top
                    structure.append(0)
                    
        writer2 = csv.writer(f2)  
        writer2.writerow(structure)
 
    # set matrix properties 
        epoxy_e = 20000
        epoxy_v = 0.4
        epoxy_CLTE = 0.00006

        weave.SetMatrixYoungsModulus(epoxy_e, 'MPa')
        weave.SetMatrixPoissonsRatio(epoxy_v)
        weave.SetMatrixAlpha(epoxy_CLTE)
        
    # set yarns properties        
        fibre_D = 2550  # kg/m^3
        FibreSectionArea = pi * (YarnWidth/2) * (YarnHeight/2)  # mm^2
        FibreWrapVolume = FibreSectionArea*nweft*s # mm^3
        Wrapmass = fibre_D*FibreWrapVolume # 10^-9 kg
        
        fibre_LD = Wrapmass/(nweft * s * 1000000 ) # kg/m
        fibre_A = FibreSectionArea*0.8
        fibre_diameter = 0.017  # mm
        fibre_num = 3000
        fibre_e1 = 72000
        fibre_e2 = 72000
        fibre_e3 = 72000
        fibre_g12 = 30000
        fibre_g23 = 30000
        fibre_g13 = 30000
        fibre_v1 = 0.2
        fibre_v2 = 0.2
        fibre_v3 = 0.2
        fibre_CLTE1 = 0.000005
        fibre_CLTE2 = 0.000005
        fibre_CLTE3 = 0.000005

        Yarn = weave.GetYarns()
        for i in range(len( Yarn )):
            Yarn[i].SetYarnLinearDensity(fibre_LD, 'kg/m')
            Yarn[i].SetFibreDensity(fibre_D, 'kg/m^3')
            Yarn[i].SetFibreArea(fibre_A, 'mm^2')
            Yarn[i].SetFibreDiameter(fibre_diameter, 'mm')
            Yarn[i].SetFibresPerYarn(fibre_num)
            Yarn[i].SetYoungsModulusX(fibre_e1, 'MPa')
            Yarn[i].SetYoungsModulusY(fibre_e2, 'MPa')
            Yarn[i].SetYoungsModulusZ(fibre_e3, 'MPa')
            Yarn[i].SetShearModulusXY(fibre_g12, 'MPa')
            Yarn[i].SetShearModulusYZ(fibre_g23, 'MPa')
            Yarn[i].SetShearModulusXZ(fibre_g13, 'MPa')
            Yarn[i].SetPoissonsRatioX(fibre_v1)
            Yarn[i].SetPoissonsRatioY(fibre_v2)
            Yarn[i].SetPoissonsRatioZ(fibre_v3)
            Yarn[i].SetAlphaX(fibre_CLTE1)
            Yarn[i].SetAlphaY(fibre_CLTE2)
            Yarn[i].SetAlphaZ(fibre_CLTE3)
            Yarn[i].SetResolution(50)
        #Yarn[i].AddRepeat( XYZ(1, 1, 1))
        #Yarn.AssignSection(CYarnSectionConstant(CSectionEllipse(yarnWidthX, yarnHeightX)))

    #min = XYZ(0, 0, -0.2*t)
    #max = XYZ(nwarp*s, nweft*s, 1.2*t)    
    #Domain = CDomainPlanes(min, max)
    #weave.AssignDomain(Domain)

        weave.AssignDefaultDomain(True)
    
# Add the textile
        AddTextile(modelname, weave)
    
        SaveToXML(modelname+'.tg3', modelname, OUTPUT_STANDARD)


        NumXVoxels = 50
        NumYVoxels = 50
        NumZVoxels = 50
# Create a voxel mesh object and then save the mesh
        mesh = CRectangularVoxelMesh()
# SaveVoxelMesh Parameters: Textile name, Filename, Number X voxels, Number Y voxels, Number Z voxels...
# Output matrix (true/false), Output yarns (true/false), Boundaries untied (true - z untied, false - all tied)...
# Element type ( 0 - C3D8R, 1 - C3D8 )
    #inputname = 'Epoxy-3_E-glass_'+ str(step)
        mesh.SaveVoxelMesh(weave, inputname+'.inp', NumXVoxels, NumYVoxels, NumZVoxels, True, True, False, 0)


        angle_data.append(rotation)
        yarnwidth_data.append(YarnWidth)
        yarnheight_data.append(YarnHeight)
        space_data.append(s)
        
        epoxy_e_data.append(epoxy_e)
        epoxy_v_data.append(epoxy_v)
        epoxy_CLTE_data.append(epoxy_CLTE)
        
        fibre_D_data.append(fibre_D)
        fibre_LD_data.append(fibre_LD)
        fibre_A_data.append(fibre_A)
        fibre_diameter_data.append(fibre_diameter)
        fibre_num_data.append(fibre_num)
        fibre_e1_data.append(fibre_e1)
        fibre_e2_data.append(fibre_e2)
        fibre_e3_data.append(fibre_e3)
        fibre_g12_data.append(fibre_g12)
        fibre_g23_data.append(fibre_g23)
        fibre_g13_data.append(fibre_g13)
        fibre_v1_data.append(fibre_v1)
        fibre_v2_data.append(fibre_v2)
        fibre_v3_data.append(fibre_v3)
        fibre_CLTE1_data.append(fibre_CLTE1)
        fibre_CLTE2_data.append(fibre_CLTE2)
        fibre_CLTE3_data.append(fibre_CLTE3)
        
        
        
        
        

    with open(file_geometry_path, 'wb') as f1:
        writer1 = csv.writer(f1)
        writer1.writerow(angle_data)
        writer1.writerow(yarnwidth_data)
        writer1.writerow(yarnheight_data)
        writer1.writerow(space_data)
        
        writer1.writerow(epoxy_e_data)
        writer1.writerow(epoxy_v_data)
        writer1.writerow(epoxy_CLTE_data)
        
        writer1.writerow(fibre_D_data)
        writer1.writerow(fibre_LD_data)
        writer1.writerow(fibre_A_data)
        writer1.writerow(fibre_diameter_data)
        writer1.writerow(fibre_num_data)
        writer1.writerow(fibre_e1_data)
        writer1.writerow(fibre_e2_data)
        writer1.writerow(fibre_e3_data)
        writer1.writerow(fibre_g12_data)
        writer1.writerow(fibre_g23_data)
        writer1.writerow(fibre_g13_data)
        writer1.writerow(fibre_v1_data)
        writer1.writerow(fibre_v2_data)
        writer1.writerow(fibre_v3_data)
        writer1.writerow(fibre_CLTE1_data)
        writer1.writerow(fibre_CLTE2_data)
        writer1.writerow(fibre_CLTE3_data)
        

#with open(file_structure_path, 'w') as f2:
    #writer2 = csv.writer(f2)
    #writer2.writerow(position)    
#data = pd.read_csv(file_path, header=None)
#data = data.values
#data = list(map(list, zip(*data)))
#data = pd.DataFrame(data)
#data.to_csv(file_path, header=None, index=False) 