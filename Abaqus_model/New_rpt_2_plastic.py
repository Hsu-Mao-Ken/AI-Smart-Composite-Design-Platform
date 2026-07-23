#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from abaqus import *
from odbAccess import *
from abaqusConstants import *
from odbMaterial import *
from odbSection import *
import math
import visualization
import odbAccess

modelName = 'Plastic_CEL-400_E-glass_0_1'
fileName = modelName + '.odb'
resultODB = visualization.openOdb(path=fileName, readOnly=True)
Plasticstep = resultODB.steps.values()[0]

Parameters = {'RF_Time':[]}


MaterialProperties = []
Time = []
for i in range(len(resultODB.steps[Plasticstep.name].frames)):
    frameRFx = resultODB.steps[Plasticstep.name].frames[i]
    Time.append(frameRFx.frameValue)
    RFx = frameRFx.fieldOutputs['RF'].values[0].data[0]
    Parameters['RF_Time'].append(RFx)


reportFilename = modelName + '.rpt'
fOutputReport = open(reportFilename, 'w')

RF_data = []
Time_data = []
for i in range(len(Time)):
    Time_string = [Time[i]]
    Properties_string = [Parameters['RF_Time'][i]]
    
    Time_data.append(''.join(map(str, Time_string)))
    RF_data.append(''.join(map(str, Properties_string)))
    
    fOutputReport.write(Time_data[i])
    fOutputReport.write('\n')
    fOutputReport.write(RF_data[i])
    fOutputReport.write('\n')
    
    
fOutputReport.close()



#getInputs(fields=fieldsMaterialProperties,
        #label='',
        #dialogTitle='Effective material properties obtained from the unit cell analysis',)

