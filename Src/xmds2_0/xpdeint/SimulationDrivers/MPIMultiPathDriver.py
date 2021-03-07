#!/usr/bin/env python




##################################################
## DEPENDENCIES
import sys
import os
import os.path
try:
    import builtins as builtin
except ImportError:
    import builtins as builtin
from os.path import getmtime, exists
import time
import types
from Cheetah.Version import MinCompatibleVersion as RequiredCheetahVersion
from Cheetah.Version import MinCompatibleVersionTuple as RequiredCheetahVersionTuple
from Cheetah.Template import Template
from Cheetah.DummyTransaction import *
from Cheetah.NameMapper import NotFound, valueForName, valueFromSearchList, valueFromFrameOrSearchList
from Cheetah.CacheRegion import CacheRegion
import Cheetah.Filters as Filters
import Cheetah.ErrorCatchers as ErrorCatchers
from xpdeint.SimulationDrivers._MPIMultiPathDriver import _MPIMultiPathDriver

##################################################
## MODULE CONSTANTS
VFFSL=valueFromFrameOrSearchList
VFSL=valueFromSearchList
VFN=valueForName
currentTime=time.time
__CHEETAH_version__ = '2.4.4'
__CHEETAH_versionTuple__ = (2, 4, 4, 'development', 0)
__CHEETAH_genTime__ = 1484975072.680343
__CHEETAH_genTimestamp__ = 'Sat Jan 21 16:04:32 2017'
__CHEETAH_src__ = '/home/mattias/xmds-2.2.3/admin/staging/xmds-2.2.3/xpdeint/SimulationDrivers/MPIMultiPathDriver.tmpl'
__CHEETAH_srcLastModified__ = 'Mon Nov 18 19:21:08 2013'
__CHEETAH_docstring__ = 'Autogenerated by Cheetah: The Python-Powered Template Engine'

if __CHEETAH_versionTuple__ < RequiredCheetahVersionTuple:
    raise AssertionError(
      'This template was compiled with Cheetah version'
      ' %s. Templates compiled before version %s must be recompiled.'%(
         __CHEETAH_version__, RequiredCheetahVersion))

##################################################
## CLASSES

class MPIMultiPathDriver(_MPIMultiPathDriver):

    ##################################################
    ## CHEETAH GENERATED METHODS


    def __init__(self, *args, **KWs):

        super(MPIMultiPathDriver, self).__init__(*args, **KWs)
        if not self._CHEETAH__instanceInitialized:
            cheetahKWArgs = {}
            allowedKWs = 'searchList namespaces filter filtersLib errorCatcher'.split()
            for k,v in list(KWs.items()):
                if k in allowedKWs: cheetahKWArgs[k] = v
            self._initCheetahInstance(**cheetahKWArgs)
        

    def description(self, **KWS):



        ## Generated from @def description: MPI Multipath Simulation Driver at line 26, col 1.
        trans = KWS.get("trans")
        if (not trans and not self._CHEETAH__isBuffering and not callable(self.transaction)):
            trans = self.transaction # is None unless self.awake() was called
        if not trans:
            trans = DummyTransaction()
            _dummyTrans = True
        else: _dummyTrans = False
        write = trans.response().write
        SL = self._CHEETAH__searchList
        _filter = self._CHEETAH__currentFilter
        
        ########################################
        ## START - generated method body
        
        write('''MPI Multipath Simulation Driver''')
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def mainRoutine(self, **KWS):



        ## CHEETAH: generated from @def mainRoutine at line 31, col 1.
        trans = KWS.get("trans")
        if (not trans and not self._CHEETAH__isBuffering and not callable(self.transaction)):
            trans = self.transaction # is None unless self.awake() was called
        if not trans:
            trans = DummyTransaction()
            _dummyTrans = True
        else: _dummyTrans = False
        write = trans.response().write
        SL = self._CHEETAH__searchList
        _filter = self._CHEETAH__currentFilter
        
        ########################################
        ## START - generated method body
        
        # 
        write('''int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
  
  ''')
        _v = VFFSL(SL,"mainRoutineInnerContent",True) # u'${mainRoutineInnerContent, autoIndent=True}' on line 39, col 3
        if _v is not None: write(_filter(_v, autoIndent=True, rawExpr='${mainRoutineInnerContent, autoIndent=True}')) # from line 39, col 3.
        write('''  
  MPI_Finalize();
  
  return 0;
}
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def segment0End(self, **KWS):



        ## CHEETAH: generated from @def segment0End at line 48, col 1.
        trans = KWS.get("trans")
        if (not trans and not self._CHEETAH__isBuffering and not callable(self.transaction)):
            trans = self.transaction # is None unless self.awake() was called
        if not trans:
            trans = DummyTransaction()
            _dummyTrans = True
        else: _dummyTrans = False
        write = trans.response().write
        SL = self._CHEETAH__searchList
        _filter = self._CHEETAH__currentFilter
        
        ########################################
        ## START - generated method body
        
        # 
        _v = VFFSL(SL,"segment0ReduceBlock",True) # u'${segment0ReduceBlock}' on line 50, col 1
        if _v is not None: write(_filter(_v, rawExpr='${segment0ReduceBlock}')) # from line 50, col 1.
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def topLevelSegmentFunctionImplementation(self, **KWS):



        ## CHEETAH: generated from @def topLevelSegmentFunctionImplementation at line 54, col 1.
        trans = KWS.get("trans")
        if (not trans and not self._CHEETAH__isBuffering and not callable(self.transaction)):
            trans = self.transaction # is None unless self.awake() was called
        if not trans:
            trans = DummyTransaction()
            _dummyTrans = True
        else: _dummyTrans = False
        write = trans.response().write
        SL = self._CHEETAH__searchList
        _filter = self._CHEETAH__currentFilter
        
        ########################################
        ## START - generated method body
        
        # 
        write('''void _segment0()
{
''')
        #  And now insert the code for the features that apply in the top level sequence
        featureOrdering = ['ErrorCheck', 'Stochastic']
        dict = {'extraIndent': 0}
        write('''  ''')
        _v = VFFSL(SL,"insertCodeForFeatures",False)('topLevelSequenceBegin', featureOrdering, dict) # u"${insertCodeForFeatures('topLevelSequenceBegin', featureOrdering, dict), autoIndent=True}" on line 61, col 3
        if _v is not None: write(_filter(_v, autoIndent=True, rawExpr="${insertCodeForFeatures('topLevelSequenceBegin', featureOrdering, dict), autoIndent=True}")) # from line 61, col 3.
        extraIndent = dict['extraIndent']
        write('''  
  ''')
        _v = VFFSL(SL,"topLevelSegmentPathLoop",True) # u'${topLevelSegmentPathLoop, autoIndent=True, extraIndent=extraIndent}' on line 64, col 3
        if _v is not None: write(_filter(_v, autoIndent=True, extraIndent=extraIndent, rawExpr='${topLevelSegmentPathLoop, autoIndent=True, extraIndent=extraIndent}')) # from line 64, col 3.
        write('''  
  ''')
        _v = VFFSL(SL,"insertCodeForFeaturesInReverseOrder",False)('topLevelSequenceEnd', featureOrdering, dict) # u"${insertCodeForFeaturesInReverseOrder('topLevelSequenceEnd', featureOrdering, dict), autoIndent=True}" on line 66, col 3
        if _v is not None: write(_filter(_v, autoIndent=True, rawExpr="${insertCodeForFeaturesInReverseOrder('topLevelSequenceEnd', featureOrdering, dict), autoIndent=True}")) # from line 66, col 3.
        write('''  
  ''')
        _v = VFFSL(SL,"segment0End",True) # u'${segment0End, autoIndent=True}' on line 68, col 3
        if _v is not None: write(_filter(_v, autoIndent=True, rawExpr='${segment0End, autoIndent=True}')) # from line 68, col 3.
        write('''}
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def segment0ReduceBlock(self, **KWS):



        ## CHEETAH: generated from @def segment0ReduceBlock at line 73, col 1.
        trans = KWS.get("trans")
        if (not trans and not self._CHEETAH__isBuffering and not callable(self.transaction)):
            trans = self.transaction # is None unless self.awake() was called
        if not trans:
            trans = DummyTransaction()
            _dummyTrans = True
        else: _dummyTrans = False
        write = trans.response().write
        SL = self._CHEETAH__searchList
        _filter = self._CHEETAH__currentFilter
        
        ########################################
        ## START - generated method body
        
        # 
        for mg in VFFSL(SL,"momentGroups",True): # generated from line 75, col 3
            for vector in mg.outputField.managedVectors: # generated from line 76, col 5
                arrayNames = [''.join(['_',str(VFFSL(SL,"vector.id",True))])]
                VFN(VFFSL(SL,"arrayNames",True),"extend",False)(VFFSL(SL,"vector.aliases",True))
                for arrayName in arrayNames: # generated from line 79, col 7
                    write('''  
if (_rank == 0)
  MPI_Reduce(MPI_IN_PLACE, ''')
                    _v = VFFSL(SL,"arrayName",True) # u'$arrayName' on line 82, col 28
                    if _v is not None: write(_filter(_v, rawExpr='$arrayName')) # from line 82, col 28.
                    write(''', ''')
                    _v = VFN(VFFSL(SL,"vector",True),"sizeInBasisInReals",False)(mg.outputBasis) # u'${vector.sizeInBasisInReals(mg.outputBasis)}' on line 82, col 40
                    if _v is not None: write(_filter(_v, rawExpr='${vector.sizeInBasisInReals(mg.outputBasis)}')) # from line 82, col 40.
                    write(''',
             MPI_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
else
  MPI_Reduce(''')
                    _v = VFFSL(SL,"arrayName",True) # u'$arrayName' on line 85, col 14
                    if _v is not None: write(_filter(_v, rawExpr='$arrayName')) # from line 85, col 14.
                    write(''', NULL, ''')
                    _v = VFN(VFFSL(SL,"vector",True),"sizeInBasisInReals",False)(mg.outputBasis) # u'${vector.sizeInBasisInReals(mg.outputBasis)}' on line 85, col 32
                    if _v is not None: write(_filter(_v, rawExpr='${vector.sizeInBasisInReals(mg.outputBasis)}')) # from line 85, col 32.
                    write(''',
             MPI_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def writeOutBegin(self, dict, **KWS):



        ## CHEETAH: generated from @def writeOutBegin($dict) at line 93, col 1.
        trans = KWS.get("trans")
        if (not trans and not self._CHEETAH__isBuffering and not callable(self.transaction)):
            trans = self.transaction # is None unless self.awake() was called
        if not trans:
            trans = DummyTransaction()
            _dummyTrans = True
        else: _dummyTrans = False
        write = trans.response().write
        SL = self._CHEETAH__searchList
        _filter = self._CHEETAH__currentFilter
        
        ########################################
        ## START - generated method body
        
        # 
        write("""// If we aren't rank 0, then we don't want to write anything.
if (_rank != 0)
  return;
""")
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def writeBody(self, **KWS):



        ## CHEETAH: main method generated for this template
        trans = KWS.get("trans")
        if (not trans and not self._CHEETAH__isBuffering and not callable(self.transaction)):
            trans = self.transaction # is None unless self.awake() was called
        if not trans:
            trans = DummyTransaction()
            _dummyTrans = True
        else: _dummyTrans = False
        write = trans.response().write
        SL = self._CHEETAH__searchList
        _filter = self._CHEETAH__currentFilter
        
        ########################################
        ## START - generated method body
        
        # 
        # MPIMultiPathDriver.tmpl
        # 
        # Created by Graham Dennis on 2008-02-25
        # Modified by Liam Madge on 2013-09-11
        # Modified by Gregory Bogomiagkov on 2013-09-12
        # 
        # Copyright (c) 2008-2012, Graham Dennis
        # 
        # This program is free software: you can redistribute it and/or modify
        # it under the terms of the GNU General Public License as published by
        # the Free Software Foundation, either version 2 of the License, or
        # (at your option) any later version.
        # 
        # This program is distributed in the hope that it will be useful,
        # but WITHOUT ANY WARRANTY; without even the implied warranty of
        # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        # GNU General Public License for more details.
        # 
        # You should have received a copy of the GNU General Public License
        # along with this program.  If not, see <http://www.gnu.org/licenses/>.
        # 
        write('''






''')
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        
    ##################################################
    ## CHEETAH GENERATED ATTRIBUTES


    _CHEETAH__instanceInitialized = False

    _CHEETAH_version = __CHEETAH_version__

    _CHEETAH_versionTuple = __CHEETAH_versionTuple__

    _CHEETAH_genTime = __CHEETAH_genTime__

    _CHEETAH_genTimestamp = __CHEETAH_genTimestamp__

    _CHEETAH_src = __CHEETAH_src__

    _CHEETAH_srcLastModified = __CHEETAH_srcLastModified__

    pathLoopStart = '_rank'

    pathLoopStep = '_size'

    _mainCheetahMethod_for_MPIMultiPathDriver= 'writeBody'

## END CLASS DEFINITION

if not hasattr(MPIMultiPathDriver, '_initCheetahAttributes'):
    templateAPIClass = getattr(MPIMultiPathDriver, '_CHEETAH_templateClass', Template)
    templateAPIClass._addCheetahPlumbingCodeToClass(MPIMultiPathDriver)


# CHEETAH was developed by Tavis Rudd and Mike Orr
# with code, advice and input from many other volunteers.
# For more information visit http://www.CheetahTemplate.org/

##################################################
## if run from command line:
if __name__ == '__main__':
    from Cheetah.TemplateCmdLineIface import CmdLineIface
    CmdLineIface(templateObj=MPIMultiPathDriver()).run()

