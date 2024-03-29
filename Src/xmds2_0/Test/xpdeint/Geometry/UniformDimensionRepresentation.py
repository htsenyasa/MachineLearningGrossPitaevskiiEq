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
from xpdeint.Geometry._UniformDimensionRepresentation import _UniformDimensionRepresentation

##################################################
## MODULE CONSTANTS
VFFSL=valueFromFrameOrSearchList
VFSL=valueFromSearchList
VFN=valueForName
currentTime=time.time
__CHEETAH_version__ = '2.4.4'
__CHEETAH_versionTuple__ = (2, 4, 4, 'development', 0)
__CHEETAH_genTime__ = 1484975071.810958
__CHEETAH_genTimestamp__ = 'Sat Jan 21 16:04:31 2017'
__CHEETAH_src__ = '/home/mattias/xmds-2.2.3/admin/staging/xmds-2.2.3/xpdeint/Geometry/UniformDimensionRepresentation.tmpl'
__CHEETAH_srcLastModified__ = 'Fri Jul 13 16:21:46 2012'
__CHEETAH_docstring__ = 'Autogenerated by Cheetah: The Python-Powered Template Engine'

if __CHEETAH_versionTuple__ < RequiredCheetahVersionTuple:
    raise AssertionError(
      'This template was compiled with Cheetah version'
      ' %s. Templates compiled before version %s must be recompiled.'%(
         __CHEETAH_version__, RequiredCheetahVersion))

##################################################
## CLASSES

class UniformDimensionRepresentation(_UniformDimensionRepresentation):

    ##################################################
    ## CHEETAH GENERATED METHODS


    def __init__(self, *args, **KWs):

        super(UniformDimensionRepresentation, self).__init__(*args, **KWs)
        if not self._CHEETAH__instanceInitialized:
            cheetahKWArgs = {}
            allowedKWs = 'searchList namespaces filter filtersLib errorCatcher'.split()
            for k,v in list(KWs.items()):
                if k in allowedKWs: cheetahKWArgs[k] = v
            self._initCheetahInstance(**cheetahKWArgs)
        

    def defines(self, **KWS):



        ## CHEETAH: generated from @def defines at line 24, col 1.
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
        _v = super(UniformDimensionRepresentation, self).defines()
        if _v is not None: write(_filter(_v))
        if VFFSL(SL,"silent",True): # generated from line 27, col 3
            return _dummyTrans and trans.response().getvalue() or ""
        # 
        write('''#define ''')
        _v = VFFSL(SL,"minimum",True) # u'${minimum}' on line 31, col 9
        if _v is not None: write(_filter(_v, rawExpr='${minimum}')) # from line 31, col 9.
        write('''     ((''')
        _v = VFFSL(SL,"type",True) # u'$type' on line 31, col 26
        if _v is not None: write(_filter(_v, rawExpr='$type')) # from line 31, col 26.
        write(''')''')
        _v = VFFSL(SL,"_minimum",True) # u'${_minimum}' on line 31, col 32
        if _v is not None: write(_filter(_v, rawExpr='${_minimum}')) # from line 31, col 32.
        write(''')
#define ''')
        _v = VFFSL(SL,"maximum",True) # u'${maximum}' on line 32, col 9
        if _v is not None: write(_filter(_v, rawExpr='${maximum}')) # from line 32, col 9.
        write('''     ((''')
        _v = VFFSL(SL,"type",True) # u'$type' on line 32, col 26
        if _v is not None: write(_filter(_v, rawExpr='$type')) # from line 32, col 26.
        write(''')''')
        _v = VFFSL(SL,"_maximum",True) # u'${_maximum}' on line 32, col 32
        if _v is not None: write(_filter(_v, rawExpr='${_maximum}')) # from line 32, col 32.
        write(''')
#define ''')
        _v = VFFSL(SL,"stepSize",True) # u'${stepSize}' on line 33, col 9
        if _v is not None: write(_filter(_v, rawExpr='${stepSize}')) # from line 33, col 9.
        write('''        ((''')
        _v = VFFSL(SL,"type",True) # u'$type' on line 33, col 30
        if _v is not None: write(_filter(_v, rawExpr='$type')) # from line 33, col 30.
        write(''')''')
        _v = VFFSL(SL,"stepSizeString",True) # u'${stepSizeString}' on line 33, col 36
        if _v is not None: write(_filter(_v, rawExpr='${stepSizeString}')) # from line 33, col 36.
        write(''')
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def openLoopAscending(self, **KWS):



        ## CHEETAH: generated from @def openLoopAscending at line 37, col 1.
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
        write('''#define ''')
        _v = VFFSL(SL,"name",True) # u'${name}' on line 39, col 9
        if _v is not None: write(_filter(_v, rawExpr='${name}')) # from line 39, col 9.
        write(''' ''')
        _v = VFFSL(SL,"arrayName",True) # u'${arrayName}' on line 39, col 17
        if _v is not None: write(_filter(_v, rawExpr='${arrayName}')) # from line 39, col 17.
        write('''[''')
        _v = VFFSL(SL,"loopIndex",True) # u'${loopIndex}' on line 39, col 30
        if _v is not None: write(_filter(_v, rawExpr='${loopIndex}')) # from line 39, col 30.
        write(''' + ''')
        _v = VFFSL(SL,"localOffset",True) # u'${localOffset}' on line 39, col 45
        if _v is not None: write(_filter(_v, rawExpr='${localOffset}')) # from line 39, col 45.
        write(''']
#define d''')
        _v = VFFSL(SL,"name",True) # u'${name}' on line 40, col 10
        if _v is not None: write(_filter(_v, rawExpr='${name}')) # from line 40, col 10.
        write(''' (''')
        _v = VFFSL(SL,"stepSize",True) # u'${stepSize}' on line 40, col 19
        if _v is not None: write(_filter(_v, rawExpr='${stepSize}')) # from line 40, col 19.
        write(''' * (''')
        _v = VFFSL(SL,"volumePrefactor",True) # u'${volumePrefactor}' on line 40, col 34
        if _v is not None: write(_filter(_v, rawExpr='${volumePrefactor}')) # from line 40, col 34.
        write('''))

for (long ''')
        _v = VFFSL(SL,"loopIndex",True) # u'${loopIndex}' on line 42, col 11
        if _v is not None: write(_filter(_v, rawExpr='${loopIndex}')) # from line 42, col 11.
        write(''' = 0; ''')
        _v = VFFSL(SL,"loopIndex",True) # u'${loopIndex}' on line 42, col 29
        if _v is not None: write(_filter(_v, rawExpr='${loopIndex}')) # from line 42, col 29.
        write(''' < ''')
        _v = VFFSL(SL,"localLattice",True) # u'${localLattice}' on line 42, col 44
        if _v is not None: write(_filter(_v, rawExpr='${localLattice}')) # from line 42, col 44.
        write('''; ''')
        _v = VFFSL(SL,"loopIndex",True) # u'${loopIndex}' on line 42, col 61
        if _v is not None: write(_filter(_v, rawExpr='${loopIndex}')) # from line 42, col 61.
        write('''++) {
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def closeLoopAscending(self, **KWS):



        ## CHEETAH: generated from @def closeLoopAscending at line 46, col 1.
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
        write('''}
#undef ''')
        _v = VFFSL(SL,"name",True) # u'${name}' on line 49, col 8
        if _v is not None: write(_filter(_v, rawExpr='${name}')) # from line 49, col 8.
        write('''
#undef d''')
        _v = VFFSL(SL,"name",True) # u'${name}' on line 50, col 9
        if _v is not None: write(_filter(_v, rawExpr='${name}')) # from line 50, col 9.
        write('''
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def openLoopDescending(self, **KWS):



        ## CHEETAH: generated from @def openLoopDescending at line 54, col 1.
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
        write('''#define ''')
        _v = VFFSL(SL,"name",True) # u'${name}' on line 56, col 9
        if _v is not None: write(_filter(_v, rawExpr='${name}')) # from line 56, col 9.
        write(''' ''')
        _v = VFFSL(SL,"arrayName",True) # u'${arrayName}' on line 56, col 17
        if _v is not None: write(_filter(_v, rawExpr='${arrayName}')) # from line 56, col 17.
        write('''[''')
        _v = VFFSL(SL,"loopIndex",True) # u'${loopIndex}' on line 56, col 30
        if _v is not None: write(_filter(_v, rawExpr='${loopIndex}')) # from line 56, col 30.
        write(''' + ''')
        _v = VFFSL(SL,"localOffset",True) # u'${localOffset}' on line 56, col 45
        if _v is not None: write(_filter(_v, rawExpr='${localOffset}')) # from line 56, col 45.
        write(''']
#define d''')
        _v = VFFSL(SL,"name",True) # u'${name}' on line 57, col 10
        if _v is not None: write(_filter(_v, rawExpr='${name}')) # from line 57, col 10.
        write(''' (''')
        _v = VFFSL(SL,"stepSize",True) # u'${stepSize}' on line 57, col 19
        if _v is not None: write(_filter(_v, rawExpr='${stepSize}')) # from line 57, col 19.
        write(''' * (''')
        _v = VFFSL(SL,"volumePrefactor",True) # u'${volumePrefactor}' on line 57, col 34
        if _v is not None: write(_filter(_v, rawExpr='${volumePrefactor}')) # from line 57, col 34.
        write('''))

for (long ''')
        _v = VFFSL(SL,"loopIndex",True) # u'${loopIndex}' on line 59, col 11
        if _v is not None: write(_filter(_v, rawExpr='${loopIndex}')) # from line 59, col 11.
        write(''' = ''')
        _v = VFFSL(SL,"localLattice",True) # u'${localLattice}' on line 59, col 26
        if _v is not None: write(_filter(_v, rawExpr='${localLattice}')) # from line 59, col 26.
        write('''-1; ''')
        _v = VFFSL(SL,"loopIndex",True) # u'${loopIndex}' on line 59, col 45
        if _v is not None: write(_filter(_v, rawExpr='${loopIndex}')) # from line 59, col 45.
        write(''' >= 0; ''')
        _v = VFFSL(SL,"loopIndex",True) # u'${loopIndex}' on line 59, col 64
        if _v is not None: write(_filter(_v, rawExpr='${loopIndex}')) # from line 59, col 64.
        write('''--) {
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def closeLoopDescending(self, **KWS):



        ## CHEETAH: generated from @def closeLoopDescending at line 63, col 1.
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
        write('''}
#undef ''')
        _v = VFFSL(SL,"name",True) # u'${name}' on line 66, col 8
        if _v is not None: write(_filter(_v, rawExpr='${name}')) # from line 66, col 8.
        write('''
#undef d''')
        _v = VFFSL(SL,"name",True) # u'${name}' on line 67, col 9
        if _v is not None: write(_filter(_v, rawExpr='${name}')) # from line 67, col 9.
        write('''
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def localIndexFromIndexForDimensionRep(self, dimRep, **KWS):



        ## CHEETAH: generated from @def localIndexFromIndexForDimensionRep($dimRep) at line 71, col 1.
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
        
        if VFFSL(SL,"dimRep.runtimeLattice",True) == VFFSL(SL,"runtimeLattice",True) or VFFSL(SL,"dimRep.reductionMethod",True) == VFFSL(SL,"ReductionMethod.fixedStep",True): # generated from line 72, col 3
            _v = VFFSL(SL,"dimRep.loopIndex",True) # u'${dimRep.loopIndex}' on line 73, col 1
            if _v is not None: write(_filter(_v, rawExpr='${dimRep.loopIndex}')) # from line 73, col 1.
            write(''' + ''')
            _v = VFFSL(SL,"dimRep.localOffset",True) # u'${dimRep.localOffset}' on line 73, col 23
            if _v is not None: write(_filter(_v, rawExpr='${dimRep.localOffset}')) # from line 73, col 23.
            write(''' - ''')
            _v = VFFSL(SL,"localOffset",True) # u'${localOffset}' on line 73, col 47
            if _v is not None: write(_filter(_v, rawExpr='${localOffset}')) # from line 73, col 47.
        elif VFFSL(SL,"dimRep.reductionMethod",True) == VFFSL(SL,"ReductionMethod.fixedRange",True): # generated from line 74, col 3
            #  We are using a fixed-range reduction method.
            # 
            write('''(''')
            _v = VFFSL(SL,"dimRep.loopIndex",True) # u'${dimRep.loopIndex}' on line 77, col 2
            if _v is not None: write(_filter(_v, rawExpr='${dimRep.loopIndex}')) # from line 77, col 2.
            write(''' + ''')
            _v = VFFSL(SL,"dimRep.localOffset",True) # u'${dimRep.localOffset}' on line 77, col 24
            if _v is not None: write(_filter(_v, rawExpr='${dimRep.localOffset}')) # from line 77, col 24.
            write(''') * (''')
            _v = VFFSL(SL,"globalLattice",True) # u'${globalLattice}' on line 77, col 50
            if _v is not None: write(_filter(_v, rawExpr='${globalLattice}')) # from line 77, col 50.
            write('''/''')
            _v = VFFSL(SL,"dimRep.globalLattice",True) # u'${dimRep.globalLattice}' on line 77, col 67
            if _v is not None: write(_filter(_v, rawExpr='${dimRep.globalLattice}')) # from line 77, col 67.
            write(''') - ''')
            _v = VFFSL(SL,"localOffset",True) # u'${localOffset}' on line 77, col 94
            if _v is not None: write(_filter(_v, rawExpr='${localOffset}')) # from line 77, col 94.
        else: # generated from line 78, col 3
            assert False
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def strictlyAscendingGlobalIndex(self, **KWS):



        ## CHEETAH: generated from @def strictlyAscendingGlobalIndex at line 84, col 1.
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
        if not VFFSL(SL,"hasLocalOffset",True): # generated from line 86, col 3
            return VFFSL(SL,"loopIndex",True)
        else: # generated from line 88, col 3
            write('''lround((''')
            _v = VFFSL(SL,"name",True) # u'${name}' on line 89, col 9
            if _v is not None: write(_filter(_v, rawExpr='${name}')) # from line 89, col 9.
            write(''' - ''')
            _v = VFFSL(SL,"minimum",True) # u'${minimum}' on line 89, col 19
            if _v is not None: write(_filter(_v, rawExpr='${minimum}')) # from line 89, col 19.
            write(''')/''')
            _v = VFFSL(SL,"stepSize",True) # u'${stepSize}' on line 89, col 31
            if _v is not None: write(_filter(_v, rawExpr='${stepSize}')) # from line 89, col 31.
            write(''')''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def indexForSinglePointSample(self, **KWS):



        ## CHEETAH: generated from @def indexForSinglePointSample at line 94, col 1.
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
        #  Take the middle point, which is in the middle of the array
        _v = VFFSL(SL,"globalLattice",True) # u'${globalLattice}' on line 97, col 1
        if _v is not None: write(_filter(_v, rawExpr='${globalLattice}')) # from line 97, col 1.
        write('''/2''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def createCoordinateVariableForSinglePointSample(self, **KWS):



        ## CHEETAH: generated from @def createCoordinateVariableForSinglePointSample at line 101, col 1.
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
        _v = VFFSL(SL,"type",True) # u'${type}' on line 103, col 1
        if _v is not None: write(_filter(_v, rawExpr='${type}')) # from line 103, col 1.
        write(''' ''')
        _v = VFFSL(SL,"name",True) # u'${name}' on line 103, col 9
        if _v is not None: write(_filter(_v, rawExpr='${name}')) # from line 103, col 9.
        write(''' = ''')
        _v = VFFSL(SL,"arrayName",True) # u'${arrayName}' on line 103, col 19
        if _v is not None: write(_filter(_v, rawExpr='${arrayName}')) # from line 103, col 19.
        write('''[''')
        _v = VFFSL(SL,"globalLattice",True) # u'${globalLattice}' on line 103, col 32
        if _v is not None: write(_filter(_v, rawExpr='${globalLattice}')) # from line 103, col 32.
        write('''/2];
#define d''')
        _v = VFFSL(SL,"name",True) # u'${name}' on line 104, col 10
        if _v is not None: write(_filter(_v, rawExpr='${name}')) # from line 104, col 10.
        write(''' (''')
        _v = VFFSL(SL,"stepSize",True) # u'${stepSize}' on line 104, col 19
        if _v is not None: write(_filter(_v, rawExpr='${stepSize}')) # from line 104, col 19.
        write(''' * (''')
        _v = VFFSL(SL,"volumePrefactor",True) # u'${volumePrefactor}' on line 104, col 34
        if _v is not None: write(_filter(_v, rawExpr='${volumePrefactor}')) # from line 104, col 34.
        write('''))
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def initialiseArray(self, **KWS):



        ## CHEETAH: generated from @def initialiseArray at line 108, col 1.
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
        write('''for (long ''')
        _v = VFFSL(SL,"loopIndex",True) # u'${loopIndex}' on line 110, col 11
        if _v is not None: write(_filter(_v, rawExpr='${loopIndex}')) # from line 110, col 11.
        write(''' = 0; ''')
        _v = VFFSL(SL,"loopIndex",True) # u'${loopIndex}' on line 110, col 29
        if _v is not None: write(_filter(_v, rawExpr='${loopIndex}')) # from line 110, col 29.
        write(''' < ''')
        _v = VFFSL(SL,"globalLattice",True) # u'${globalLattice}' on line 110, col 44
        if _v is not None: write(_filter(_v, rawExpr='${globalLattice}')) # from line 110, col 44.
        write('''; ''')
        _v = VFFSL(SL,"loopIndex",True) # u'${loopIndex}' on line 110, col 62
        if _v is not None: write(_filter(_v, rawExpr='${loopIndex}')) # from line 110, col 62.
        write('''++)
  ''')
        _v = VFFSL(SL,"arrayName",True) # u'${arrayName}' on line 111, col 3
        if _v is not None: write(_filter(_v, rawExpr='${arrayName}')) # from line 111, col 3.
        write('''[''')
        _v = VFFSL(SL,"loopIndex",True) # u'${loopIndex}' on line 111, col 16
        if _v is not None: write(_filter(_v, rawExpr='${loopIndex}')) # from line 111, col 16.
        write('''] = ''')
        _v = VFFSL(SL,"minimum",True) # u'${minimum}' on line 111, col 32
        if _v is not None: write(_filter(_v, rawExpr='${minimum}')) # from line 111, col 32.
        write(''' + ''')
        _v = VFFSL(SL,"loopIndex",True) # u'${loopIndex}' on line 111, col 45
        if _v is not None: write(_filter(_v, rawExpr='${loopIndex}')) # from line 111, col 45.
        write('''*''')
        _v = VFFSL(SL,"stepSize",True) # u'${stepSize}' on line 111, col 58
        if _v is not None: write(_filter(_v, rawExpr='${stepSize}')) # from line 111, col 58.
        write(''';
''')
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
        # UniformDimensionRepresentation.tmpl
        # 
        # Created by Graham Dennis on 2008-07-31.
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

    _mainCheetahMethod_for_UniformDimensionRepresentation= 'writeBody'

## END CLASS DEFINITION

if not hasattr(UniformDimensionRepresentation, '_initCheetahAttributes'):
    templateAPIClass = getattr(UniformDimensionRepresentation, '_CHEETAH_templateClass', Template)
    templateAPIClass._addCheetahPlumbingCodeToClass(UniformDimensionRepresentation)


# CHEETAH was developed by Tavis Rudd and Mike Orr
# with code, advice and input from many other volunteers.
# For more information visit http://www.CheetahTemplate.org/

##################################################
## if run from command line:
if __name__ == '__main__':
    from Cheetah.TemplateCmdLineIface import CmdLineIface
    CmdLineIface(templateObj=UniformDimensionRepresentation()).run()


