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
from xpdeint.Geometry._NonUniformDimensionRepresentation import _NonUniformDimensionRepresentation

##################################################
## MODULE CONSTANTS
VFFSL=valueFromFrameOrSearchList
VFSL=valueFromSearchList
VFN=valueForName
currentTime=time.time
__CHEETAH_version__ = '2.4.4'
__CHEETAH_versionTuple__ = (2, 4, 4, 'development', 0)
__CHEETAH_genTime__ = 1484975071.779476
__CHEETAH_genTimestamp__ = 'Sat Jan 21 16:04:31 2017'
__CHEETAH_src__ = '/home/mattias/xmds-2.2.3/admin/staging/xmds-2.2.3/xpdeint/Geometry/NonUniformDimensionRepresentation.tmpl'
__CHEETAH_srcLastModified__ = 'Tue May 22 16:27:12 2012'
__CHEETAH_docstring__ = 'Autogenerated by Cheetah: The Python-Powered Template Engine'

if __CHEETAH_versionTuple__ < RequiredCheetahVersionTuple:
    raise AssertionError(
      'This template was compiled with Cheetah version'
      ' %s. Templates compiled before version %s must be recompiled.'%(
         __CHEETAH_version__, RequiredCheetahVersion))

##################################################
## CLASSES

class NonUniformDimensionRepresentation(_NonUniformDimensionRepresentation):

    ##################################################
    ## CHEETAH GENERATED METHODS


    def __init__(self, *args, **KWs):

        super(NonUniformDimensionRepresentation, self).__init__(*args, **KWs)
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
        if VFFSL(SL,"silent",True): # generated from line 26, col 3
            return _dummyTrans and trans.response().getvalue() or ""
        if VFFSL(SL,"runtimeLattice",True): # generated from line 29, col 3
            _v = super(NonUniformDimensionRepresentation, self).defines()
            if _v is not None: write(_filter(_v))
            write('''#define ''')
            _v = VFFSL(SL,"minimum",True) # u'${minimum}' on line 31, col 9
            if _v is not None: write(_filter(_v, rawExpr='${minimum}')) # from line 31, col 9.
            write('''     (''')
            _v = VFFSL(SL,"arrayName",True) # u'${arrayName}' on line 31, col 25
            if _v is not None: write(_filter(_v, rawExpr='${arrayName}')) # from line 31, col 25.
            write('''[0])
#define ''')
            _v = VFFSL(SL,"maximum",True) # u'${maximum}' on line 32, col 9
            if _v is not None: write(_filter(_v, rawExpr='${maximum}')) # from line 32, col 9.
            write('''     (''')
            _v = VFFSL(SL,"arrayName",True) # u'${arrayName}' on line 32, col 25
            if _v is not None: write(_filter(_v, rawExpr='${arrayName}')) # from line 32, col 25.
            write('''[''')
            _v = VFFSL(SL,"globalLattice",True) # u'${globalLattice}' on line 32, col 38
            if _v is not None: write(_filter(_v, rawExpr='${globalLattice}')) # from line 32, col 38.
            write('''-1])
''')
            if not VFFSL(SL,"stepSizeArray",True): # generated from line 33, col 5
                write('''#define ''')
                _v = VFFSL(SL,"stepSize",True) # u'${stepSize}' on line 34, col 9
                if _v is not None: write(_filter(_v, rawExpr='${stepSize}')) # from line 34, col 9.
                write('''        (''')
                _v = VFFSL(SL,"arrayName",True) # u'${arrayName}' on line 34, col 29
                if _v is not None: write(_filter(_v, rawExpr='${arrayName}')) # from line 34, col 29.
                write('''[''')
                _v = VFFSL(SL,"loopIndex",True) # u'${loopIndex}' on line 34, col 42
                if _v is not None: write(_filter(_v, rawExpr='${loopIndex}')) # from line 34, col 42.
                write('''+1]-''')
                _v = VFFSL(SL,"arrayName",True) # u'${arrayName}' on line 34, col 58
                if _v is not None: write(_filter(_v, rawExpr='${arrayName}')) # from line 34, col 58.
                write('''[''')
                _v = VFFSL(SL,"loopIndex",True) # u'${loopIndex}' on line 34, col 71
                if _v is not None: write(_filter(_v, rawExpr='${loopIndex}')) # from line 34, col 71.
                write('''])
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def globals(self, **KWS):



        ## CHEETAH: generated from @def globals at line 40, col 1.
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
        if VFFSL(SL,"silent",True): # generated from line 42, col 3
            return _dummyTrans and trans.response().getvalue() or ""
        # 
        _v = super(NonUniformDimensionRepresentation, self).globals()
        if _v is not None: write(_filter(_v))
        # 
        if VFFSL(SL,"runtimeLattice",True): # generated from line 48, col 3
            if VFFSL(SL,"stepSizeArray",True): # generated from line 49, col 5
                _v = VFFSL(SL,"type",True) # u'${type}' on line 50, col 1
                if _v is not None: write(_filter(_v, rawExpr='${type}')) # from line 50, col 1.
                write('''* ''')
                _v = VFFSL(SL,"stepSizeArrayName",True) # u'${stepSizeArrayName}' on line 50, col 10
                if _v is not None: write(_filter(_v, rawExpr='${stepSizeArrayName}')) # from line 50, col 10.
                write(''' = (''')
                _v = VFFSL(SL,"type",True) # u'${type}' on line 50, col 34
                if _v is not None: write(_filter(_v, rawExpr='${type}')) # from line 50, col 34.
                write('''*) xmds_malloc(sizeof(''')
                _v = VFFSL(SL,"type",True) # u'${type}' on line 50, col 63
                if _v is not None: write(_filter(_v, rawExpr='${type}')) # from line 50, col 63.
                write(''') * (''')
                _v = VFFSL(SL,"globalLattice",True) # u'${globalLattice}' on line 50, col 75
                if _v is not None: write(_filter(_v, rawExpr='${globalLattice}')) # from line 50, col 75.
                write('''));
''')
            else: # generated from line 51, col 5
                write('''unsigned long ''')
                _v = VFFSL(SL,"index",True) # u'${index}' on line 52, col 15
                if _v is not None: write(_filter(_v, rawExpr='${index}')) # from line 52, col 15.
                write(''' = 0;
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def openLoopAscending(self, **KWS):



        ## CHEETAH: generated from @def openLoopAscending at line 58, col 1.
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
        _v = VFFSL(SL,"name",True) # u'${name}' on line 60, col 9
        if _v is not None: write(_filter(_v, rawExpr='${name}')) # from line 60, col 9.
        write(''' ''')
        _v = VFFSL(SL,"arrayName",True) # u'${arrayName}' on line 60, col 17
        if _v is not None: write(_filter(_v, rawExpr='${arrayName}')) # from line 60, col 17.
        write('''[''')
        _v = VFFSL(SL,"loopIndex",True) # u'${loopIndex}' on line 60, col 30
        if _v is not None: write(_filter(_v, rawExpr='${loopIndex}')) # from line 60, col 30.
        write(''' + ''')
        _v = VFFSL(SL,"localOffset",True) # u'${localOffset}' on line 60, col 45
        if _v is not None: write(_filter(_v, rawExpr='${localOffset}')) # from line 60, col 45.
        write(''']
''')
        if VFFSL(SL,"stepSizeArray",True): # generated from line 61, col 3
            write('''#define d''')
            _v = VFFSL(SL,"name",True) # u'${name}' on line 62, col 10
            if _v is not None: write(_filter(_v, rawExpr='${name}')) # from line 62, col 10.
            write(''' (''')
            _v = VFFSL(SL,"stepSizeArrayName",True) # u'${stepSizeArrayName}' on line 62, col 19
            if _v is not None: write(_filter(_v, rawExpr='${stepSizeArrayName}')) # from line 62, col 19.
            write('''[''')
            _v = VFFSL(SL,"loopIndex",True) # u'${loopIndex}' on line 62, col 40
            if _v is not None: write(_filter(_v, rawExpr='${loopIndex}')) # from line 62, col 40.
            write(''' + ''')
            _v = VFFSL(SL,"localOffset",True) # u'${localOffset}' on line 62, col 55
            if _v is not None: write(_filter(_v, rawExpr='${localOffset}')) # from line 62, col 55.
            write('''] * (''')
            _v = VFFSL(SL,"volumePrefactor",True) # u'${volumePrefactor}' on line 62, col 74
            if _v is not None: write(_filter(_v, rawExpr='${volumePrefactor}')) # from line 62, col 74.
            write('''))
''')
        write('''
for (long ''')
        _v = VFFSL(SL,"loopIndex",True) # u'${loopIndex}' on line 65, col 11
        if _v is not None: write(_filter(_v, rawExpr='${loopIndex}')) # from line 65, col 11.
        write(''' = 0; ''')
        _v = VFFSL(SL,"loopIndex",True) # u'${loopIndex}' on line 65, col 29
        if _v is not None: write(_filter(_v, rawExpr='${loopIndex}')) # from line 65, col 29.
        write(''' < ''')
        _v = VFFSL(SL,"localLattice",True) # u'${localLattice}' on line 65, col 44
        if _v is not None: write(_filter(_v, rawExpr='${localLattice}')) # from line 65, col 44.
        write('''; ''')
        _v = VFFSL(SL,"loopIndex",True) # u'${loopIndex}' on line 65, col 61
        if _v is not None: write(_filter(_v, rawExpr='${loopIndex}')) # from line 65, col 61.
        write('''++) {
''')
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def closeLoopAscending(self, **KWS):



        ## CHEETAH: generated from @def closeLoopAscending at line 68, col 1.
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
        _v = VFFSL(SL,"name",True) # u'${name}' on line 71, col 8
        if _v is not None: write(_filter(_v, rawExpr='${name}')) # from line 71, col 8.
        write('''
''')
        if VFFSL(SL,"stepSizeArray",True): # generated from line 72, col 3
            write('''#undef d''')
            _v = VFFSL(SL,"name",True) # u'${name}' on line 73, col 9
            if _v is not None: write(_filter(_v, rawExpr='${name}')) # from line 73, col 9.
            write('''
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def localIndexFromIndexForDimensionRep(self, dimRep, **KWS):



        ## CHEETAH: generated from @def localIndexFromIndexForDimensionRep($dimRep) at line 78, col 1.
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
        write('''(''')
        _v = VFFSL(SL,"dimRep.loopIndex",True) # u'${dimRep.loopIndex}' on line 80, col 2
        if _v is not None: write(_filter(_v, rawExpr='${dimRep.loopIndex}')) # from line 80, col 2.
        write(''' + ''')
        _v = VFFSL(SL,"dimRep.localOffset",True) # u'${dimRep.localOffset}' on line 80, col 24
        if _v is not None: write(_filter(_v, rawExpr='${dimRep.localOffset}')) # from line 80, col 24.
        write(''') * (''')
        _v = VFFSL(SL,"globalLattice",True) # u'${globalLattice}' on line 80, col 50
        if _v is not None: write(_filter(_v, rawExpr='${globalLattice}')) # from line 80, col 50.
        write('''/''')
        _v = VFFSL(SL,"dimRep.globalLattice",True) # u'${dimRep.globalLattice}' on line 80, col 67
        if _v is not None: write(_filter(_v, rawExpr='${dimRep.globalLattice}')) # from line 80, col 67.
        write(''') - ''')
        _v = VFFSL(SL,"localOffset",True) # u'${localOffset}' on line 80, col 94
        if _v is not None: write(_filter(_v, rawExpr='${localOffset}')) # from line 80, col 94.
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
        _v = VFFSL(SL,"loopIndex",True) # u'${loopIndex}' on line 86, col 1
        if _v is not None: write(_filter(_v, rawExpr='${loopIndex}')) # from line 86, col 1.
        write(''' + ''')
        _v = VFFSL(SL,"localOffset",True) # u'${localOffset}' on line 86, col 16
        if _v is not None: write(_filter(_v, rawExpr='${localOffset}')) # from line 86, col 16.
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def initialiseArray(self, **KWS):



        ## CHEETAH: generated from @def initialiseArray at line 90, col 1.
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
        if VFFSL(SL,"stepSizeArray",True): # generated from line 92, col 3
            write('''for (long ''')
            _v = VFFSL(SL,"loopIndex",True) # u'${loopIndex}' on line 93, col 11
            if _v is not None: write(_filter(_v, rawExpr='${loopIndex}')) # from line 93, col 11.
            write(''' = 0; ''')
            _v = VFFSL(SL,"loopIndex",True) # u'${loopIndex}' on line 93, col 29
            if _v is not None: write(_filter(_v, rawExpr='${loopIndex}')) # from line 93, col 29.
            write(''' < ''')
            _v = VFFSL(SL,"globalLattice",True) # u'${globalLattice}' on line 93, col 44
            if _v is not None: write(_filter(_v, rawExpr='${globalLattice}')) # from line 93, col 44.
            write('''; ''')
            _v = VFFSL(SL,"loopIndex",True) # u'${loopIndex}' on line 93, col 62
            if _v is not None: write(_filter(_v, rawExpr='${loopIndex}')) # from line 93, col 62.
            write('''++) {
  ''')
            _v = VFFSL(SL,"gridAndStepAtIndex",False)(self.loopIndex) # u'${gridAndStepAtIndex(self.loopIndex), autoIndent=True}' on line 94, col 3
            if _v is not None: write(_filter(_v, autoIndent=True, rawExpr='${gridAndStepAtIndex(self.loopIndex), autoIndent=True}')) # from line 94, col 3.
            write('''}
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def gridAndStepAtIndex(self, index, **KWS):



        ## CHEETAH: generated from @def gridAndStepAtIndex($index) at line 100, col 1.
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
        _v = VFFSL(SL,"arrayName",True) # u'${arrayName}' on line 102, col 1
        if _v is not None: write(_filter(_v, rawExpr='${arrayName}')) # from line 102, col 1.
        write('''[''')
        _v = VFFSL(SL,"index",True) # u'$index' on line 102, col 14
        if _v is not None: write(_filter(_v, rawExpr='$index')) # from line 102, col 14.
        write('''] = ''')
        _v = VFFSL(SL,"gridAtIndex",False)(index) # u'${gridAtIndex(index)}' on line 102, col 24
        if _v is not None: write(_filter(_v, rawExpr='${gridAtIndex(index)}')) # from line 102, col 24.
        write(''';
''')
        _v = VFFSL(SL,"stepSizeArrayName",True) # u'${stepSizeArrayName}' on line 103, col 1
        if _v is not None: write(_filter(_v, rawExpr='${stepSizeArrayName}')) # from line 103, col 1.
        write('''[''')
        _v = VFFSL(SL,"index",True) # u'$index' on line 103, col 22
        if _v is not None: write(_filter(_v, rawExpr='$index')) # from line 103, col 22.
        write('''] = ''')
        _v = VFFSL(SL,"stepWeightAtIndex",False)(index) # u'${stepWeightAtIndex(index)}' on line 103, col 32
        if _v is not None: write(_filter(_v, rawExpr='${stepWeightAtIndex(index)}')) # from line 103, col 32.
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
        # NonUniformDimensionRepresentation.tmpl
        # 
        # Created by Graham Dennis on 2008-07-30.
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

    _mainCheetahMethod_for_NonUniformDimensionRepresentation= 'writeBody'

## END CLASS DEFINITION

if not hasattr(NonUniformDimensionRepresentation, '_initCheetahAttributes'):
    templateAPIClass = getattr(NonUniformDimensionRepresentation, '_CHEETAH_templateClass', Template)
    templateAPIClass._addCheetahPlumbingCodeToClass(NonUniformDimensionRepresentation)


# CHEETAH was developed by Tavis Rudd and Mike Orr
# with code, advice and input from many other volunteers.
# For more information visit http://www.CheetahTemplate.org/

##################################################
## if run from command line:
if __name__ == '__main__':
    from Cheetah.TemplateCmdLineIface import CmdLineIface
    CmdLineIface(templateObj=NonUniformDimensionRepresentation()).run()


