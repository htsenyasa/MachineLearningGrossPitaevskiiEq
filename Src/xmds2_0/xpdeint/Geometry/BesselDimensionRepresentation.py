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
from xpdeint.Geometry.NonUniformDimensionRepresentation import NonUniformDimensionRepresentation

##################################################
## MODULE CONSTANTS
VFFSL=valueFromFrameOrSearchList
VFSL=valueFromSearchList
VFN=valueForName
currentTime=time.time
__CHEETAH_version__ = '2.4.4'
__CHEETAH_versionTuple__ = (2, 4, 4, 'development', 0)
__CHEETAH_genTime__ = 1484975071.731568
__CHEETAH_genTimestamp__ = 'Sat Jan 21 16:04:31 2017'
__CHEETAH_src__ = '/home/mattias/xmds-2.2.3/admin/staging/xmds-2.2.3/xpdeint/Geometry/BesselDimensionRepresentation.tmpl'
__CHEETAH_srcLastModified__ = 'Tue Nov 26 20:52:00 2013'
__CHEETAH_docstring__ = 'Autogenerated by Cheetah: The Python-Powered Template Engine'

if __CHEETAH_versionTuple__ < RequiredCheetahVersionTuple:
    raise AssertionError(
      'This template was compiled with Cheetah version'
      ' %s. Templates compiled before version %s must be recompiled.'%(
         __CHEETAH_version__, RequiredCheetahVersion))

##################################################
## CLASSES

class BesselDimensionRepresentation(NonUniformDimensionRepresentation):

    ##################################################
    ## CHEETAH GENERATED METHODS


    def __init__(self, *args, **KWs):

        super(BesselDimensionRepresentation, self).__init__(*args, **KWs)
        if not self._CHEETAH__instanceInitialized:
            cheetahKWArgs = {}
            allowedKWs = 'searchList namespaces filter filtersLib errorCatcher'.split()
            for k,v in list(KWs.items()):
                if k in allowedKWs: cheetahKWArgs[k] = v
            self._initCheetahInstance(**cheetahKWArgs)
        

    def besselJFunctionCall(self, order, argument, **KWS):



        ## CHEETAH: generated from @def besselJFunctionCall($order, $argument) at line 28, col 1.
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
        
        if VFFSL(SL,"order",True) in [0, 1]: # generated from line 29, col 3
            write('''j''')
            _v = VFFSL(SL,"order",True) # u'${order}' on line 30, col 2
            if _v is not None: write(_filter(_v, rawExpr='${order}')) # from line 30, col 2.
            write('''(''')
            _v = VFFSL(SL,"argument",True) # u'$argument' on line 30, col 11
            if _v is not None: write(_filter(_v, rawExpr='$argument')) # from line 30, col 11.
            write(''')''')
        else: # generated from line 31, col 3
            write('''jn(''')
            _v = VFFSL(SL,"order",True) # u'${order}' on line 32, col 4
            if _v is not None: write(_filter(_v, rawExpr='${order}')) # from line 32, col 4.
            write(''', ''')
            _v = VFFSL(SL,"argument",True) # u'${argument}' on line 32, col 14
            if _v is not None: write(_filter(_v, rawExpr='${argument}')) # from line 32, col 14.
            write(''')''')
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def gridAndStepAtIndex(self, index, **KWS):



        ## CHEETAH: generated from @def gridAndStepAtIndex($index) at line 36, col 1.
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
        write('''const real besselFactor = ''')
        _v = VFFSL(SL,"besselJFunctionCall",False)(VFFSL(SL,"_weightOrder",True), ''.join(['_besseljzeros_',str(VFFSL(SL,"parent.name",True)),'[',str(VFFSL(SL,"index",True)),']'])) # u"${besselJFunctionCall($_weightOrder, c'_besseljzeros_${parent.name}[${index}]')}" on line 38, col 27
        if _v is not None: write(_filter(_v, rawExpr="${besselJFunctionCall($_weightOrder, c'_besseljzeros_${parent.name}[${index}]')}")) # from line 38, col 27.
        write(''';
const real ''')
        _v = VFFSL(SL,"name",True) # u'${name}' on line 39, col 12
        if _v is not None: write(_filter(_v, rawExpr='${name}')) # from line 39, col 12.
        write('''_max = ''')
        _v = VFFSL(SL,"_maximum",True) # u'${_maximum}' on line 39, col 26
        if _v is not None: write(_filter(_v, rawExpr='${_maximum}')) # from line 39, col 26.
        write(''';
''')
        # 
        _v = super(BesselDimensionRepresentation, self).gridAndStepAtIndex(index)
        if _v is not None: write(_filter(_v))
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def gridAtIndex(self, index, **KWS):



        ## CHEETAH: generated from @def gridAtIndex($index) at line 45, col 1.
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
        write('''(_besseljzeros_''')
        _v = VFFSL(SL,"parent.name",True) # u'${parent.name}' on line 47, col 16
        if _v is not None: write(_filter(_v, rawExpr='${parent.name}')) # from line 47, col 16.
        write('''[''')
        _v = VFFSL(SL,"index",True) # u'$index' on line 47, col 31
        if _v is not None: write(_filter(_v, rawExpr='$index')) # from line 47, col 31.
        write('''] / _besseljS_''')
        _v = VFFSL(SL,"parent.name",True) # u'${parent.name}' on line 47, col 51
        if _v is not None: write(_filter(_v, rawExpr='${parent.name}')) # from line 47, col 51.
        write(''') * ''')
        _v = VFFSL(SL,"name",True) # u'${name}' on line 47, col 69
        if _v is not None: write(_filter(_v, rawExpr='${name}')) # from line 47, col 69.
        write('''_max''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def stepWeightAtIndex(self, index, **KWS):



        ## CHEETAH: generated from @def stepWeightAtIndex($index) at line 51, col 1.
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
        write('''2.0 / (besselFactor * besselFactor * _besseljS_''')
        _v = VFFSL(SL,"parent.name",True) # u'${parent.name}' on line 53, col 48
        if _v is not None: write(_filter(_v, rawExpr='${parent.name}')) # from line 53, col 48.
        write(''' * _besseljS_''')
        _v = VFFSL(SL,"parent.name",True) # u'${parent.name}' on line 53, col 75
        if _v is not None: write(_filter(_v, rawExpr='${parent.name}')) # from line 53, col 75.
        write(''') * ''')
        _v = VFFSL(SL,"name",True) # u'${name}' on line 53, col 93
        if _v is not None: write(_filter(_v, rawExpr='${name}')) # from line 53, col 93.
        write('''_max * ''')
        _v = VFFSL(SL,"name",True) # u'${name}' on line 53, col 107
        if _v is not None: write(_filter(_v, rawExpr='${name}')) # from line 53, col 107.
        write('''_max''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def indexForSinglePointSample(self, **KWS):



        ## CHEETAH: generated from @def indexForSinglePointSample at line 57, col 1.
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
        #  Take the first point, which is close to r=0
        write('''0''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def createCoordinateVariableForSinglePointSample(self, **KWS):



        ## CHEETAH: generated from @def createCoordinateVariableForSinglePointSample at line 64, col 1.
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
        _v = VFFSL(SL,"type",True) # u'${type}' on line 66, col 1
        if _v is not None: write(_filter(_v, rawExpr='${type}')) # from line 66, col 1.
        write(''' ''')
        _v = VFFSL(SL,"name",True) # u'${name}' on line 66, col 9
        if _v is not None: write(_filter(_v, rawExpr='${name}')) # from line 66, col 9.
        write(''' = ''')
        _v = VFFSL(SL,"arrayName",True) # u'${arrayName}' on line 66, col 19
        if _v is not None: write(_filter(_v, rawExpr='${arrayName}')) # from line 66, col 19.
        write('''[0];
#define d''')
        _v = VFFSL(SL,"name",True) # u'${name}' on line 67, col 10
        if _v is not None: write(_filter(_v, rawExpr='${name}')) # from line 67, col 10.
        write(''' (''')
        _v = VFFSL(SL,"stepSizeArrayName",True) # u'${stepSizeArrayName}' on line 67, col 19
        if _v is not None: write(_filter(_v, rawExpr='${stepSizeArrayName}')) # from line 67, col 19.
        write('''[0] * (''')
        _v = VFFSL(SL,"volumePrefactor",True) # u'${volumePrefactor}' on line 67, col 46
        if _v is not None: write(_filter(_v, rawExpr='${volumePrefactor}')) # from line 67, col 46.
        write('''))
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
        # BesselDimensionRepresentation.tmpl
        # 
        # Created by Graham Dennis on 2009-08-11.
        # 
        # Copyright (c) 2009-2012, Graham Dennis
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

    instanceAttributes = ['_maximum', '_order', '_weightOrder']

    orderOffset = 0

    _mainCheetahMethod_for_BesselDimensionRepresentation= 'writeBody'

## END CLASS DEFINITION

if not hasattr(BesselDimensionRepresentation, '_initCheetahAttributes'):
    templateAPIClass = getattr(BesselDimensionRepresentation, '_CHEETAH_templateClass', Template)
    templateAPIClass._addCheetahPlumbingCodeToClass(BesselDimensionRepresentation)


# CHEETAH was developed by Tavis Rudd and Mike Orr
# with code, advice and input from many other volunteers.
# For more information visit http://www.CheetahTemplate.org/

##################################################
## if run from command line:
if __name__ == '__main__':
    from Cheetah.TemplateCmdLineIface import CmdLineIface
    CmdLineIface(templateObj=BesselDimensionRepresentation()).run()


