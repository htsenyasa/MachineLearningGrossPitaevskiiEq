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
from xpdeint.Features.Transforms._BesselTransform import _BesselTransform

##################################################
## MODULE CONSTANTS
VFFSL=valueFromFrameOrSearchList
VFSL=valueFromSearchList
VFN=valueForName
currentTime=time.time
__CHEETAH_version__ = '2.4.4'
__CHEETAH_versionTuple__ = (2, 4, 4, 'development', 0)
__CHEETAH_genTime__ = 1484975071.499117
__CHEETAH_genTimestamp__ = 'Sat Jan 21 16:04:31 2017'
__CHEETAH_src__ = '/home/mattias/xmds-2.2.3/admin/staging/xmds-2.2.3/xpdeint/Features/Transforms/BesselNeumannTransform.tmpl'
__CHEETAH_srcLastModified__ = 'Thu Nov 28 08:44:05 2013'
__CHEETAH_docstring__ = 'Autogenerated by Cheetah: The Python-Powered Template Engine'

if __CHEETAH_versionTuple__ < RequiredCheetahVersionTuple:
    raise AssertionError(
      'This template was compiled with Cheetah version'
      ' %s. Templates compiled before version %s must be recompiled.'%(
         __CHEETAH_version__, RequiredCheetahVersion))

##################################################
## CLASSES

class BesselNeumannTransform(_BesselTransform):

    ##################################################
    ## CHEETAH GENERATED METHODS


    def __init__(self, *args, **KWs):

        super(BesselNeumannTransform, self).__init__(*args, **KWs)
        if not self._CHEETAH__instanceInitialized:
            cheetahKWArgs = {}
            allowedKWs = 'searchList namespaces filter filtersLib errorCatcher'.split()
            for k,v in list(KWs.items()):
                if k in allowedKWs: cheetahKWArgs[k] = v
            self._initCheetahInstance(**cheetahKWArgs)
        

    def globalsForDim(self, dimName, basisDict, **KWS):



        ## CHEETAH: generated from @def globalsForDim($dimName, $basisDict) at line 24, col 1.
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
        besselOrder = basisDict['order']
        lattice = basisDict['lattice']
        write('''const real _besseljzeros_''')
        _v = VFFSL(SL,"dimName",True) # u'${dimName}' on line 28, col 26
        if _v is not None: write(_filter(_v, rawExpr='${dimName}')) # from line 28, col 26.
        write('''[] = {''')
        _v = VFFSL(SL,"wrapArray",False)(VFFSL(SL,"besselJPrimeZeros",False)(besselOrder, lattice)) # u'${wrapArray($besselJPrimeZeros(besselOrder, lattice))}' on line 28, col 42
        if _v is not None: write(_filter(_v, rawExpr='${wrapArray($besselJPrimeZeros(besselOrder, lattice))}')) # from line 28, col 42.
        write('''};
const real _besseljS_''')
        _v = VFFSL(SL,"dimName",True) # u'${dimName}' on line 29, col 22
        if _v is not None: write(_filter(_v, rawExpr='${dimName}')) # from line 29, col 22.
        write(''' = ''')
        _v = VFFSL(SL,"besselNeumannSFactor",False)(besselOrder, lattice) # u'${besselNeumannSFactor(besselOrder, lattice)}' on line 29, col 35
        if _v is not None: write(_filter(_v, rawExpr='${besselNeumannSFactor(besselOrder, lattice)}')) # from line 29, col 35.
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
        # BesselNeumannTransform.tmpl
        # 
        # Created by Graham Dennis on 2013-11-26.
        # 
        # Copyright (c) 2013, Graham Dennis
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

    _mainCheetahMethod_for_BesselNeumannTransform= 'writeBody'

## END CLASS DEFINITION

if not hasattr(BesselNeumannTransform, '_initCheetahAttributes'):
    templateAPIClass = getattr(BesselNeumannTransform, '_CHEETAH_templateClass', Template)
    templateAPIClass._addCheetahPlumbingCodeToClass(BesselNeumannTransform)


# CHEETAH was developed by Tavis Rudd and Mike Orr
# with code, advice and input from many other volunteers.
# For more information visit http://www.CheetahTemplate.org/

##################################################
## if run from command line:
if __name__ == '__main__':
    from Cheetah.TemplateCmdLineIface import CmdLineIface
    CmdLineIface(templateObj=BesselNeumannTransform()).run()


