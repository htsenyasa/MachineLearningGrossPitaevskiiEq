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
from xpdeint.Features.Transforms.HermiteGaussEPBasis import HermiteGaussEPBasis

##################################################
## MODULE CONSTANTS
VFFSL=valueFromFrameOrSearchList
VFSL=valueFromSearchList
VFN=valueForName
currentTime=time.time
__CHEETAH_version__ = '2.4.4'
__CHEETAH_versionTuple__ = (2, 4, 4, 'development', 0)
__CHEETAH_genTime__ = 1484975071.661459
__CHEETAH_genTimestamp__ = 'Sat Jan 21 16:04:31 2017'
__CHEETAH_src__ = '/home/mattias/xmds-2.2.3/admin/staging/xmds-2.2.3/xpdeint/Features/Transforms/HermiteGaussFourierEPBasis.tmpl'
__CHEETAH_srcLastModified__ = 'Wed Jul  3 14:24:15 2013'
__CHEETAH_docstring__ = 'Autogenerated by Cheetah: The Python-Powered Template Engine'

if __CHEETAH_versionTuple__ < RequiredCheetahVersionTuple:
    raise AssertionError(
      'This template was compiled with Cheetah version'
      ' %s. Templates compiled before version %s must be recompiled.'%(
         __CHEETAH_version__, RequiredCheetahVersion))

##################################################
## CLASSES

class HermiteGaussFourierEPBasis(HermiteGaussEPBasis):

    ##################################################
    ## CHEETAH GENERATED METHODS


    def __init__(self, *args, **KWs):

        super(HermiteGaussFourierEPBasis, self).__init__(*args, **KWs)
        if not self._CHEETAH__instanceInitialized:
            cheetahKWArgs = {}
            allowedKWs = 'searchList namespaces filter filtersLib errorCatcher'.split()
            for k,v in list(KWs.items()):
                if k in allowedKWs: cheetahKWArgs[k] = v
            self._initCheetahInstance(**cheetahKWArgs)
        

    def description(self, **KWS):



        ## Generated from @def description: Hermite-Gauss Fourier basis (Harmonic oscillator) at line 27, col 1.
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
        
        write('''Hermite-Gauss Fourier basis (Harmonic oscillator)''')
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def transformMatricesForwardDimConstantsAtIndex(self, forwardDimRep, backwardDimRep, forwardIndex, **KWS):



        ## CHEETAH: generated from @def transformMatricesForwardDimConstantsAtIndex($forwardDimRep, $backwardDimRep, $forwardIndex) at line 31, col 1.
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
        
        _v = super(HermiteGaussFourierEPBasis, self).transformMatricesForwardDimConstantsAtIndex(forwardDimRep,backwardDimRep,forwardIndex)
        if _v is not None: write(_filter(_v))
        write('''complex eigenvalue_factor = 1.0;
''')
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def transformMatricesForDimRepsAtIndices(self, forwardDimRep, backwardDimRep, forwardIndex, backwardIndex, **KWS):



        ## CHEETAH: generated from @def transformMatricesForDimRepsAtIndices($forwardDimRep, $backwardDimRep, $forwardIndex, $backwardIndex) at line 36, col 1.
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
        _v = super(HermiteGaussFourierEPBasis, self).transformMatricesForDimRepsAtIndices(forwardDimRep,backwardDimRep,forwardIndex,backwardIndex)
        if _v is not None: write(_filter(_v))
        write('''eigenvalue_factor *= i;
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def forwardMatrixForDimAtIndices(self, forwardDimRep, backwardDimRep, forwardIndex, backwardIndex, **KWS):



        ## CHEETAH: generated from @def forwardMatrixForDimAtIndices($forwardDimRep, $backwardDimRep, $forwardIndex, $backwardIndex) at line 43, col 1.
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
        
        write('''eigenvalue_factor * ''')
        _v = super(HermiteGaussFourierEPBasis, self).forwardMatrixForDimAtIndices(forwardDimRep,backwardDimRep,forwardIndex,backwardIndex)
        if _v is not None: write(_filter(_v))
        write('''
''')
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def backwardMatrixForDimAtIndices(self, forwardDimRep, backwardDimRep, forwardIndex, backwardIndex, **KWS):



        ## CHEETAH: generated from @def backwardMatrixForDimAtIndices($forwardDimRep, $backwardDimRep, $forwardIndex, $backwardIndex) at line 47, col 1.
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
        
        write('''conj(eigenvalue_factor) * ''')
        _v = super(HermiteGaussFourierEPBasis, self).backwardMatrixForDimAtIndices(forwardDimRep,backwardDimRep,forwardIndex,backwardIndex)
        if _v is not None: write(_filter(_v))
        write('''
''')
        
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
        # HermiteGaussFourierEPBasis.tmpl
        # 
        # Hermite-Gauss Fourier basis using the definite parity of the basis functions to remove
        # half the work.
        # 
        # Created by Graham Dennis on 2009-08-12.
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

    matrixType = 'complex'

    _mainCheetahMethod_for_HermiteGaussFourierEPBasis= 'writeBody'

## END CLASS DEFINITION

if not hasattr(HermiteGaussFourierEPBasis, '_initCheetahAttributes'):
    templateAPIClass = getattr(HermiteGaussFourierEPBasis, '_CHEETAH_templateClass', Template)
    templateAPIClass._addCheetahPlumbingCodeToClass(HermiteGaussFourierEPBasis)


# CHEETAH was developed by Tavis Rudd and Mike Orr
# with code, advice and input from many other volunteers.
# For more information visit http://www.CheetahTemplate.org/

##################################################
## if run from command line:
if __name__ == '__main__':
    from Cheetah.TemplateCmdLineIface import CmdLineIface
    CmdLineIface(templateObj=HermiteGaussFourierEPBasis()).run()

