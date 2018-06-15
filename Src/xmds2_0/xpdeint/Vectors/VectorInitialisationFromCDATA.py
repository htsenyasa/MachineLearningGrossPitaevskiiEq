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
from xpdeint.Vectors.VectorInitialisation import VectorInitialisation

##################################################
## MODULE CONSTANTS
VFFSL=valueFromFrameOrSearchList
VFSL=valueFromSearchList
VFN=valueForName
currentTime=time.time
__CHEETAH_version__ = '2.4.4'
__CHEETAH_versionTuple__ = (2, 4, 4, 'development', 0)
__CHEETAH_genTime__ = 1484975072.898797
__CHEETAH_genTimestamp__ = 'Sat Jan 21 16:04:32 2017'
__CHEETAH_src__ = '/home/mattias/xmds-2.2.3/admin/staging/xmds-2.2.3/xpdeint/Vectors/VectorInitialisationFromCDATA.tmpl'
__CHEETAH_srcLastModified__ = 'Fri May 25 16:17:13 2012'
__CHEETAH_docstring__ = 'Autogenerated by Cheetah: The Python-Powered Template Engine'

if __CHEETAH_versionTuple__ < RequiredCheetahVersionTuple:
    raise AssertionError(
      'This template was compiled with Cheetah version'
      ' %s. Templates compiled before version %s must be recompiled.'%(
         __CHEETAH_version__, RequiredCheetahVersion))

##################################################
## CLASSES

class VectorInitialisationFromCDATA(VectorInitialisation):

    ##################################################
    ## CHEETAH GENERATED METHODS


    def __init__(self, *args, **KWs):

        super(VectorInitialisationFromCDATA, self).__init__(*args, **KWs)
        if not self._CHEETAH__instanceInitialized:
            cheetahKWArgs = {}
            allowedKWs = 'searchList namespaces filter filtersLib errorCatcher'.split()
            for k,v in list(KWs.items()):
                if k in allowedKWs: cheetahKWArgs[k] = v
            self._initCheetahInstance(**cheetahKWArgs)
        

    def description(self, **KWS):



        ## Generated from @def description: Vector initialisation from CDATA at line 27, col 1.
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
        
        write('''Vector initialisation from CDATA''')
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def initialiseVector(self, **KWS):



        ## CHEETAH: generated from @def initialiseVector at line 33, col 1.
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
        vectorOverrides = []
        if VFFSL(SL,"vector.integratingComponents",True): # generated from line 36, col 2
            vectorOverrides.append(VFFSL(SL,"vector",True))
        _v = VFN(VFFSL(SL,"codeBlocks",True)['initialisation'],"loop",False)(self.insideInitialisationLoops, vectorOverrides=vectorOverrides) # u"$codeBlocks['initialisation'].loop(self.insideInitialisationLoops, vectorOverrides=vectorOverrides)" on line 39, col 1
        if _v is not None: write(_filter(_v, rawExpr="$codeBlocks['initialisation'].loop(self.insideInitialisationLoops, vectorOverrides=vectorOverrides)")) # from line 39, col 1.
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def insideInitialisationLoops(self, codeString, **KWS):



        ## CHEETAH: generated from @def insideInitialisationLoops($codeString) at line 44, col 1.
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
        
        write("""// The purpose of the following define is to give a (somewhat helpful) compile-time error
// if the user has attempted to use the propagation dimension variable in the initialisation
// block of a <vector> element. If they're trying to do this, what they really want is a 
// <computed_vector> instead.
#define """)
        _v = VFFSL(SL,"propagationDimension",True) # u'${propagationDimension}' on line 49, col 9
        if _v is not None: write(_filter(_v, rawExpr='${propagationDimension}')) # from line 49, col 9.
        write(''' Dont_use_propagation_dimension_''')
        _v = VFFSL(SL,"propagationDimension",True) # u'${propagationDimension}' on line 49, col 64
        if _v is not None: write(_filter(_v, rawExpr='${propagationDimension}')) # from line 49, col 64.
        write('''_in_vector_element_CDATA_block___Use_a_computed_vector_instead

// ********** Initialisation code ***************
''')
        _v = VFFSL(SL,"codeString",True) # u'${codeString}' on line 52, col 1
        if _v is not None: write(_filter(_v, rawExpr='${codeString}')) # from line 52, col 1.
        write('''// **********************************************
#undef ''')
        _v = VFFSL(SL,"propagationDimension",True) # u'${propagationDimension}' on line 54, col 8
        if _v is not None: write(_filter(_v, rawExpr='${propagationDimension}')) # from line 54, col 8.
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
        # VectorInitialisationFromCDATA.tmpl
        # 
        # Created by Graham Dennis on 2007-08-29.
        # 
        # Copyright (c) 2007-2012, Graham Dennis
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
        # 
        #   Description of initialisation method
        write('''

''')
        # 
        #   Initialise a vector
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

    _mainCheetahMethod_for_VectorInitialisationFromCDATA= 'writeBody'

## END CLASS DEFINITION

if not hasattr(VectorInitialisationFromCDATA, '_initCheetahAttributes'):
    templateAPIClass = getattr(VectorInitialisationFromCDATA, '_CHEETAH_templateClass', Template)
    templateAPIClass._addCheetahPlumbingCodeToClass(VectorInitialisationFromCDATA)


# CHEETAH was developed by Tavis Rudd and Mike Orr
# with code, advice and input from many other volunteers.
# For more information visit http://www.CheetahTemplate.org/

##################################################
## if run from command line:
if __name__ == '__main__':
    from Cheetah.TemplateCmdLineIface import CmdLineIface
    CmdLineIface(templateObj=VectorInitialisationFromCDATA()).run()


