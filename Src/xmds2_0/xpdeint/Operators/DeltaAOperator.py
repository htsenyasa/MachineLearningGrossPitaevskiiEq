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
from xpdeint.Operators._DeltaAOperator import _DeltaAOperator

##################################################
## MODULE CONSTANTS
VFFSL=valueFromFrameOrSearchList
VFSL=valueFromSearchList
VFN=valueForName
currentTime=time.time
__CHEETAH_version__ = '2.4.4'
__CHEETAH_versionTuple__ = (2, 4, 4, 'development', 0)
__CHEETAH_genTime__ = 1484975071.918227
__CHEETAH_genTimestamp__ = 'Sat Jan 21 16:04:31 2017'
__CHEETAH_src__ = '/home/mattias/xmds-2.2.3/admin/staging/xmds-2.2.3/xpdeint/Operators/DeltaAOperator.tmpl'
__CHEETAH_srcLastModified__ = 'Mon Jul 23 09:42:26 2012'
__CHEETAH_docstring__ = 'Autogenerated by Cheetah: The Python-Powered Template Engine'

if __CHEETAH_versionTuple__ < RequiredCheetahVersionTuple:
    raise AssertionError(
      'This template was compiled with Cheetah version'
      ' %s. Templates compiled before version %s must be recompiled.'%(
         __CHEETAH_version__, RequiredCheetahVersion))

##################################################
## CLASSES

class DeltaAOperator(_DeltaAOperator):

    ##################################################
    ## CHEETAH GENERATED METHODS


    def __init__(self, *args, **KWs):

        super(DeltaAOperator, self).__init__(*args, **KWs)
        if not self._CHEETAH__instanceInitialized:
            cheetahKWArgs = {}
            allowedKWs = 'searchList namespaces filter filtersLib errorCatcher'.split()
            for k,v in list(KWs.items()):
                if k in allowedKWs: cheetahKWArgs[k] = v
            self._initCheetahInstance(**cheetahKWArgs)
        

    def description(self, **KWS):



        ## Generated from @def description: Delta A propagation operator for field $field.name at line 26, col 1.
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
        
        write('''Delta A propagation operator for field ''')
        _v = VFFSL(SL,"field.name",True) # u'$field.name' on line 26, col 58
        if _v is not None: write(_filter(_v, rawExpr='$field.name')) # from line 26, col 58.
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def copyDeltaAFunctionContents(self, function, **KWS):



        ## CHEETAH: generated from @def copyDeltaAFunctionContents($function) at line 28, col 1.
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
        
        loopingField = VFFSL(SL,"primaryCodeBlock.field",True)
        dimensionsWithIndexOverrides = [dim for dim in loopingField.dimensions if not VFN(VFFSL(SL,"deltaAField",True),"hasDimension",False)(dim)]
        setOfVectorsToLoopOver = set(VFFSL(SL,"deltaAField.vectors",True))
        setOfVectorsToLoopOver.update(VFFSL(SL,"vectorsForcingReordering",True))
        indexOverrides = dict([(dim.name, {loopingField: ''.join(['_',str(VFN(VFN(VFFSL(SL,"dim",True),"inBasis",False)(VFFSL(SL,"operatorBasis",True)),"name",True)),'_index'])}) for dim in dimensionsWithIndexOverrides])
        _v = VFFSL(SL,"loopOverFieldInBasisWithVectorsAndInnerContent",False)(loopingField, VFFSL(SL,"operatorBasis",True), VFFSL(SL,"setOfVectorsToLoopOver",True), VFFSL(SL,"insideCopyDeltaALoops",True), VFFSL(SL,"indexOverrides",True)) # u'${loopOverFieldInBasisWithVectorsAndInnerContent(loopingField, $operatorBasis, $setOfVectorsToLoopOver, $insideCopyDeltaALoops, $indexOverrides)}' on line 34, col 1
        if _v is not None: write(_filter(_v, rawExpr='${loopOverFieldInBasisWithVectorsAndInnerContent(loopingField, $operatorBasis, $setOfVectorsToLoopOver, $insideCopyDeltaALoops, $indexOverrides)}')) # from line 34, col 1.
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def insideCopyDeltaALoops(self, **KWS):



        ## CHEETAH: generated from @def insideCopyDeltaALoops at line 38, col 1.
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
        write('''// This code copies the increments for the components back into the vectors themselves.
''')
        for vector in VFFSL(SL,"vectorsForcingReordering",True): # generated from line 41, col 3
            for componentName in VFFSL(SL,"vector.components",True): # generated from line 42, col 5
                # 
                _v = VFFSL(SL,"componentName",True) # u'${componentName}' on line 44, col 1
                if _v is not None: write(_filter(_v, rawExpr='${componentName}')) # from line 44, col 1.
                write(''' = d''')
                _v = VFFSL(SL,"componentName",True) # u'${componentName}' on line 44, col 21
                if _v is not None: write(_filter(_v, rawExpr='${componentName}')) # from line 44, col 21.
                write('''_d''')
                _v = VFFSL(SL,"propagationDimension",True) # u'${propagationDimension}' on line 44, col 39
                if _v is not None: write(_filter(_v, rawExpr='${propagationDimension}')) # from line 44, col 39.
                write(''' * _step;
''')
                # 
                if VFN(VFFSL(SL,"deltaAVectorMap",True)[VFFSL(SL,"vector",True)],"needsInitialisation",True): # generated from line 46, col 7
                    #  If the delta a vector needs initialisation, then we need to
                    #  reset it now that we have copied what we need out of it.
                    # 
                    write('''d''')
                    _v = VFFSL(SL,"componentName",True) # u'${componentName}' on line 50, col 2
                    if _v is not None: write(_filter(_v, rawExpr='${componentName}')) # from line 50, col 2.
                    write('''_d''')
                    _v = VFFSL(SL,"propagationDimension",True) # u'${propagationDimension}' on line 50, col 20
                    if _v is not None: write(_filter(_v, rawExpr='${propagationDimension}')) # from line 50, col 20.
                    write(''' = 0.0;
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def insideEvaluateOperatorLoops(self, codeString, **KWS):



        ## CHEETAH: generated from @def insideEvaluateOperatorLoops($codeString) at line 57, col 1.
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
        _v = VFFSL(SL,"insideEvaluateOperatorLoopsBegin",True) # u'${insideEvaluateOperatorLoopsBegin}' on line 59, col 1
        if _v is not None: write(_filter(_v, rawExpr='${insideEvaluateOperatorLoopsBegin}')) # from line 59, col 1.
        # 
        #  The Operator class will have defined for us all of the dVariableName_dPropagationDimension variables.
        #  Note that we assume that all of the integration vectors have an operotor component defined for them.
        write('''#define d''')
        _v = VFFSL(SL,"propagationDimension",True) # u'${propagationDimension}' on line 63, col 10
        if _v is not None: write(_filter(_v, rawExpr='${propagationDimension}')) # from line 63, col 10.
        write(''' _step

// ************* Propagation code ***************
''')
        _v = VFFSL(SL,"codeString",True) # u'${codeString}' on line 66, col 1
        if _v is not None: write(_filter(_v, rawExpr='${codeString}')) # from line 66, col 1.
        write('''// **********************************************

#undef d''')
        _v = VFFSL(SL,"propagationDimension",True) # u'${propagationDimension}' on line 69, col 9
        if _v is not None: write(_filter(_v, rawExpr='${propagationDimension}')) # from line 69, col 9.
        write('''


''')
        #  Loop over the components of the integration vectors
        for operatorComponentName in VFN(VFFSL(SL,"operatorComponents",True),"iterkeys",False)(): # generated from line 73, col 3
            assert len(VFFSL(SL,"operatorComponents",True)[VFFSL(SL,"operatorComponentName",True)]) == 1
            for integrationVector, integrationVectorComponentList in VFN(VFFSL(SL,"operatorComponents",True)[VFFSL(SL,"operatorComponentName",True)],"iteritems",False)(): # generated from line 75, col 5
                integrationVectorComponentName = VFFSL(SL,"integrationVectorComponentList",True)[0]
                assert VFFSL(SL,"integrationVectorComponentName",True) in VFFSL(SL,"integrationVector.components",True)
                write('''_active_''')
                _v = VFFSL(SL,"integrationVector.id",True) # u'${integrationVector.id}' on line 78, col 9
                if _v is not None: write(_filter(_v, rawExpr='${integrationVector.id}')) # from line 78, col 9.
                write('''[_''')
                _v = VFFSL(SL,"integrationVector.id",True) # u'${integrationVector.id}' on line 78, col 34
                if _v is not None: write(_filter(_v, rawExpr='${integrationVector.id}')) # from line 78, col 34.
                write('''_index_pointer + ''')
                _v = VFN(VFFSL(SL,"integrationVector.components",True),"index",False)(VFFSL(SL,"integrationVectorComponentName",True)) # u'${integrationVector.components.index($integrationVectorComponentName)}' on line 78, col 74
                if _v is not None: write(_filter(_v, rawExpr='${integrationVector.components.index($integrationVectorComponentName)}')) # from line 78, col 74.
                write('''] = ''')
                _v = VFFSL(SL,"operatorComponentName",True) # u'$operatorComponentName' on line 79, col 1
                if _v is not None: write(_filter(_v, rawExpr='$operatorComponentName')) # from line 79, col 1.
                write(''' * _step;
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
        # DeltaAOperator.tmpl
        # 
        # delta-a operator, i.e. dstuff_dt = otherStuff;
        # 
        # Created by Graham Dennis on 2007-10-13.
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

    _mainCheetahMethod_for_DeltaAOperator= 'writeBody'

## END CLASS DEFINITION

if not hasattr(DeltaAOperator, '_initCheetahAttributes'):
    templateAPIClass = getattr(DeltaAOperator, '_CHEETAH_templateClass', Template)
    templateAPIClass._addCheetahPlumbingCodeToClass(DeltaAOperator)


# CHEETAH was developed by Tavis Rudd and Mike Orr
# with code, advice and input from many other volunteers.
# For more information visit http://www.CheetahTemplate.org/

##################################################
## if run from command line:
if __name__ == '__main__':
    from Cheetah.TemplateCmdLineIface import CmdLineIface
    CmdLineIface(templateObj=DeltaAOperator()).run()


