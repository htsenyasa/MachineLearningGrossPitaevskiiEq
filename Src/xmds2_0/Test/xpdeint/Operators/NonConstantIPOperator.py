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
from xpdeint.Operators._IPOperator import _IPOperator
from xpdeint.CallOnceGuards import callOncePerInstanceGuard

##################################################
## MODULE CONSTANTS
VFFSL=valueFromFrameOrSearchList
VFSL=valueFromSearchList
VFN=valueForName
currentTime=time.time
__CHEETAH_version__ = '2.4.4'
__CHEETAH_versionTuple__ = (2, 4, 4, 'development', 0)
__CHEETAH_genTime__ = 1484975071.951559
__CHEETAH_genTimestamp__ = 'Sat Jan 21 16:04:31 2017'
__CHEETAH_src__ = '/home/mattias/xmds-2.2.3/admin/staging/xmds-2.2.3/xpdeint/Operators/NonConstantIPOperator.tmpl'
__CHEETAH_srcLastModified__ = 'Fri Oct 11 15:53:21 2013'
__CHEETAH_docstring__ = 'Autogenerated by Cheetah: The Python-Powered Template Engine'

if __CHEETAH_versionTuple__ < RequiredCheetahVersionTuple:
    raise AssertionError(
      'This template was compiled with Cheetah version'
      ' %s. Templates compiled before version %s must be recompiled.'%(
         __CHEETAH_version__, RequiredCheetahVersion))

##################################################
## CLASSES

class NonConstantIPOperator(_IPOperator):

    ##################################################
    ## CHEETAH GENERATED METHODS


    def __init__(self, *args, **KWs):

        super(NonConstantIPOperator, self).__init__(*args, **KWs)
        if not self._CHEETAH__instanceInitialized:
            cheetahKWArgs = {}
            allowedKWs = 'searchList namespaces filter filtersLib errorCatcher'.split()
            for k,v in list(KWs.items()):
                if k in allowedKWs: cheetahKWArgs[k] = v
            self._initCheetahInstance(**cheetahKWArgs)
        

    def globals(self, **KWS):



        ## CHEETAH: generated from @def globals at line 30, col 1.
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
        _v = super(NonConstantIPOperator, self).globals()
        if _v is not None: write(_filter(_v))
        # 
        write('''real _''')
        _v = VFFSL(SL,"id",True) # u'${id}' on line 34, col 7
        if _v is not None: write(_filter(_v, rawExpr='${id}')) # from line 34, col 7.
        write('''_last_timestep_size_map[''')
        _v = VFFSL(SL,"len",False)(VFFSL(SL,"integrator.ipPropagationStepFractions",True)) # u'${len($integrator.ipPropagationStepFractions)}' on line 34, col 36
        if _v is not None: write(_filter(_v, rawExpr='${len($integrator.ipPropagationStepFractions)}')) # from line 34, col 36.
        write('''];
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    @callOncePerInstanceGuard
    def initialise(self, **KWS):



        ## CHEETAH: generated from @def initialise at line 39, col 1.
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
        write('''memset(_''')
        _v = VFFSL(SL,"id",True) # u'${id}' on line 41, col 9
        if _v is not None: write(_filter(_v, rawExpr='${id}')) # from line 41, col 9.
        write('''_last_timestep_size_map, 0, sizeof(_''')
        _v = VFFSL(SL,"id",True) # u'${id}' on line 41, col 50
        if _v is not None: write(_filter(_v, rawExpr='${id}')) # from line 41, col 50.
        write('''_last_timestep_size_map));
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def calculateOperatorFieldFunctionContents(self, function, **KWS):



        ## CHEETAH: generated from @def calculateOperatorFieldFunctionContents($function) at line 45, col 1.
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
        if len(VFFSL(SL,"integrator.ipPropagationStepFractions",True)) > 1: # generated from line 47, col 3
            write('''static const real _propagationStepFractions[] = {
''')
            for propagationStepFraction in VFFSL(SL,"integrator.ipPropagationStepFractions",True): # generated from line 49, col 3
                write('''  ''')
                _v = VFFSL(SL,"propagationStepFraction",True) # u'$propagationStepFraction' on line 50, col 3
                if _v is not None: write(_filter(_v, rawExpr='$propagationStepFraction')) # from line 50, col 3.
                write(''',
''')
            write('''};
const long _arrayIndex = _exponent - 1;
const real _propagationStepFraction = _propagationStepFractions[_arrayIndex];
''')
        else: # generated from line 55, col 3
            write('''const real _propagationStepFraction = ''')
            _v = VFN(VFFSL(SL,"integrator",True),"ipPropagationStepFractions",True)[0] # u'${integrator.ipPropagationStepFractions[0]}' on line 56, col 39
            if _v is not None: write(_filter(_v, rawExpr='${integrator.ipPropagationStepFractions[0]}')) # from line 56, col 39.
            write(''';
const long _arrayIndex = 0;
''')
        write("""
// If the timestep hasn't changed from the last time, then we're done.
if (_propagationStepFraction * _step == _""")
        _v = VFFSL(SL,"id",True) # u'${id}' on line 61, col 42
        if _v is not None: write(_filter(_v, rawExpr='${id}')) # from line 61, col 42.
        write('''_last_timestep_size_map[_arrayIndex])
  return;

''')
        # 
        _v = super(NonConstantIPOperator, self).calculateOperatorFieldFunctionContents(function)
        if _v is not None: write(_filter(_v))
        # 
        write('''
_''')
        _v = VFFSL(SL,"id",True) # u'${id}' on line 68, col 2
        if _v is not None: write(_filter(_v, rawExpr='${id}')) # from line 68, col 2.
        write('''_last_timestep_size_map[_arrayIndex] = _propagationStepFraction * _step;
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def insideCalculateOperatorFieldLoops(self, codeString, **KWS):



        ## CHEETAH: generated from @def insideCalculateOperatorFieldLoops($codeString) at line 72, col 1.
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
        _v = VFFSL(SL,"insideCalculateOperatorFieldLoopsBegin",True) # u'${insideCalculateOperatorFieldLoopsBegin}' on line 74, col 1
        if _v is not None: write(_filter(_v, rawExpr='${insideCalculateOperatorFieldLoopsBegin}')) # from line 74, col 1.
        # 
        #  We expect the integrator to have defined '_step'
        # 
        write("""// The purpose of the following define is to give a (somewhat helpful) compile-time error
// if the user has attempted to use the propagation dimension variable in a constant IP operator/
// The user probably shouldn't be doing this, but if they must, they should use a non-constant EX
// operator instead
#define """)
        _v = VFFSL(SL,"propagationDimension",True) # u'${propagationDimension}' on line 82, col 9
        if _v is not None: write(_filter(_v, rawExpr='${propagationDimension}')) # from line 82, col 9.
        write(''' Dont_use_propagation_dimension_''')
        _v = VFFSL(SL,"propagationDimension",True) # u'${propagationDimension}' on line 82, col 64
        if _v is not None: write(_filter(_v, rawExpr='${propagationDimension}')) # from line 82, col 64.
        write('''_in_constant_IP_operator___Use_non_constant_EX_operator_instead
// ************** Operator code *****************
''')
        _v = VFFSL(SL,"codeString",True) # u'${codeString}' on line 84, col 1
        if _v is not None: write(_filter(_v, rawExpr='${codeString}')) # from line 84, col 1.
        write('''// **********************************************
#undef ''')
        _v = VFFSL(SL,"propagationDimension",True) # u'${propagationDimension}' on line 86, col 8
        if _v is not None: write(_filter(_v, rawExpr='${propagationDimension}')) # from line 86, col 8.
        write('''

''')
        #  Loop over each operator component
        for operatorComponentNumber, operatorComponent in enumerate(VFN(VFFSL(SL,"operatorComponents",True),"iterkeys",False)()): # generated from line 89, col 3
            write('''_''')
            _v = VFFSL(SL,"operatorVector.id",True) # u'${operatorVector.id}' on line 90, col 2
            if _v is not None: write(_filter(_v, rawExpr='${operatorVector.id}')) # from line 90, col 2.
            write('''[_''')
            _v = VFFSL(SL,"operatorVector.id",True) # u'${operatorVector.id}' on line 90, col 24
            if _v is not None: write(_filter(_v, rawExpr='${operatorVector.id}')) # from line 90, col 24.
            write('''_index_pointer + _arrayIndex * ''')
            _v = VFFSL(SL,"len",False)(VFFSL(SL,"operatorComponents",True)) # u'$len($operatorComponents)' on line 90, col 75
            if _v is not None: write(_filter(_v, rawExpr='$len($operatorComponents)')) # from line 90, col 75.
            write(''' + ''')
            _v = VFFSL(SL,"operatorComponentNumber",True) # u'${operatorComponentNumber}' on line 90, col 103
            if _v is not None: write(_filter(_v, rawExpr='${operatorComponentNumber}')) # from line 90, col 103.
            write(''']''')
            write(''' = ''')
            _v = VFFSL(SL,"expFunction",True) # u'${expFunction}' on line 91, col 4
            if _v is not None: write(_filter(_v, rawExpr='${expFunction}')) # from line 91, col 4.
            write('''(''')
            _v = VFFSL(SL,"operatorComponent",True) # u'${operatorComponent}' on line 91, col 19
            if _v is not None: write(_filter(_v, rawExpr='${operatorComponent}')) # from line 91, col 19.
            _v = VFFSL(SL,"valueSuffix",True) # u'${valueSuffix}' on line 91, col 39
            if _v is not None: write(_filter(_v, rawExpr='${valueSuffix}')) # from line 91, col 39.
            write(''' * _propagationStepFraction * _step);
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def exponentIndex(self, **KWS):



        ## CHEETAH: generated from @def exponentIndex at line 96, col 1.
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
        write('''(abs(_exponent) - 1) * ''')
        _v = VFFSL(SL,"len",False)(VFFSL(SL,"operatorComponents",True)) # u'${len($operatorComponents)}' on line 98, col 24
        if _v is not None: write(_filter(_v, rawExpr='${len($operatorComponents)}')) # from line 98, col 24.
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
        # NonConstantIPOperator.tmpl
        # 
        # Interaction-picture transverse derivative operator
        # 
        # Created by Graham Dennis on 2007-11-21.
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

    calculateOperatorFieldFunctionArguments = [('real', '_step'), ('int', '_exponent')]

    _mainCheetahMethod_for_NonConstantIPOperator= 'writeBody'

## END CLASS DEFINITION

if not hasattr(NonConstantIPOperator, '_initCheetahAttributes'):
    templateAPIClass = getattr(NonConstantIPOperator, '_CHEETAH_templateClass', Template)
    templateAPIClass._addCheetahPlumbingCodeToClass(NonConstantIPOperator)


# CHEETAH was developed by Tavis Rudd and Mike Orr
# with code, advice and input from many other volunteers.
# For more information visit http://www.CheetahTemplate.org/

##################################################
## if run from command line:
if __name__ == '__main__':
    from Cheetah.TemplateCmdLineIface import CmdLineIface
    CmdLineIface(templateObj=NonConstantIPOperator()).run()

