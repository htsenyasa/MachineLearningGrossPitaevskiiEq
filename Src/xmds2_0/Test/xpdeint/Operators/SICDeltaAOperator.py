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
from xpdeint.Operators._SICDeltaAOperator import _SICDeltaAOperator

##################################################
## MODULE CONSTANTS
VFFSL=valueFromFrameOrSearchList
VFSL=valueFromSearchList
VFN=valueForName
currentTime=time.time
__CHEETAH_version__ = '2.4.4'
__CHEETAH_versionTuple__ = (2, 4, 4, 'development', 0)
__CHEETAH_genTime__ = 1484975072.07045
__CHEETAH_genTimestamp__ = 'Sat Jan 21 16:04:32 2017'
__CHEETAH_src__ = '/home/mattias/xmds-2.2.3/admin/staging/xmds-2.2.3/xpdeint/Operators/SICDeltaAOperator.tmpl'
__CHEETAH_srcLastModified__ = 'Mon Jul 23 09:42:26 2012'
__CHEETAH_docstring__ = 'Autogenerated by Cheetah: The Python-Powered Template Engine'

if __CHEETAH_versionTuple__ < RequiredCheetahVersionTuple:
    raise AssertionError(
      'This template was compiled with Cheetah version'
      ' %s. Templates compiled before version %s must be recompiled.'%(
         __CHEETAH_version__, RequiredCheetahVersion))

##################################################
## CLASSES

class SICDeltaAOperator(_SICDeltaAOperator):

    ##################################################
    ## CHEETAH GENERATED METHODS


    def __init__(self, *args, **KWs):

        super(SICDeltaAOperator, self).__init__(*args, **KWs)
        if not self._CHEETAH__instanceInitialized:
            cheetahKWArgs = {}
            allowedKWs = 'searchList namespaces filter filtersLib errorCatcher'.split()
            for k,v in list(KWs.items()):
                if k in allowedKWs: cheetahKWArgs[k] = v
            self._initCheetahInstance(**cheetahKWArgs)
        

    def description(self, **KWS):



        ## Generated from @def description: Left/Right Delta A propagation operator for field $field.name at line 26, col 1.
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
        
        write('''Left/Right Delta A propagation operator for field ''')
        _v = VFFSL(SL,"field.name",True) # u'$field.name' on line 26, col 69
        if _v is not None: write(_filter(_v, rawExpr='$field.name')) # from line 26, col 69.
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def callEvaluateLoop(self, **KWS):



        ## CHEETAH: generated from @def callEvaluateLoop at line 28, col 1.
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
        for crossIntegrationVector in VFFSL(SL,"crossIntegrationVectors",True): # generated from line 30, col 3
            for componentName in crossIntegrationVector.components: # generated from line 31, col 5
                _v = VFFSL(SL,"crossIntegrationVector.type",True) # u'${crossIntegrationVector.type}' on line 32, col 1
                if _v is not None: write(_filter(_v, rawExpr='${crossIntegrationVector.type}')) # from line 32, col 1.
                write(''' _old_d''')
                _v = VFFSL(SL,"componentName",True) # u'${componentName}' on line 32, col 38
                if _v is not None: write(_filter(_v, rawExpr='${componentName}')) # from line 32, col 38.
                write('''_d''')
                _v = VFFSL(SL,"crossPropagationDimension",True) # u'${crossPropagationDimension}' on line 32, col 56
                if _v is not None: write(_filter(_v, rawExpr='${crossPropagationDimension}')) # from line 32, col 56.
                write(''';
''')
        # 
        loopingOrder = {                        '+': SICDeltaAOperator.LoopingOrder.StrictlyAscendingOrder,                        '-': SICDeltaAOperator.LoopingOrder.StrictlyDescendingOrder                       }[self.crossPropagationDirection]
        _v = VFN(VFFSL(SL,"codeBlocks",True)['operatorDefinition'],"loop",False)(self.insideEvaluateOperatorLoops, loopingOrder = loopingOrder) # u"${codeBlocks['operatorDefinition'].loop(self.insideEvaluateOperatorLoops, loopingOrder = loopingOrder)}" on line 40, col 1
        if _v is not None: write(_filter(_v, rawExpr="${codeBlocks['operatorDefinition'].loop(self.insideEvaluateOperatorLoops, loopingOrder = loopingOrder)}")) # from line 40, col 1.
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def insideEvaluateOperatorLoops(self, codeString, **KWS):



        ## CHEETAH: generated from @def insideEvaluateOperatorLoops($codeString) at line 43, col 1.
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
        _v = VFFSL(SL,"insideEvaluateOperatorLoopsBegin",True) # u'${insideEvaluateOperatorLoopsBegin}' on line 45, col 1
        if _v is not None: write(_filter(_v, rawExpr='${insideEvaluateOperatorLoopsBegin}')) # from line 45, col 1.
        # 
        #  The Operator class will have defined for us all of the dVariableName_dPropagationDimension variables.
        #  Note that we assume that all of the integration vectors have an operotor component defined for them.
        write('''  
// UNVECTORISABLE
''')
        for crossIntegrationVector in VFFSL(SL,"crossIntegrationVectors",True): # generated from line 51, col 3
            for componentName in crossIntegrationVector.components: # generated from line 52, col 5
                write('''d''')
                _v = VFFSL(SL,"componentName",True) # u'${componentName}' on line 53, col 2
                if _v is not None: write(_filter(_v, rawExpr='${componentName}')) # from line 53, col 2.
                write('''_d''')
                _v = VFFSL(SL,"crossPropagationDimension",True) # u'${crossPropagationDimension}' on line 53, col 20
                if _v is not None: write(_filter(_v, rawExpr='${crossPropagationDimension}')) # from line 53, col 20.
                write(''' = _old_d''')
                _v = VFFSL(SL,"componentName",True) # u'${componentName}' on line 53, col 57
                if _v is not None: write(_filter(_v, rawExpr='${componentName}')) # from line 53, col 57.
                write('''_d''')
                _v = VFFSL(SL,"crossPropagationDimension",True) # u'${crossPropagationDimension}' on line 53, col 75
                if _v is not None: write(_filter(_v, rawExpr='${crossPropagationDimension}')) # from line 53, col 75.
                write(''';
''')
        write('''
''')
        crossDimRep = VFN(VFN(VFFSL(SL,"loopingField",True),"dimensionWithName",False)(VFFSL(SL,"crossPropagationDimension",True)),"inBasis",False)(VFFSL(SL,"operatorBasis",True))
        if VFFSL(SL,"crossPropagationDirection",True) == '+': # generated from line 58, col 3
            write('''if (''')
            _v = VFFSL(SL,"crossDimRep.loopIndex",True) # u'${crossDimRep.loopIndex}' on line 59, col 5
            if _v is not None: write(_filter(_v, rawExpr='${crossDimRep.loopIndex}')) # from line 59, col 5.
            write(''' == 0) {
''')
        else: # generated from line 60, col 3
            write('''if (''')
            _v = VFFSL(SL,"crossDimRep.loopIndex",True) # u'${crossDimRep.loopIndex}' on line 61, col 5
            if _v is not None: write(_filter(_v, rawExpr='${crossDimRep.loopIndex}')) # from line 61, col 5.
            write(''' == ''')
            _v = VFFSL(SL,"crossDimRep.globalLattice",True) # u'${crossDimRep.globalLattice}' on line 61, col 33
            if _v is not None: write(_filter(_v, rawExpr='${crossDimRep.globalLattice}')) # from line 61, col 33.
            write(''' - 1) {
''')
        write('''  // ********** Boundary condition code ***********
  ''')
        _v = VFN(VFFSL(SL,"codeBlocks",True)['boundaryCondition'],"loopCodeString",True) # u"${codeBlocks['boundaryCondition'].loopCodeString, autoIndent=True}" on line 64, col 3
        if _v is not None: write(_filter(_v, autoIndent=True, rawExpr="${codeBlocks['boundaryCondition'].loopCodeString, autoIndent=True}")) # from line 64, col 3.
        write('''  // **********************************************
  
''')
        for crossIntegrationVector in VFFSL(SL,"crossIntegrationVectors",True): # generated from line 67, col 3
            write('''  for (long _cmp = 0; _cmp < _''')
            _v = VFFSL(SL,"crossIntegrationVector.id",True) # u'${crossIntegrationVector.id}' on line 68, col 31
            if _v is not None: write(_filter(_v, rawExpr='${crossIntegrationVector.id}')) # from line 68, col 31.
            write('''_ncomponents; _cmp++)
    _old_''')
            _v = VFFSL(SL,"crossIntegrationVector.id",True) # u'${crossIntegrationVector.id}' on line 69, col 10
            if _v is not None: write(_filter(_v, rawExpr='${crossIntegrationVector.id}')) # from line 69, col 10.
            write('''[_cmp] = _active_''')
            _v = VFFSL(SL,"crossIntegrationVector.id",True) # u'${crossIntegrationVector.id}' on line 69, col 55
            if _v is not None: write(_filter(_v, rawExpr='${crossIntegrationVector.id}')) # from line 69, col 55.
            write('''[_''')
            _v = VFFSL(SL,"crossIntegrationVector.id",True) # u'${crossIntegrationVector.id}' on line 69, col 85
            if _v is not None: write(_filter(_v, rawExpr='${crossIntegrationVector.id}')) # from line 69, col 85.
            write('''_index_pointer + _cmp];
''')
        write('''  
''')
        #  This is where one (half-step) cross-IP step would go
        write('''} else {
  // Update the next guess for iteration.
''')
        for crossIntegrationVector in VFFSL(SL,"crossIntegrationVectors",True): # generated from line 75, col 3
            for componentNumber, componentName, in enumerate(crossIntegrationVector.components): # generated from line 76, col 5
                write('''  ''')
                _v = VFFSL(SL,"componentName",True) # u'${componentName}' on line 77, col 3
                if _v is not None: write(_filter(_v, rawExpr='${componentName}')) # from line 77, col 3.
                write(''' = _old_''')
                _v = VFFSL(SL,"crossIntegrationVector.id",True) # u'${crossIntegrationVector.id}' on line 77, col 27
                if _v is not None: write(_filter(_v, rawExpr='${crossIntegrationVector.id}')) # from line 77, col 27.
                write('''[''')
                _v = VFFSL(SL,"componentNumber",True) # u'${componentNumber}' on line 77, col 56
                if _v is not None: write(_filter(_v, rawExpr='${componentNumber}')) # from line 77, col 56.
                write('''] + d''')
                _v = VFFSL(SL,"componentName",True) # u'${componentName}' on line 77, col 79
                if _v is not None: write(_filter(_v, rawExpr='${componentName}')) # from line 77, col 79.
                write('''_d''')
                _v = VFFSL(SL,"crossPropagationDimension",True) # u'${crossPropagationDimension}' on line 77, col 97
                if _v is not None: write(_filter(_v, rawExpr='${crossPropagationDimension}')) # from line 77, col 97.
                write(''' * (''')
                _v = VFFSL(SL,"crossPropagationDirection",True) # u'${crossPropagationDirection}' on line 77, col 129
                if _v is not None: write(_filter(_v, rawExpr='${crossPropagationDirection}')) # from line 77, col 129.
                write('''0.5*d''')
                _v = VFFSL(SL,"crossPropagationDimension",True) # u'${crossPropagationDimension}' on line 77, col 162
                if _v is not None: write(_filter(_v, rawExpr='${crossPropagationDimension}')) # from line 77, col 162.
                write(''');
''')
        write('''}

for (long _iter = 0; _iter < ''')
        _v = VFFSL(SL,"iterations",True) # u'${iterations}' on line 82, col 30
        if _v is not None: write(_filter(_v, rawExpr='${iterations}')) # from line 82, col 30.
        write('''; _iter++) {
  
  #define d''')
        _v = VFFSL(SL,"propagationDimension",True) # u'${propagationDimension}' on line 84, col 12
        if _v is not None: write(_filter(_v, rawExpr='${propagationDimension}')) # from line 84, col 12.
        write(''' _step
  {
    // ************* Propagation code ***************
    ''')
        _v = VFFSL(SL,"codeString",True) # u'${codeString, autoIndent=True}' on line 87, col 5
        if _v is not None: write(_filter(_v, autoIndent=True, rawExpr='${codeString, autoIndent=True}')) # from line 87, col 5.
        write('''    // **********************************************
  }
  #undef d''')
        _v = VFFSL(SL,"propagationDimension",True) # u'${propagationDimension}' on line 90, col 11
        if _v is not None: write(_filter(_v, rawExpr='${propagationDimension}')) # from line 90, col 11.
        write('''
  
  {
    // *********** Cross-propagation code ***********
    ''')
        _v = VFN(VFFSL(SL,"codeBlocks",True)['crossPropagation'],"loopCodeString",True) # u"${codeBlocks['crossPropagation'].loopCodeString, autoIndent=True}" on line 94, col 5
        if _v is not None: write(_filter(_v, autoIndent=True, rawExpr="${codeBlocks['crossPropagation'].loopCodeString, autoIndent=True}")) # from line 94, col 5.
        write('''    // **********************************************
  }
  
  // Update propagation vectors (note that _step is actually half a step)
''')
        for integrationVector in VFFSL(SL,"integrationVectors",True): # generated from line 99, col 3
            for componentNumber, componentName in enumerate(integrationVector.components): # generated from line 100, col 5
                write('''  ''')
                _v = VFFSL(SL,"componentName",True) # u'${componentName}' on line 101, col 3
                if _v is not None: write(_filter(_v, rawExpr='${componentName}')) # from line 101, col 3.
                write(''' = _''')
                _v = VFFSL(SL,"integrator.name",True) # u'${integrator.name}' on line 101, col 23
                if _v is not None: write(_filter(_v, rawExpr='${integrator.name}')) # from line 101, col 23.
                write('''_oldcopy_''')
                _v = VFFSL(SL,"integrationVector.id",True) # u'${integrationVector.id}' on line 101, col 50
                if _v is not None: write(_filter(_v, rawExpr='${integrationVector.id}')) # from line 101, col 50.
                write('''[_''')
                _v = VFFSL(SL,"integrationVector.id",True) # u'${integrationVector.id}' on line 101, col 75
                if _v is not None: write(_filter(_v, rawExpr='${integrationVector.id}')) # from line 101, col 75.
                write('''_index_pointer + ''')
                _v = VFFSL(SL,"componentNumber",True) # u'${componentNumber}' on line 101, col 115
                if _v is not None: write(_filter(_v, rawExpr='${componentNumber}')) # from line 101, col 115.
                write('''] + d''')
                _v = VFFSL(SL,"componentName",True) # u'${componentName}' on line 101, col 138
                if _v is not None: write(_filter(_v, rawExpr='${componentName}')) # from line 101, col 138.
                write('''_d''')
                _v = VFFSL(SL,"propagationDimension",True) # u'${propagationDimension}' on line 101, col 156
                if _v is not None: write(_filter(_v, rawExpr='${propagationDimension}')) # from line 101, col 156.
                write(''' * _step;
''')
        write('''  
  // Update cross-propagation vectors
''')
        for crossIntegrationVector in VFFSL(SL,"crossIntegrationVectors",True): # generated from line 106, col 3
            for componentNumber, componentName in enumerate(VFFSL(SL,"crossIntegrationVector.components",True)): # generated from line 107, col 5
                write('''  ''')
                _v = VFFSL(SL,"componentName",True) # u'${componentName}' on line 108, col 3
                if _v is not None: write(_filter(_v, rawExpr='${componentName}')) # from line 108, col 3.
                write(''' = _old_''')
                _v = VFFSL(SL,"crossIntegrationVector.id",True) # u'${crossIntegrationVector.id}' on line 108, col 27
                if _v is not None: write(_filter(_v, rawExpr='${crossIntegrationVector.id}')) # from line 108, col 27.
                write('''[''')
                _v = VFFSL(SL,"componentNumber",True) # u'${componentNumber}' on line 108, col 56
                if _v is not None: write(_filter(_v, rawExpr='${componentNumber}')) # from line 108, col 56.
                write('''] + d''')
                _v = VFFSL(SL,"componentName",True) # u'${componentName}' on line 108, col 79
                if _v is not None: write(_filter(_v, rawExpr='${componentName}')) # from line 108, col 79.
                write('''_d''')
                _v = VFFSL(SL,"crossPropagationDimension",True) # u'${crossPropagationDimension}' on line 108, col 97
                if _v is not None: write(_filter(_v, rawExpr='${crossPropagationDimension}')) # from line 108, col 97.
                write(''' * (''')
                _v = VFFSL(SL,"crossPropagationDirection",True) # u'${crossPropagationDirection}' on line 108, col 129
                if _v is not None: write(_filter(_v, rawExpr='${crossPropagationDirection}')) # from line 108, col 129.
                write('''0.5*d''')
                _v = VFFSL(SL,"crossPropagationDimension",True) # u'${crossPropagationDimension}' on line 108, col 162
                if _v is not None: write(_filter(_v, rawExpr='${crossPropagationDimension}')) # from line 108, col 162.
                write(''');
''')
        write("""}

// Update the 'old' copy for the next half-step
""")
        for crossIntegrationVector in VFFSL(SL,"crossIntegrationVectors",True): # generated from line 114, col 1
            for componentNumber, componentName in enumerate(crossIntegrationVector.components): # generated from line 115, col 3
                write('''_old_''')
                _v = VFFSL(SL,"crossIntegrationVector.id",True) # u'${crossIntegrationVector.id}' on line 116, col 6
                if _v is not None: write(_filter(_v, rawExpr='${crossIntegrationVector.id}')) # from line 116, col 6.
                write('''[''')
                _v = VFFSL(SL,"componentNumber",True) # u'${componentNumber}' on line 116, col 35
                if _v is not None: write(_filter(_v, rawExpr='${componentNumber}')) # from line 116, col 35.
                write('''] += d''')
                _v = VFFSL(SL,"componentName",True) # u'${componentName}' on line 116, col 59
                if _v is not None: write(_filter(_v, rawExpr='${componentName}')) # from line 116, col 59.
                write('''_d''')
                _v = VFFSL(SL,"crossPropagationDimension",True) # u'${crossPropagationDimension}' on line 116, col 77
                if _v is not None: write(_filter(_v, rawExpr='${crossPropagationDimension}')) # from line 116, col 77.
                write(''' * (''')
                _v = VFFSL(SL,"crossPropagationDirection",True) # u'${crossPropagationDirection}' on line 116, col 109
                if _v is not None: write(_filter(_v, rawExpr='${crossPropagationDirection}')) # from line 116, col 109.
                write('''d''')
                _v = VFFSL(SL,"crossPropagationDimension",True) # u'${crossPropagationDimension}' on line 116, col 138
                if _v is not None: write(_filter(_v, rawExpr='${crossPropagationDimension}')) # from line 116, col 138.
                write(''');
''')
        write('''
''')
        #  This is where one (full step) cross-IP step would go
        write('''
''')
        for crossIntegrationVector in VFFSL(SL,"crossIntegrationVectors",True): # generated from line 122, col 3
            for componentName in crossIntegrationVector.components: # generated from line 123, col 5
                write('''_old_d''')
                _v = VFFSL(SL,"componentName",True) # u'${componentName}' on line 124, col 7
                if _v is not None: write(_filter(_v, rawExpr='${componentName}')) # from line 124, col 7.
                write('''_d''')
                _v = VFFSL(SL,"crossPropagationDimension",True) # u'${crossPropagationDimension}' on line 124, col 25
                if _v is not None: write(_filter(_v, rawExpr='${crossPropagationDimension}')) # from line 124, col 25.
                write(''' = d''')
                _v = VFFSL(SL,"componentName",True) # u'${componentName}' on line 124, col 57
                if _v is not None: write(_filter(_v, rawExpr='${componentName}')) # from line 124, col 57.
                write('''_d''')
                _v = VFFSL(SL,"crossPropagationDimension",True) # u'${crossPropagationDimension}' on line 124, col 75
                if _v is not None: write(_filter(_v, rawExpr='${crossPropagationDimension}')) # from line 124, col 75.
                write(''';
''')
        write('''
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def evaluateOperatorFunctionContentsWithCodeBlock(self, function, **KWS):



        ## CHEETAH: generated from @def evaluateOperatorFunctionContentsWithCodeBlock($function) at line 131, col 1.
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
        #  We shouldn't have a deltaAField. It doesn't work with cross-propagation.
        assert not VFFSL(SL,"deltaAField",True)
        # 
        for crossIntegrationVector in VFFSL(SL,"crossIntegrationVectors",True): # generated from line 136, col 3
            _v = VFFSL(SL,"crossIntegrationVector.type",True) # u'${crossIntegrationVector.type}' on line 137, col 1
            if _v is not None: write(_filter(_v, rawExpr='${crossIntegrationVector.type}')) # from line 137, col 1.
            write(''' _old_''')
            _v = VFFSL(SL,"crossIntegrationVector.id",True) # u'${crossIntegrationVector.id}' on line 137, col 37
            if _v is not None: write(_filter(_v, rawExpr='${crossIntegrationVector.id}')) # from line 137, col 37.
            write('''[_''')
            _v = VFFSL(SL,"crossIntegrationVector.id",True) # u'${crossIntegrationVector.id}' on line 137, col 67
            if _v is not None: write(_filter(_v, rawExpr='${crossIntegrationVector.id}')) # from line 137, col 67.
            write('''_ncomponents];
''')
        # 
        _v = super(SICDeltaAOperator, self).evaluateOperatorFunctionContentsWithCodeBlock(function)
        if _v is not None: write(_filter(_v))
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
        # SICDeltaAOperator.tmpl
        # 
        # delta-a operator for the left/right propagation in the SIC integrator.
        # 
        # Created by Graham Dennis on 2008-08-07.
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

    _mainCheetahMethod_for_SICDeltaAOperator= 'writeBody'

## END CLASS DEFINITION

if not hasattr(SICDeltaAOperator, '_initCheetahAttributes'):
    templateAPIClass = getattr(SICDeltaAOperator, '_CHEETAH_templateClass', Template)
    templateAPIClass._addCheetahPlumbingCodeToClass(SICDeltaAOperator)


# CHEETAH was developed by Tavis Rudd and Mike Orr
# with code, advice and input from many other volunteers.
# For more information visit http://www.CheetahTemplate.org/

##################################################
## if run from command line:
if __name__ == '__main__':
    from Cheetah.TemplateCmdLineIface import CmdLineIface
    CmdLineIface(templateObj=SICDeltaAOperator()).run()

