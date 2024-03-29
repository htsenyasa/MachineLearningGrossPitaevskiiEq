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
from xpdeint.Features.OutputFormat import OutputFormat

##################################################
## MODULE CONSTANTS
VFFSL=valueFromFrameOrSearchList
VFSL=valueFromSearchList
VFN=valueForName
currentTime=time.time
__CHEETAH_version__ = '2.4.4'
__CHEETAH_versionTuple__ = (2, 4, 4, 'development', 0)
__CHEETAH_genTime__ = 1484975071.200571
__CHEETAH_genTimestamp__ = 'Sat Jan 21 16:04:31 2017'
__CHEETAH_src__ = '/home/mattias/xmds-2.2.3/admin/staging/xmds-2.2.3/xpdeint/Features/AsciiFormat.tmpl'
__CHEETAH_srcLastModified__ = 'Mon Jul 23 09:42:26 2012'
__CHEETAH_docstring__ = 'Autogenerated by Cheetah: The Python-Powered Template Engine'

if __CHEETAH_versionTuple__ < RequiredCheetahVersionTuple:
    raise AssertionError(
      'This template was compiled with Cheetah version'
      ' %s. Templates compiled before version %s must be recompiled.'%(
         __CHEETAH_version__, RequiredCheetahVersion))

##################################################
## CLASSES

class AsciiFormat(OutputFormat):

    ##################################################
    ## CHEETAH GENERATED METHODS


    def __init__(self, *args, **KWs):

        super(AsciiFormat, self).__init__(*args, **KWs)
        if not self._CHEETAH__instanceInitialized:
            cheetahKWArgs = {}
            allowedKWs = 'searchList namespaces filter filtersLib errorCatcher'.split()
            for k,v in list(KWs.items()):
                if k in allowedKWs: cheetahKWArgs[k] = v
            self._initCheetahInstance(**cheetahKWArgs)
        

    def description(self, **KWS):



        ## Generated from @def description: ascii output format at line 24, col 1.
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
        
        write('''ascii output format''')
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def writeOutFunctionImplementationBody(self, dict, **KWS):



        ## CHEETAH: generated from @def writeOutFunctionImplementationBody($dict) at line 30, col 1.
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
        
        #  The function prototype in which this code is going is:
        #   void _mgN_write_out(FILE* _outfile)
        #  And this code is being automatically indented.
        # 
        fp = dict['fp']
        field = dict['field']
        basis = dict['basis']
        # 
        _v = VFFSL(SL,"writeOutFunctionImplementationBegin",False)(dict) # u'${writeOutFunctionImplementationBegin(dict)}' on line 39, col 1
        if _v is not None: write(_filter(_v, rawExpr='${writeOutFunctionImplementationBegin(dict)}')) # from line 39, col 1.
        write('''  
''')
        dependentVariables = VFFSL(SL,"dict.dependentVariables",True)
        write('''fprintf(''')
        _v = VFFSL(SL,"fp",True) # u'$fp' on line 42, col 9
        if _v is not None: write(_filter(_v, rawExpr='$fp')) # from line 42, col 9.
        write(''', "    <Stream><Metalink Format=\\"Text\\" Delimiter=\\" \\\\n\\"/>\\n");

''')
        vectors = set([variable['vector'] for variable in dependentVariables])
        _v = VFFSL(SL,"loopOverFieldInBasisWithVectorsAndInnerContent",False)(field, basis, vectors, VFFSL(SL,"insideOutputLoops",False)(dict), loopingOrder=VFFSL(SL,"LoopingOrder.StrictlyAscendingOrder",True), vectorsNotNeedingDefines=vectors) # u'$loopOverFieldInBasisWithVectorsAndInnerContent(field, basis, vectors, $insideOutputLoops(dict), $loopingOrder=$LoopingOrder.StrictlyAscendingOrder, vectorsNotNeedingDefines=vectors)' on line 45, col 1
        if _v is not None: write(_filter(_v, rawExpr='$loopOverFieldInBasisWithVectorsAndInnerContent(field, basis, vectors, $insideOutputLoops(dict), $loopingOrder=$LoopingOrder.StrictlyAscendingOrder, vectorsNotNeedingDefines=vectors)')) # from line 45, col 1.
        write('''
fprintf(''')
        _v = VFFSL(SL,"fp",True) # u'$fp' on line 47, col 9
        if _v is not None: write(_filter(_v, rawExpr='$fp')) # from line 47, col 9.
        write(''', "    </Stream>\\n");

''')
        _v = VFFSL(SL,"writeOutFunctionImplementationEnd",False)(dict) # u'${writeOutFunctionImplementationEnd(dict)}' on line 49, col 1
        if _v is not None: write(_filter(_v, rawExpr='${writeOutFunctionImplementationEnd(dict)}')) # from line 49, col 1.
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def insideOutputLoops(self, dict, **KWS):



        ## CHEETAH: generated from @def insideOutputLoops(dict) at line 53, col 1.
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
        fp = dict['fp']
        dependentVariables = dict['dependentVariables']
        field = dict['field']
        basis = dict['basis']
        # 
        componentCount = 0
        for variable in VFFSL(SL,"dependentVariables",True): # generated from line 61, col 3
            componentCount += len(VFFSL(SL,"variable.vector.components",True))
            if VFFSL(SL,"variable.vector.type",True) == 'complex': # generated from line 63, col 5
                componentCount += len(VFFSL(SL,"variable.vector.components",True))
        # 
        dict['componentCount'] = componentCount
        variableCount = len(VFFSL(SL,"field.dimensions",True)) + componentCount
        precisionPrefix = '.12'
        if VFFSL(SL,"precision",True) == 'single': # generated from line 71, col 3
            precisionPrefix = ''
        write('''// UNVECTORISABLE
fprintf(''')
        _v = VFFSL(SL,"fp",True) # u'$fp' on line 75, col 9
        if _v is not None: write(_filter(_v, rawExpr='$fp')) # from line 75, col 9.
        write(''', "''')
        #  Loop over the variables that we're writing out
        for variableNumber in range(VFFSL(SL,"variableCount",True)): # generated from line 77, col 3
            if VFFSL(SL,"variableNumber",True) != 0: # generated from line 78, col 5
                #  If this isn't the first dimension, include a space at the start
                write(''' %''')
                _v = VFFSL(SL,"precisionPrefix",True) # u'${precisionPrefix}' on line 80, col 3
                if _v is not None: write(_filter(_v, rawExpr='${precisionPrefix}')) # from line 80, col 3.
                write('''e''')
            else: # generated from line 81, col 5
                write('''%''')
                _v = VFFSL(SL,"precisionPrefix",True) # u'${precisionPrefix}' on line 82, col 2
                if _v is not None: write(_filter(_v, rawExpr='${precisionPrefix}')) # from line 82, col 2.
                write('''e''')
        write('''\\n"''')
        # 
        for dimension in VFFSL(SL,"field.dimensions",True): # generated from line 87, col 3
            write(''', (real)''')
            _v = VFN(VFN(VFFSL(SL,"dimension",True),"inBasis",False)(basis),"name",True) # u'${dimension.inBasis(basis).name}' on line 88, col 9
            if _v is not None: write(_filter(_v, rawExpr='${dimension.inBasis(basis).name}')) # from line 88, col 9.
        # 
        #  Now loop over the (dependent) variables
        for variable in VFFSL(SL,"dependentVariables",True): # generated from line 92, col 3
            for componentNumber, component in enumerate(VFFSL(SL,"variable.components",True)): # generated from line 93, col 5
                if VFFSL(SL,"variable.vector.type",True) == 'real': # generated from line 94, col 7
                    write(''', ''')
                    _v = VFFSL(SL,"variable.arrayName",True) # u'${variable.arrayName}' on line 95, col 3
                    if _v is not None: write(_filter(_v, rawExpr='${variable.arrayName}')) # from line 95, col 3.
                    write('''[_''')
                    _v = VFFSL(SL,"variable.vector.id",True) # u'${variable.vector.id}' on line 95, col 26
                    if _v is not None: write(_filter(_v, rawExpr='${variable.vector.id}')) # from line 95, col 26.
                    write('''_index_pointer + ''')
                    _v = VFFSL(SL,"componentNumber",True) # u'${componentNumber}' on line 95, col 64
                    if _v is not None: write(_filter(_v, rawExpr='${componentNumber}')) # from line 95, col 64.
                    write(''']''')
                else: # generated from line 96, col 7
                    write(''', ''')
                    _v = VFFSL(SL,"variable.arrayName",True) # u'${variable.arrayName}' on line 97, col 3
                    if _v is not None: write(_filter(_v, rawExpr='${variable.arrayName}')) # from line 97, col 3.
                    write('''[_''')
                    _v = VFFSL(SL,"variable.vector.id",True) # u'${variable.vector.id}' on line 97, col 26
                    if _v is not None: write(_filter(_v, rawExpr='${variable.vector.id}')) # from line 97, col 26.
                    write('''_index_pointer + ''')
                    _v = VFFSL(SL,"componentNumber",True) # u'${componentNumber}' on line 97, col 64
                    if _v is not None: write(_filter(_v, rawExpr='${componentNumber}')) # from line 97, col 64.
                    write('''].Re(), ''')
                    _v = VFFSL(SL,"variable.arrayName",True) # u'${variable.arrayName}' on line 97, col 90
                    if _v is not None: write(_filter(_v, rawExpr='${variable.arrayName}')) # from line 97, col 90.
                    write('''[_''')
                    _v = VFFSL(SL,"variable.vector.id",True) # u'${variable.vector.id}' on line 97, col 113
                    if _v is not None: write(_filter(_v, rawExpr='${variable.vector.id}')) # from line 97, col 113.
                    write('''_index_pointer + ''')
                    _v = VFFSL(SL,"componentNumber",True) # u'${componentNumber}' on line 97, col 151
                    if _v is not None: write(_filter(_v, rawExpr='${componentNumber}')) # from line 97, col 151.
                    write('''].Im()''')
        write(''');
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
        # AsciiFormat.tmpl
        # 
        # Created by Graham Dennis on 2007-09-18.
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
        #   Write out the data in ASCII format
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

    name = 'ascii'

    _mainCheetahMethod_for_AsciiFormat= 'writeBody'

## END CLASS DEFINITION

if not hasattr(AsciiFormat, '_initCheetahAttributes'):
    templateAPIClass = getattr(AsciiFormat, '_CHEETAH_templateClass', Template)
    templateAPIClass._addCheetahPlumbingCodeToClass(AsciiFormat)


# CHEETAH was developed by Tavis Rudd and Mike Orr
# with code, advice and input from many other volunteers.
# For more information visit http://www.CheetahTemplate.org/

##################################################
## if run from command line:
if __name__ == '__main__':
    from Cheetah.TemplateCmdLineIface import CmdLineIface
    CmdLineIface(templateObj=AsciiFormat()).run()


