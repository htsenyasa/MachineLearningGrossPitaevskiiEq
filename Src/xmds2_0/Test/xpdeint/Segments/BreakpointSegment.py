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
from xpdeint.Segments._BreakpointSegment import _BreakpointSegment
from xpdeint.CallOnceGuards import callOnceGuard

##################################################
## MODULE CONSTANTS
VFFSL=valueFromFrameOrSearchList
VFSL=valueFromSearchList
VFN=valueForName
currentTime=time.time
__CHEETAH_version__ = '2.4.4'
__CHEETAH_versionTuple__ = (2, 4, 4, 'development', 0)
__CHEETAH_genTime__ = 1484975072.063163
__CHEETAH_genTimestamp__ = 'Sat Jan 21 16:04:32 2017'
__CHEETAH_src__ = '/home/mattias/xmds-2.2.3/admin/staging/xmds-2.2.3/xpdeint/Segments/BreakpointSegment.tmpl'
__CHEETAH_srcLastModified__ = 'Wed Aug  1 11:52:34 2012'
__CHEETAH_docstring__ = 'Autogenerated by Cheetah: The Python-Powered Template Engine'

if __CHEETAH_versionTuple__ < RequiredCheetahVersionTuple:
    raise AssertionError(
      'This template was compiled with Cheetah version'
      ' %s. Templates compiled before version %s must be recompiled.'%(
         __CHEETAH_version__, RequiredCheetahVersion))

##################################################
## CLASSES

class BreakpointSegment(_BreakpointSegment):

    ##################################################
    ## CHEETAH GENERATED METHODS


    def __init__(self, *args, **KWs):

        super(BreakpointSegment, self).__init__(*args, **KWs)
        if not self._CHEETAH__instanceInitialized:
            cheetahKWArgs = {}
            allowedKWs = 'searchList namespaces filter filtersLib errorCatcher'.split()
            for k,v in list(KWs.items()):
                if k in allowedKWs: cheetahKWArgs[k] = v
            self._initCheetahInstance(**cheetahKWArgs)
        

    def description(self, **KWS):



        ## Generated from @def description: segment $segmentNumber (Breakpoint) at line 28, col 1.
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
        
        write('''segment ''')
        _v = VFFSL(SL,"segmentNumber",True) # u'$segmentNumber' on line 28, col 27
        if _v is not None: write(_filter(_v, rawExpr='$segmentNumber')) # from line 28, col 27.
        write(''' (Breakpoint)''')
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    @callOnceGuard
    def static_globals(self, **KWS):



        ## CHEETAH: generated from @def static_globals at line 34, col 1.
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
        _v = super(BreakpointSegment, self).static_globals()
        if _v is not None: write(_filter(_v))
        # 
        write('''long _breakpointAutoNameCounter = 0;
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def segmentFunctionBody(self, function, **KWS):



        ## CHEETAH: generated from @def segmentFunctionBody($function) at line 45, col 1.
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
        
        _v = VFFSL(SL,"evaluateComputedVectors",False)(VFFSL(SL,"dependencies",True)) # u'${evaluateComputedVectors($dependencies)}' on line 46, col 1
        if _v is not None: write(_filter(_v, rawExpr='${evaluateComputedVectors($dependencies)}')) # from line 46, col 1.
        write('''
''')
        _v = VFFSL(SL,"transformVectorsToBasis",False)(VFFSL(SL,"dependencies",True), VFFSL(SL,"breakpointBasis",True)) # u'${transformVectorsToBasis($dependencies, $breakpointBasis)}' on line 48, col 1
        if _v is not None: write(_filter(_v, rawExpr='${transformVectorsToBasis($dependencies, $breakpointBasis)}')) # from line 48, col 1.
        write('''
''')
        featureOrdering = ['Driver']
        dict = {'extraIndent': 0}
        _v = VFFSL(SL,"insertCodeForFeatures",False)('breakpointBegin', featureOrdering, dict) # u"${insertCodeForFeatures('breakpointBegin', featureOrdering, dict)}" on line 52, col 1
        if _v is not None: write(_filter(_v, rawExpr="${insertCodeForFeatures('breakpointBegin', featureOrdering, dict)}")) # from line 52, col 1.
        extraIndent = dict['extraIndent']
        write('''
''')
        _v = VFFSL(SL,"breakpointFunctionContents",True) # u'${breakpointFunctionContents, extraIndent=extraIndent}' on line 55, col 1
        if _v is not None: write(_filter(_v, extraIndent=extraIndent, rawExpr='${breakpointFunctionContents, extraIndent=extraIndent}')) # from line 55, col 1.
        write('''
''')
        _v = VFFSL(SL,"insertCodeForFeaturesInReverseOrder",False)('breakpointEnd', VFFSL(SL,"featureOrdering",True), dict) # u"${insertCodeForFeaturesInReverseOrder('breakpointEnd', $featureOrdering, dict)}" on line 57, col 1
        if _v is not None: write(_filter(_v, rawExpr="${insertCodeForFeaturesInReverseOrder('breakpointEnd', $featureOrdering, dict)}")) # from line 57, col 1.
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def breakpointFunctionContents(self, **KWS):



        ## CHEETAH: generated from @def breakpointFunctionContents at line 61, col 1.
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
        write('''char *_baseFilename = (char*)malloc(255);
''')
        if not VFFSL(SL,"filename",True): # generated from line 64, col 3
            #  If we don't have a filename, then we are auto-naming
            write('''_breakpointAutoNameCounter++;
snprintf(_baseFilename, 255, "%li%s", _breakpointAutoNameCounter, gsArgsAndValues.c_str());
''')
        else: # generated from line 68, col 3
            #  We have a name, rip off the extension if its 'xsil'
            baseFilename = VFFSL(SL,"filename",True)
            if baseFilename.endswith('.xsil'): # generated from line 71, col 5
                baseFilename = baseFilename[0:-5]
            write('''snprintf(_baseFilename, 255, "%s%s", "''')
            _v = VFFSL(SL,"baseFilename",True) # u'$baseFilename' on line 74, col 39
            if _v is not None: write(_filter(_v, rawExpr='$baseFilename')) # from line 74, col 39.
            write('''", gsArgsAndValues.c_str());
''')
        write('''
''')
        featureOrdering = ['Output']
        _v = VFN(VFFSL(SL,"outputFormat",True),"writeOutSetup",False)('_baseFilename', self) # u"${outputFormat.writeOutSetup('_baseFilename', self)}" on line 78, col 1
        if _v is not None: write(_filter(_v, rawExpr="${outputFormat.writeOutSetup('_baseFilename', self)}")) # from line 78, col 1.
        write('''
''')
        dependentVariables = [{'vector': vector, 'arrayName': ''.join(['_active_',str(VFFSL(SL,"vector.id",True))]), 'components': vector.components} for vector in VFFSL(SL,"dependencies",True)]
        writeOutDict = {'field': VFFSL(SL,"field",True),                        'basis': VFFSL(SL,"breakpointBasis",True),                        'fp': '_outfile',                        'baseFilename': "_baseFilename",                        'outputGroupFilenameSuffix': '',                        'dependentVariables': dependentVariables,                        'xsilElementName': "breakpoint",                        'groupID': 1                       }
        _v = VFN(VFFSL(SL,"outputFormat",True),"writeOutFunctionImplementationBody",False)(writeOutDict) # u'${outputFormat.writeOutFunctionImplementationBody(writeOutDict)}' on line 90, col 1
        if _v is not None: write(_filter(_v, rawExpr='${outputFormat.writeOutFunctionImplementationBody(writeOutDict)}')) # from line 90, col 1.
        write('''
''')
        _v = VFFSL(SL,"outputFormat.writeOutTearDown",True) # u'${outputFormat.writeOutTearDown}' on line 92, col 1
        if _v is not None: write(_filter(_v, rawExpr='${outputFormat.writeOutTearDown}')) # from line 92, col 1.
        write('''
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
        # BreakpointSegment.tmpl
        # 
        # Created by Graham Dennis on 2008-03-15.
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
        # 
        #   Description of template
        write('''
''')
        # 
        #   Static globals
        write('''
''')
        # 
        #   Function implementations
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

    _mainCheetahMethod_for_BreakpointSegment= 'writeBody'

## END CLASS DEFINITION

if not hasattr(BreakpointSegment, '_initCheetahAttributes'):
    templateAPIClass = getattr(BreakpointSegment, '_CHEETAH_templateClass', Template)
    templateAPIClass._addCheetahPlumbingCodeToClass(BreakpointSegment)


# CHEETAH was developed by Tavis Rudd and Mike Orr
# with code, advice and input from many other volunteers.
# For more information visit http://www.CheetahTemplate.org/

##################################################
## if run from command line:
if __name__ == '__main__':
    from Cheetah.TemplateCmdLineIface import CmdLineIface
    CmdLineIface(templateObj=BreakpointSegment()).run()

