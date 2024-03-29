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
from xpdeint.ScriptElement import ScriptElement
from xpdeint.PrintfSafeFilter import PrintfSafeFilter
from xpdeint.CallOnceGuards import callOnceGuard

##################################################
## MODULE CONSTANTS
VFFSL=valueFromFrameOrSearchList
VFSL=valueFromSearchList
VFN=valueForName
currentTime=time.time
__CHEETAH_version__ = '2.4.4'
__CHEETAH_versionTuple__ = (2, 4, 4, 'development', 0)
__CHEETAH_genTime__ = 1484975072.775341
__CHEETAH_genTimestamp__ = 'Sat Jan 21 16:04:32 2017'
__CHEETAH_src__ = '/home/mattias/xmds-2.2.3/admin/staging/xmds-2.2.3/xpdeint/Stochastic/Generators/Generator.tmpl'
__CHEETAH_srcLastModified__ = 'Thu Jan  5 14:39:24 2017'
__CHEETAH_docstring__ = 'Autogenerated by Cheetah: The Python-Powered Template Engine'

if __CHEETAH_versionTuple__ < RequiredCheetahVersionTuple:
    raise AssertionError(
      'This template was compiled with Cheetah version'
      ' %s. Templates compiled before version %s must be recompiled.'%(
         __CHEETAH_version__, RequiredCheetahVersion))

##################################################
## CLASSES

class Generator(ScriptElement):

    ##################################################
    ## CHEETAH GENERATED METHODS


    def __init__(self, *args, **KWs):

        super(Generator, self).__init__(*args, **KWs)
        if not self._CHEETAH__instanceInitialized:
            cheetahKWArgs = {}
            allowedKWs = 'searchList namespaces filter filtersLib errorCatcher'.split()
            for k,v in list(KWs.items()):
                if k in allowedKWs: cheetahKWArgs[k] = v
            self._initCheetahInstance(**cheetahKWArgs)
        

    def seedCount(self, **KWS):



        ## CHEETAH: generated from @def seedCount at line 26, col 1.
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
        
        return len(VFFSL(SL,"seedArray",True)) or 10
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

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
        write('''uint32_t ''')
        _v = VFFSL(SL,"generatorName",True) # u'${generatorName}' on line 32, col 10
        if _v is not None: write(_filter(_v, rawExpr='${generatorName}')) # from line 32, col 10.
        write('''_seeds[''')
        _v = VFFSL(SL,"seedCount",True) # u'${seedCount}' on line 32, col 33
        if _v is not None: write(_filter(_v, rawExpr='${seedCount}')) # from line 32, col 33.
        write('''];
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def initialiseGlobalSeeds(self, **KWS):



        ## CHEETAH: generated from @def initialiseGlobalSeeds at line 36, col 1.
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
        featureOrdering = ['Driver']
        # 
        if not VFFSL(SL,"seedArray",True): # generated from line 40, col 3
            _v = VFFSL(SL,"seedSystemRandomNumberGenerator",True) # u'${seedSystemRandomNumberGenerator}' on line 41, col 1
            if _v is not None: write(_filter(_v, rawExpr='${seedSystemRandomNumberGenerator}')) # from line 41, col 1.
            # 
            seedGenerationDict = {'extraIndent': 0}
            _v = VFFSL(SL,"insertCodeForFeatures",False)('runtimeSeedGenerationBegin', VFFSL(SL,"featureOrdering",True), VFFSL(SL,"seedGenerationDict",True)) # u"${insertCodeForFeatures('runtimeSeedGenerationBegin', $featureOrdering, $seedGenerationDict)}" on line 44, col 1
            if _v is not None: write(_filter(_v, rawExpr="${insertCodeForFeatures('runtimeSeedGenerationBegin', $featureOrdering, $seedGenerationDict)}")) # from line 44, col 1.
            extraIndent = seedGenerationDict['extraIndent']
            _v = VFFSL(SL,"insertCodeForFeaturesInReverseOrder",False)('runtimeSeedGenerationEnd', VFFSL(SL,"featureOrdering",True), VFFSL(SL,"seedGenerationDict",True)) # u"${insertCodeForFeaturesInReverseOrder('runtimeSeedGenerationEnd', $featureOrdering, $seedGenerationDict)}" on line 46, col 1
            if _v is not None: write(_filter(_v, rawExpr="${insertCodeForFeaturesInReverseOrder('runtimeSeedGenerationEnd', $featureOrdering, $seedGenerationDict)}")) # from line 46, col 1.
        else: # generated from line 47, col 3
            for seedIdx, seed in enumerate(VFFSL(SL,"seedArray",True)): # generated from line 48, col 5
                _v = VFFSL(SL,"generatorName",True) # u'${generatorName}' on line 49, col 1
                if _v is not None: write(_filter(_v, rawExpr='${generatorName}')) # from line 49, col 1.
                write('''_seeds[''')
                _v = VFFSL(SL,"seedIdx",True) # u'$seedIdx' on line 49, col 24
                if _v is not None: write(_filter(_v, rawExpr='$seedIdx')) # from line 49, col 24.
                write('''] = ''')
                _v = VFFSL(SL,"seed",True) # u'$seed' on line 49, col 36
                if _v is not None: write(_filter(_v, rawExpr='$seed')) # from line 49, col 36.
                write(''';
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    @callOnceGuard
    def seedSystemRandomNumberGenerator(self, **KWS):



        ## CHEETAH: generated from @def seedSystemRandomNumberGenerator at line 57, col 1.
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
        
        write('''#if HAVE_DEV_URANDOM
  uint32_t __seeds[10];
  FILE *__urandom_fp = fopen("/dev/urandom", "r");

  if (__urandom_fp == NULL) {
      _LOG(_ERROR_LOG_LEVEL, "Unable to seed random number generator from /dev/urandom.  Is it accessible?\\n");
      // Implicit quit
  }

  size_t __entries_read = 0;
  __entries_read = fread(__seeds, sizeof(uint32_t), 10, __urandom_fp);

  if (__entries_read != 10) {
    _LOG(_ERROR_LOG_LEVEL, "Unable to read from /dev/urandom while seeding the random number generator.\\n");
      // Implicit quit
  }

  fclose(__urandom_fp);

  for (unsigned long _i0=0; _i0 < ''')
        _v = VFFSL(SL,"seedCount",True) # u'${seedCount}' on line 77, col 35
        if _v is not None: write(_filter(_v, rawExpr='${seedCount}')) # from line 77, col 35.
        write('''; _i0++) {
    ''')
        _v = VFFSL(SL,"generatorName",True) # u'${generatorName}' on line 78, col 5
        if _v is not None: write(_filter(_v, rawExpr='${generatorName}')) # from line 78, col 5.
        write('''_seeds[_i0] = (uint32_t) __seeds[_i0];
  }

#else
#error Do not have a run-time random number source! Please supply seeds manually.
#endif
''')
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def initialiseLocalSeeds(self, **KWS):



        ## CHEETAH: generated from @def initialiseLocalSeeds at line 86, col 1.
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
        featureOrdering = ['Driver']
        seedOffset = VFFSL(SL,"insertCodeForFeatures",False)('seedOffset', VFFSL(SL,"featureOrdering",True))
        # 
        write('''uint32_t ''')
        _v = VFFSL(SL,"generatorName",True) # u'${generatorName}' on line 91, col 10
        if _v is not None: write(_filter(_v, rawExpr='${generatorName}')) # from line 91, col 10.
        write('''_local_seeds[''')
        _v = VFFSL(SL,"seedCount",True) # u'${seedCount}' on line 91, col 39
        if _v is not None: write(_filter(_v, rawExpr='${seedCount}')) # from line 91, col 39.
        write('''] = {
  ''')
        _v = ',\n  '.join([''.join([str(VFFSL(SL,"generatorName",True)),'_seeds[',str(VFFSL(SL,"i",True)),']+(0',str(VFFSL(SL,"seedOffset",True)),')*',str(VFFSL(SL,"i",True)+1)]) for i in range(VFFSL(SL,"seedCount",True))]) # u"${',\\n  '.join([c'${generatorName}_seeds[$i]+(0${seedOffset})*${i+1}' for i in xrange($seedCount)])}" on line 92, col 3
        if _v is not None: write(_filter(_v, rawExpr="${',\\n  '.join([c'${generatorName}_seeds[$i]+(0${seedOffset})*${i+1}' for i in xrange($seedCount)])}")) # from line 92, col 3.
        write('''
};
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def xsilOutputInfo(self, dict, **KWS):


        """
        Write to the XSIL file lines naming the seeds that we generated if no seed was provided
        in the script file. These are the seeds that should be provided in the script file to get
        the same results.
        """

        ## CHEETAH: generated from @def xsilOutputInfo($dict) at line 97, col 1.
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
        # 
        if len(VFFSL(SL,"seedArray",True)): # generated from line 106, col 3
            return
        # 
        write('''fprintf(''')
        _v = VFFSL(SL,"fp",True) # u'$fp' on line 110, col 9
        if _v is not None: write(_filter(_v, rawExpr='$fp')) # from line 110, col 9.
        write(''', "\\nNo seeds were provided for noise vector \'''')
        _v = VFFSL(SL,"parent.parent.name",True) # u'${parent.parent.name}' on line 110, col 58
        if _v is not None: write(_filter(_v, rawExpr='${parent.parent.name}')) # from line 110, col 58.
        write('''\'. The seeds generated were:\\n");
fprintf(''')
        _v = VFFSL(SL,"fp",True) # u'$fp' on line 111, col 9
        if _v is not None: write(_filter(_v, rawExpr='$fp')) # from line 111, col 9.
        write(''', "    ''')
        _v = ', '.join(['%u' for _ in range(VFFSL(SL,"seedCount",True))]) # u"${', '.join(['%u' for _ in xrange($seedCount)])}" on line 111, col 19
        if _v is not None: write(_filter(_v, rawExpr="${', '.join(['%u' for _ in xrange($seedCount)])}")) # from line 111, col 19.
        write('''\\n", ''')
        _v = ', '.join([''.join([str(VFFSL(SL,"generatorName",True)),'_seeds[',str(VFFSL(SL,"i",True)),']']) for i in range(VFFSL(SL,"seedCount",True))]) # u"${', '.join([c'${generatorName}_seeds[$i]' for i in xrange($seedCount)])}" on line 111, col 72
        if _v is not None: write(_filter(_v, rawExpr="${', '.join([c'${generatorName}_seeds[$i]' for i in xrange($seedCount)])}")) # from line 111, col 72.
        write(''');
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
        # POSIXGenerator.tmpl
        # 
        # Created by Graham Dennis on 2010-11-28.
        # 
        # Copyright (c) 2010-2012, Graham Dennis
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
        #  We only need to seed the system random number generator once, even if there are multiple Generator objects.
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

    _mainCheetahMethod_for_Generator= 'writeBody'

## END CLASS DEFINITION

if not hasattr(Generator, '_initCheetahAttributes'):
    templateAPIClass = getattr(Generator, '_CHEETAH_templateClass', Template)
    templateAPIClass._addCheetahPlumbingCodeToClass(Generator)


# CHEETAH was developed by Tavis Rudd and Mike Orr
# with code, advice and input from many other volunteers.
# For more information visit http://www.CheetahTemplate.org/

##################################################
## if run from command line:
if __name__ == '__main__':
    from Cheetah.TemplateCmdLineIface import CmdLineIface
    CmdLineIface(templateObj=Generator()).run()


