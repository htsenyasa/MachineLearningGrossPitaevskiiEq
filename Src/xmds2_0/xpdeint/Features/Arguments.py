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
from xpdeint.Features._Feature import _Feature

##################################################
## MODULE CONSTANTS
VFFSL=valueFromFrameOrSearchList
VFSL=valueFromSearchList
VFN=valueForName
currentTime=time.time
__CHEETAH_version__ = '2.4.4'
__CHEETAH_versionTuple__ = (2, 4, 4, 'development', 0)
__CHEETAH_genTime__ = 1484975071.209611
__CHEETAH_genTimestamp__ = 'Sat Jan 21 16:04:31 2017'
__CHEETAH_src__ = '/home/mattias/xmds-2.2.3/admin/staging/xmds-2.2.3/xpdeint/Features/Arguments.tmpl'
__CHEETAH_srcLastModified__ = 'Tue May 22 16:27:12 2012'
__CHEETAH_docstring__ = 'Autogenerated by Cheetah: The Python-Powered Template Engine'

if __CHEETAH_versionTuple__ < RequiredCheetahVersionTuple:
    raise AssertionError(
      'This template was compiled with Cheetah version'
      ' %s. Templates compiled before version %s must be recompiled.'%(
         __CHEETAH_version__, RequiredCheetahVersion))

##################################################
## CLASSES

class Arguments(_Feature):

    ##################################################
    ## CHEETAH GENERATED METHODS


    def __init__(self, *args, **KWs):

        super(Arguments, self).__init__(*args, **KWs)
        if not self._CHEETAH__instanceInitialized:
            cheetahKWArgs = {}
            allowedKWs = 'searchList namespaces filter filtersLib errorCatcher'.split()
            for k,v in list(KWs.items()):
                if k in allowedKWs: cheetahKWArgs[k] = v
            self._initCheetahInstance(**cheetahKWArgs)
        

    def description(self, **KWS):



        ## Generated from @def description: Command line argument processing at line 24, col 1.
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
        
        write('''Command line argument processing''')
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def includes(self, **KWS):



        ## CHEETAH: generated from @def includes at line 27, col 1.
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
        _v = super(Arguments, self).includes()
        if _v is not None: write(_filter(_v))
        # 
        write('''#include <getopt.h>
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def globals(self, **KWS):



        ## CHEETAH: generated from @def globals at line 35, col 1.
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
        _v = super(Arguments, self).globals()
        if _v is not None: write(_filter(_v))
        # 
        for argument in VFFSL(SL,"argumentList",True): # generated from line 39, col 3
            if VFFSL(SL,"argument.type",True) == "string": # generated from line 40, col 5
                _v = VFFSL(SL,"argument.type",True) # u'$argument.type' on line 41, col 1
                if _v is not None: write(_filter(_v, rawExpr='$argument.type')) # from line 41, col 1.
                write(''' ''')
                _v = VFFSL(SL,"argument.name",True) # u'$argument.name' on line 41, col 16
                if _v is not None: write(_filter(_v, rawExpr='$argument.name')) # from line 41, col 16.
                write(''' = "''')
                _v = VFFSL(SL,"argument.defaultValue",True) # u'$argument.defaultValue' on line 41, col 34
                if _v is not None: write(_filter(_v, rawExpr='$argument.defaultValue')) # from line 41, col 34.
                write('''";
''')
            else: # generated from line 42, col 5
                _v = VFFSL(SL,"argument.type",True) # u'$argument.type' on line 43, col 1
                if _v is not None: write(_filter(_v, rawExpr='$argument.type')) # from line 43, col 1.
                write(''' ''')
                _v = VFFSL(SL,"argument.name",True) # u'$argument.name' on line 43, col 16
                if _v is not None: write(_filter(_v, rawExpr='$argument.name')) # from line 43, col 16.
                write(''' = ''')
                _v = VFFSL(SL,"argument.defaultValue",True) # u'$argument.defaultValue' on line 43, col 33
                if _v is not None: write(_filter(_v, rawExpr='$argument.defaultValue')) # from line 43, col 33.
                write('''; 
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def functionPrototypes(self, **KWS):



        ## CHEETAH: generated from @def functionPrototypes at line 49, col 1.
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
        _v = super(Arguments, self).functionPrototypes()
        if _v is not None: write(_filter(_v))
        # 
        write('''void _print_usage();
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def functionImplementations(self, **KWS):



        ## CHEETAH: generated from @def functionImplementations at line 57, col 1.
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
        _v = super(Arguments, self).functionImplementations()
        if _v is not None: write(_filter(_v))
        # 
        write('''void _print_usage()
{
  // This function does not return.
  _LOG(_NO_ERROR_TERMINATE_LOG_LEVEL, "\\n\\nUsage: ''')
        _v = VFFSL(SL,"simulationName",True) # u'$simulationName' on line 64, col 51
        if _v is not None: write(_filter(_v, rawExpr='$simulationName')) # from line 64, col 51.
        for argument in VFFSL(SL,"argumentList",True): # generated from line 65, col 1
            write(''' --''')
            _v = VFFSL(SL,"argument.name",True) # u'$argument.name' on line 66, col 4
            if _v is not None: write(_filter(_v, rawExpr='$argument.name')) # from line 66, col 4.
            write(''' <''')
            _v = VFFSL(SL,"argument.type",True) # u'$argument.type' on line 66, col 20
            if _v is not None: write(_filter(_v, rawExpr='$argument.type')) # from line 66, col 20.
            write('''>''')
        write('''\\n\\n"
                         "Details:\\n"
                         "Option\\t\\tType\\t\\tDefault value\\n"
''')
        for argument in VFFSL(SL,"argumentList",True): # generated from line 71, col 1
            write('''                         "-''')
            _v = VFFSL(SL,"argument.shortName",True) # u'$argument.shortName' on line 72, col 28
            if _v is not None: write(_filter(_v, rawExpr='$argument.shortName')) # from line 72, col 28.
            write(''',  --''')
            _v = VFFSL(SL,"argument.name",True) # u'$argument.name' on line 72, col 52
            if _v is not None: write(_filter(_v, rawExpr='$argument.name')) # from line 72, col 52.
            write('''\\t''')
            _v = VFFSL(SL,"argument.type",True) # u'$argument.type' on line 72, col 68
            if _v is not None: write(_filter(_v, rawExpr='$argument.type')) # from line 72, col 68.
            write(''' \\t\\t''')
            _v = VFFSL(SL,"argument.defaultValue",True) # u'$argument.defaultValue' on line 72, col 87
            if _v is not None: write(_filter(_v, rawExpr='$argument.defaultValue')) # from line 72, col 87.
            write('''\\n"
''')
        write('''                         );
  // _LOG terminates the simulation.
}
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def preAllocation(self, dict, **KWS):



        ## CHEETAH: generated from @def preAllocation($dict) at line 86, col 1.
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
        
        write('''// *********** Parse the command line for arguments, and set  *********
// *********** the appropriate global variables               *********

int resp;
std::map<string, string> mInputArgsAndValues;

while (1) {
  static struct option long_options[] = 
    {
      {"help", no_argument, 0, \'h\'},
''')
        for argument in VFFSL(SL,"argumentList",True): # generated from line 97, col 3
            write('''      {"''')
            _v = VFFSL(SL,"argument.name",True) # u'$argument.name' on line 98, col 9
            if _v is not None: write(_filter(_v, rawExpr='$argument.name')) # from line 98, col 9.
            write('''", required_argument, 0, \'''')
            _v = VFFSL(SL,"argument.shortName",True) # u'$argument.shortName' on line 98, col 49
            if _v is not None: write(_filter(_v, rawExpr='$argument.shortName')) # from line 98, col 49.
            write("""'},
""")
        write('''      {NULL, 0, 0, 0}
    };
  
  int option_index = 0;

  resp = getopt_long(argc, argv, "h''')
        for argument in VFFSL(SL,"argumentList",True): # generated from line 106, col 3
            _v = VFFSL(SL,"argument.shortName",True) # u'$argument.shortName' on line 107, col 1
            if _v is not None: write(_filter(_v, rawExpr='$argument.shortName')) # from line 107, col 1.
            write(''':''')
        write('''", long_options, &option_index);
  
  if (resp == -1)
    break;

  switch (resp) {
    case \'?\':
      // An unknown option was passed. Show allowed options and exit. 
      _print_usage(); // This causes the simulation to exit

    case \'h\':
      _print_usage(); // This causes the simulation to exit
''')
        for argument in VFFSL(SL,"argumentList",True): # generated from line 121, col 3
            write("""    
    case '""")
            _v = VFFSL(SL,"argument.shortName",True) # u'$argument.shortName' on line 123, col 11
            if _v is not None: write(_filter(_v, rawExpr='$argument.shortName')) # from line 123, col 11.
            write("""':
""")
            if VFFSL(SL,"appendArgsToOutputFilename",True): # generated from line 124, col 5
                write('''      mInputArgsAndValues.insert ( pair<string, string> (string("''')
                _v = VFFSL(SL,"argument.name",True) # u'$argument.name' on line 125, col 66
                if _v is not None: write(_filter(_v, rawExpr='$argument.name')) # from line 125, col 66.
                write('''"), string(optarg)));
''')
            if VFFSL(SL,"argument.type",True) == 'string': # generated from line 127, col 5
                write('''      ''')
                _v = VFFSL(SL,"argument.name",True) # u'$argument.name' on line 128, col 7
                if _v is not None: write(_filter(_v, rawExpr='$argument.name')) # from line 128, col 7.
                write(''' = string(optarg);
''')
            elif VFFSL(SL,"argument.type",True) in ('integer', 'int', 'long'): # generated from line 129, col 5
                write('''      ''')
                _v = VFFSL(SL,"argument.name",True) # u'$argument.name' on line 130, col 7
                if _v is not None: write(_filter(_v, rawExpr='$argument.name')) # from line 130, col 7.
                write(''' = strtol(optarg, NULL, 10);
''')
            elif VFFSL(SL,"argument.type",True) in 'real': # generated from line 131, col 5
                write('''      ''')
                _v = VFFSL(SL,"argument.name",True) # u'$argument.name' on line 132, col 7
                if _v is not None: write(_filter(_v, rawExpr='$argument.name')) # from line 132, col 7.
                write(''' = strtod(optarg, NULL);
''')
            write('''      break;
''')
        write('''      
    default:
      _LOG(_ERROR_LOG_LEVEL, "Internal error in processing arguments.\\n");
  }
}

''')
        if VFFSL(SL,"appendArgsToOutputFilename",True): # generated from line 142, col 3
            write("""// Try and insert all the default arguments; the insert will fail if that
// argument is already in the map. This way we make sure that all the 
// possible command line arguments are in the map, even if they weren't passed. 

""")
            for argument in VFFSL(SL,"argumentList",True): # generated from line 147, col 5
                write('''mInputArgsAndValues.insert ( pair<string, string> (string("''')
                _v = VFFSL(SL,"argument.name",True) # u'$argument.name' on line 148, col 60
                if _v is not None: write(_filter(_v, rawExpr='$argument.name')) # from line 148, col 60.
                write('''"), string("''')
                _v = VFFSL(SL,"argument.defaultValue",True) # u'$argument.defaultValue' on line 148, col 86
                if _v is not None: write(_filter(_v, rawExpr='$argument.defaultValue')) # from line 148, col 86.
                write('''")));
''')
            write('''
// Since the command line arguments and their values are to be appended to the
// output filenames, construct the append string here
for (map<string, string>::iterator iter=mInputArgsAndValues.begin() ; iter != mInputArgsAndValues.end(); iter++ ) {
  gsArgsAndValues += string(".") + (*iter).first + string("_") + (*iter).second;
}
''')
        write('''
if (optind < argc)
  _print_usage(); // This causes the simulation to exit.

''')
        if 'postArgumentProcessing' in self.codeBlocks: # generated from line 161, col 3
            write('''// ******** Argument post-processing code *******
''')
            _v = VFN(VFFSL(SL,"codeBlocks",True)['postArgumentProcessing'],"codeString",True) # u"${codeBlocks['postArgumentProcessing'].codeString}" on line 163, col 1
            if _v is not None: write(_filter(_v, rawExpr="${codeBlocks['postArgumentProcessing'].codeString}")) # from line 163, col 1.
            write('''// **********************************************

''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def xsilOutputInfo(self, dict, **KWS):



        ## CHEETAH: generated from @def xsilOutputInfo($dict) at line 170, col 1.
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
        formatSpecifierMap = {'string': ('s', '.c_str()'),                              'int':    ('i', ''),                              'long':   ('li', ''),                              'integer': ('li', ''),                              'real': ('e', '')}
        # 
        write('''fprintf(''')
        _v = VFFSL(SL,"fp",True) # u'$fp' on line 179, col 9
        if _v is not None: write(_filter(_v, rawExpr='$fp')) # from line 179, col 9.
        write(''', "\\nVariables that can be specified on the command line:\\n");
''')
        for argument in VFFSL(SL,"argumentList",True): # generated from line 180, col 3
            write('''
''')
            formatSpecifier, argumentSuffix = formatSpecifierMap[VFFSL(SL,"argument.type",True)]
            write('''fprintf(''')
            _v = VFFSL(SL,"fp",True) # u'$fp' on line 183, col 9
            if _v is not None: write(_filter(_v, rawExpr='$fp')) # from line 183, col 9.
            write(''', "  Command line argument ''')
            _v = VFFSL(SL,"argument.name",True) # u'${argument.name}' on line 183, col 39
            if _v is not None: write(_filter(_v, rawExpr='${argument.name}')) # from line 183, col 39.
            write(''' = %''')
            _v = VFFSL(SL,"formatSpecifier",True) # u'${formatSpecifier}' on line 183, col 59
            if _v is not None: write(_filter(_v, rawExpr='${formatSpecifier}')) # from line 183, col 59.
            write('''\\n", ''')
            _v = VFFSL(SL,"argument.name",True) # u'${argument.name}' on line 183, col 82
            if _v is not None: write(_filter(_v, rawExpr='${argument.name}')) # from line 183, col 82.
            _v = VFFSL(SL,"argumentSuffix",True) # u'${argumentSuffix}' on line 183, col 98
            if _v is not None: write(_filter(_v, rawExpr='${argumentSuffix}')) # from line 183, col 98.
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
        # Arguments.tmpl
        # 
        # Created by Mattias Johnsson on 2008-02-21.
        # 
        # Copyright (c) 2008-2012 Mattias Johnsson
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
        #  This code needs to go at the very beginning of the main function
        #  so we use the preAllocation code insertion point instead of mainBegin.
        #  See SimulationDriver.tmpl for ordering of insertion points.
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

    featureName = 'Arguments'

    _mainCheetahMethod_for_Arguments= 'writeBody'

## END CLASS DEFINITION

if not hasattr(Arguments, '_initCheetahAttributes'):
    templateAPIClass = getattr(Arguments, '_CHEETAH_templateClass', Template)
    templateAPIClass._addCheetahPlumbingCodeToClass(Arguments)


# CHEETAH was developed by Tavis Rudd and Mike Orr
# with code, advice and input from many other volunteers.
# For more information visit http://www.CheetahTemplate.org/

##################################################
## if run from command line:
if __name__ == '__main__':
    from Cheetah.TemplateCmdLineIface import CmdLineIface
    CmdLineIface(templateObj=Arguments()).run()


