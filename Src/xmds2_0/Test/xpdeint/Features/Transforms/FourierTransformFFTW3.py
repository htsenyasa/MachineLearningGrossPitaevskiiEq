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
from xpdeint.Features.Transforms._FourierTransformFFTW3 import _FourierTransformFFTW3

##################################################
## MODULE CONSTANTS
VFFSL=valueFromFrameOrSearchList
VFSL=valueFromSearchList
VFN=valueForName
currentTime=time.time
__CHEETAH_version__ = '2.4.4'
__CHEETAH_versionTuple__ = (2, 4, 4, 'development', 0)
__CHEETAH_genTime__ = 1484975071.576725
__CHEETAH_genTimestamp__ = 'Sat Jan 21 16:04:31 2017'
__CHEETAH_src__ = '/home/mattias/xmds-2.2.3/admin/staging/xmds-2.2.3/xpdeint/Features/Transforms/FourierTransformFFTW3.tmpl'
__CHEETAH_srcLastModified__ = 'Fri Jul  5 16:29:35 2013'
__CHEETAH_docstring__ = 'Autogenerated by Cheetah: The Python-Powered Template Engine'

if __CHEETAH_versionTuple__ < RequiredCheetahVersionTuple:
    raise AssertionError(
      'This template was compiled with Cheetah version'
      ' %s. Templates compiled before version %s must be recompiled.'%(
         __CHEETAH_version__, RequiredCheetahVersion))

##################################################
## CLASSES

class FourierTransformFFTW3(_FourierTransformFFTW3):

    ##################################################
    ## CHEETAH GENERATED METHODS


    def __init__(self, *args, **KWs):

        super(FourierTransformFFTW3, self).__init__(*args, **KWs)
        if not self._CHEETAH__instanceInitialized:
            cheetahKWArgs = {}
            allowedKWs = 'searchList namespaces filter filtersLib errorCatcher'.split()
            for k,v in list(KWs.items()):
                if k in allowedKWs: cheetahKWArgs[k] = v
            self._initCheetahInstance(**cheetahKWArgs)
        

    def description(self, **KWS):



        ## Generated from @def description: FFTW3 at line 24, col 1.
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
        
        write('''FFTW3''')
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def includes(self, **KWS):



        ## CHEETAH: generated from @def includes at line 29, col 1.
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
        _v = super(FourierTransformFFTW3, self).includes()
        if _v is not None: write(_filter(_v))
        # 
        write('''#if (CFG_COMPILER == CFG_COMPILER_MSVC)
  #define FFTW_DLL
#endif

#include <fftw3.h>
#include <sys/stat.h>
#include <sys/types.h>

#define _xmds_malloc ''')
        _v = VFFSL(SL,"fftwPrefix",True) # u'${fftwPrefix}' on line 41, col 22
        if _v is not None: write(_filter(_v, rawExpr='${fftwPrefix}')) # from line 41, col 22.
        write('''_malloc
#define xmds_free ''')
        _v = VFFSL(SL,"fftwPrefix",True) # u'${fftwPrefix}' on line 42, col 19
        if _v is not None: write(_filter(_v, rawExpr='${fftwPrefix}')) # from line 42, col 19.
        write('''_free
''')
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def globals(self, **KWS):


        """
        Return the string defining the globals needed by FFTW3.
        """

        ## CHEETAH: generated from @def globals at line 45, col 1.
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
        write('''const real _inverse_sqrt_2pi = 1.0 / sqrt(2.0 * M_PI); 
string _fftwWisdomPath;
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def transformFunction(self, transformID, transformDict, function, **KWS):



        ## CHEETAH: generated from @def transformFunction(transformID, transformDict, function) at line 55, col 1.
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
        runtimePrefix, prefixLattice, postfixLattice, runtimePostfix = transformDict['transformSpecifier']
        write('''// _prefix_lattice should be ''')
        _v = VFFSL(SL,"prefixLattice",True) # u'${prefixLattice}' on line 58, col 30
        if _v is not None: write(_filter(_v, rawExpr='${prefixLattice}')) # from line 58, col 30.
        _v = ''.join([' * ' + runtimeLattice for runtimeLattice in runtimePrefix]) # u"${''.join([' * ' + runtimeLattice for runtimeLattice in runtimePrefix])}" on line 58, col 46
        if _v is not None: write(_filter(_v, rawExpr="${''.join([' * ' + runtimeLattice for runtimeLattice in runtimePrefix])}")) # from line 58, col 46.
        write('''
// _postfix_lattice should be ''')
        _v = VFFSL(SL,"postfixLattice",True) # u'${postfixLattice}' on line 59, col 31
        if _v is not None: write(_filter(_v, rawExpr='${postfixLattice}')) # from line 59, col 31.
        _v = ''.join([' * ' + runtimeLattice for runtimeLattice in runtimePostfix]) # u"${''.join([' * ' + runtimeLattice for runtimeLattice in runtimePostfix])}" on line 59, col 48
        if _v is not None: write(_filter(_v, rawExpr="${''.join([' * ' + runtimeLattice for runtimeLattice in runtimePostfix])}")) # from line 59, col 48.
        write('''
static ''')
        _v = VFFSL(SL,"fftwPrefix",True) # u'${fftwPrefix}' on line 60, col 8
        if _v is not None: write(_filter(_v, rawExpr='${fftwPrefix}')) # from line 60, col 8.
        write('''_plan _fftw_forward_plan = NULL;
static ''')
        _v = VFFSL(SL,"fftwPrefix",True) # u'${fftwPrefix}' on line 61, col 8
        if _v is not None: write(_filter(_v, rawExpr='${fftwPrefix}')) # from line 61, col 8.
        write('''_plan _fftw_backward_plan = NULL;

if (!_fftw_forward_plan) {
  _LOG(_SIMULATION_LOG_LEVEL, "Planning for ''')
        _v = VFFSL(SL,"function.description",True) # u'${function.description}' on line 64, col 45
        if _v is not None: write(_filter(_v, rawExpr='${function.description}')) # from line 64, col 45.
        write('''...");
  
''')
        transformPair = transformDict['transformPair']
        dimensionsBeingTransformed = len(transformPair[0])
        transformType = transformDict['transformType']
        write('''  ''')
        _v = VFFSL(SL,"fftwPrefix",True) # u'${fftwPrefix}' on line 69, col 3
        if _v is not None: write(_filter(_v, rawExpr='${fftwPrefix}')) # from line 69, col 3.
        write('''_iodim _transform_sizes[''')
        _v = VFFSL(SL,"dimensionsBeingTransformed",True) # u'${dimensionsBeingTransformed}' on line 69, col 40
        if _v is not None: write(_filter(_v, rawExpr='${dimensionsBeingTransformed}')) # from line 69, col 40.
        write('''], _loop_sizes[2];
''')
        if transformType == 'real': # generated from line 70, col 3
            write('''  ''')
            _v = VFFSL(SL,"fftwPrefix",True) # u'${fftwPrefix}' on line 71, col 3
            if _v is not None: write(_filter(_v, rawExpr='${fftwPrefix}')) # from line 71, col 3.
            write('''_r2r_kind _r2r_kinds[''')
            _v = VFFSL(SL,"dimensionsBeingTransformed",True) # u'${dimensionsBeingTransformed}' on line 71, col 37
            if _v is not None: write(_filter(_v, rawExpr='${dimensionsBeingTransformed}')) # from line 71, col 37.
            write('''];
''')
        write('''  ''')
        _v = VFFSL(SL,"fftwPrefix",True) # u'${fftwPrefix}' on line 73, col 3
        if _v is not None: write(_filter(_v, rawExpr='${fftwPrefix}')) # from line 73, col 3.
        write('''_iodim *_iodim_ptr = NULL;
  
  int _transform_sizes_index = 0, _loop_sizes_index = 0;
  
  if (_prefix_lattice > 1) {
    _iodim_ptr = &_loop_sizes[_loop_sizes_index++];
    _iodim_ptr->n = _prefix_lattice;
    _iodim_ptr->is = _iodim_ptr->os = _postfix_lattice * ''')
        _v = ' * '.join([dimRep.globalLattice for dimRep in transformPair[0]]) # u"${' * '.join([dimRep.globalLattice for dimRep in transformPair[0]])}" on line 80, col 58
        if _v is not None: write(_filter(_v, rawExpr="${' * '.join([dimRep.globalLattice for dimRep in transformPair[0]])}")) # from line 80, col 58.
        write(''';
  }
  if (_postfix_lattice > 1) {
    _iodim_ptr = &_loop_sizes[_loop_sizes_index++];
    _iodim_ptr->n = _postfix_lattice;
    _iodim_ptr->is = _iodim_ptr->os = 1;
  }
''')
        # 
        for dimID, dimRep in enumerate(transformPair[0]): # generated from line 88, col 3
            write('''  _iodim_ptr = &_transform_sizes[_transform_sizes_index++];
  _iodim_ptr->n = ''')
            _v = VFFSL(SL,"dimRep.globalLattice",True) # u'${dimRep.globalLattice}' on line 90, col 19
            if _v is not None: write(_filter(_v, rawExpr='${dimRep.globalLattice}')) # from line 90, col 19.
            write(''';
  _iodim_ptr->is = _iodim_ptr->os = _postfix_lattice''')
            _v = ''.join(''.join([' * ',str(VFFSL(SL,"dr.globalLattice",True))]) for dr in transformPair[0][dimID+1:]) # u"${''.join(c' * ${dr.globalLattice}' for dr in transformPair[0][dimID+1:])}" on line 91, col 53
            if _v is not None: write(_filter(_v, rawExpr="${''.join(c' * ${dr.globalLattice}' for dr in transformPair[0][dimID+1:])}")) # from line 91, col 53.
            write(''';
  
''')
        # 
        if transformType == 'complex': # generated from line 95, col 3
            guruPlanFunction = self.createGuruDFTPlanInDirection
            executeSuffix = 'dft'
            reinterpretType = VFFSL(SL,"fftwPrefix",True) + '_complex'
        else: # generated from line 99, col 3
            guruPlanFunction = self.createGuruR2RPlanInDirection
            executeSuffix = 'r2r'
            reinterpretType = 'real'
        # 
        dataOut = '_data_out' if transformDict.get('outOfPlace', False) else '_data_in'
        flags = ' | FFTW_DESTROY_INPUT' if transformDict.get('outOfPlace', False) else ''
        # 
        write('''  
  ''')
        _v = VFFSL(SL,"guruPlanFunction",False)(transformDict, 'forward', dataOut, flags) # u"${guruPlanFunction(transformDict, 'forward', dataOut, flags), autoIndent=True}" on line 109, col 3
        if _v is not None: write(_filter(_v, autoIndent=True, rawExpr="${guruPlanFunction(transformDict, 'forward', dataOut, flags), autoIndent=True}")) # from line 109, col 3.
        write('''  ''')
        _v = VFFSL(SL,"guruPlanFunction",False)(transformDict, 'backward', dataOut, flags) # u"${guruPlanFunction(transformDict, 'backward', dataOut, flags), autoIndent=True}" on line 110, col 3
        if _v is not None: write(_filter(_v, autoIndent=True, rawExpr="${guruPlanFunction(transformDict, 'backward', dataOut, flags), autoIndent=True}")) # from line 110, col 3.
        write('''  
  _LOG(_SIMULATION_LOG_LEVEL, " done.\\n");
}

if (_forward) {
  ''')
        _v = VFFSL(SL,"fftwPrefix",True) # u'${fftwPrefix}' on line 116, col 3
        if _v is not None: write(_filter(_v, rawExpr='${fftwPrefix}')) # from line 116, col 3.
        write('''_execute_''')
        _v = VFFSL(SL,"executeSuffix",True) # u'${executeSuffix}' on line 116, col 25
        if _v is not None: write(_filter(_v, rawExpr='${executeSuffix}')) # from line 116, col 25.
        write('''(
    _fftw_forward_plan,
    reinterpret_cast<''')
        _v = VFFSL(SL,"reinterpretType",True) # u'${reinterpretType}' on line 118, col 22
        if _v is not None: write(_filter(_v, rawExpr='${reinterpretType}')) # from line 118, col 22.
        write('''*>(_data_in),
    reinterpret_cast<''')
        _v = VFFSL(SL,"reinterpretType",True) # u'${reinterpretType}' on line 119, col 22
        if _v is not None: write(_filter(_v, rawExpr='${reinterpretType}')) # from line 119, col 22.
        write('''*>(''')
        _v = VFFSL(SL,"dataOut",True) # u'${dataOut}' on line 119, col 43
        if _v is not None: write(_filter(_v, rawExpr='${dataOut}')) # from line 119, col 43.
        write(''')
  );
} else {
  ''')
        _v = VFFSL(SL,"fftwPrefix",True) # u'${fftwPrefix}' on line 122, col 3
        if _v is not None: write(_filter(_v, rawExpr='${fftwPrefix}')) # from line 122, col 3.
        write('''_execute_''')
        _v = VFFSL(SL,"executeSuffix",True) # u'${executeSuffix}' on line 122, col 25
        if _v is not None: write(_filter(_v, rawExpr='${executeSuffix}')) # from line 122, col 25.
        write('''(
    _fftw_backward_plan,
    reinterpret_cast<''')
        _v = VFFSL(SL,"reinterpretType",True) # u'${reinterpretType}' on line 124, col 22
        if _v is not None: write(_filter(_v, rawExpr='${reinterpretType}')) # from line 124, col 22.
        write('''*>(_data_in),
    reinterpret_cast<''')
        _v = VFFSL(SL,"reinterpretType",True) # u'${reinterpretType}' on line 125, col 22
        if _v is not None: write(_filter(_v, rawExpr='${reinterpretType}')) # from line 125, col 22.
        write('''*>(''')
        _v = VFFSL(SL,"dataOut",True) # u'${dataOut}' on line 125, col 43
        if _v is not None: write(_filter(_v, rawExpr='${dataOut}')) # from line 125, col 43.
        write(''')
  );
}
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def createGuruDFTPlanInDirection(self, transformDict, direction, dataOut, flags, **KWS):



        ## CHEETAH: generated from @def createGuruDFTPlanInDirection($transformDict, $direction, $dataOut, $flags) at line 131, col 1.
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
        write('''_fftw_''')
        _v = VFFSL(SL,"direction",True) # u'${direction}' on line 133, col 7
        if _v is not None: write(_filter(_v, rawExpr='${direction}')) # from line 133, col 7.
        write('''_plan = ''')
        _v = VFFSL(SL,"fftwPrefix",True) # u'${fftwPrefix}' on line 133, col 27
        if _v is not None: write(_filter(_v, rawExpr='${fftwPrefix}')) # from line 133, col 27.
        write('''_plan_guru_dft(
  _transform_sizes_index, _transform_sizes,
  _loop_sizes_index, _loop_sizes,
  reinterpret_cast<''')
        _v = VFFSL(SL,"fftwPrefix",True) # u'${fftwPrefix}' on line 136, col 20
        if _v is not None: write(_filter(_v, rawExpr='${fftwPrefix}')) # from line 136, col 20.
        write('''_complex*>(_data_in), reinterpret_cast<''')
        _v = VFFSL(SL,"fftwPrefix",True) # u'${fftwPrefix}' on line 136, col 72
        if _v is not None: write(_filter(_v, rawExpr='${fftwPrefix}')) # from line 136, col 72.
        write('''_complex*>(''')
        _v = VFFSL(SL,"dataOut",True) # u'$dataOut' on line 136, col 96
        if _v is not None: write(_filter(_v, rawExpr='$dataOut')) # from line 136, col 96.
        write('''),
  FFTW_''')
        _v = VFN(VFFSL(SL,"direction",True),"upper",False)() # u'${direction.upper()}' on line 137, col 8
        if _v is not None: write(_filter(_v, rawExpr='${direction.upper()}')) # from line 137, col 8.
        write(''', ''')
        _v = VFFSL(SL,"planType",True) # u'${planType}' on line 137, col 30
        if _v is not None: write(_filter(_v, rawExpr='${planType}')) # from line 137, col 30.
        _v = VFFSL(SL,"flags",True) # u'${flags}' on line 137, col 41
        if _v is not None: write(_filter(_v, rawExpr='${flags}')) # from line 137, col 41.
        write('''
);
if (!_fftw_''')
        _v = VFFSL(SL,"direction",True) # u'${direction}' on line 139, col 12
        if _v is not None: write(_filter(_v, rawExpr='${direction}')) # from line 139, col 12.
        write('''_plan)
  _LOG(_ERROR_LOG_LEVEL, "(%s: %i) Unable to create ''')
        _v = VFFSL(SL,"direction",True) # u'${direction}' on line 140, col 53
        if _v is not None: write(_filter(_v, rawExpr='${direction}')) # from line 140, col 53.
        write(''' dft plan.\\n", __FILE__, __LINE__);

''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def createGuruR2RPlanInDirection(self, transformDict, direction, dataOut, flags, **KWS):



        ## CHEETAH: generated from @def createGuruR2RPlanInDirection($transformDict, $direction, $dataOut, $flags) at line 145, col 1.
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
        for idx, dimRep in enumerate(transformDict['transformPair'][0]): # generated from line 147, col 3
            write('''_r2r_kinds[''')
            _v = VFFSL(SL,"idx",True) # u'${idx}' on line 148, col 12
            if _v is not None: write(_filter(_v, rawExpr='${idx}')) # from line 148, col 12.
            write('''] = ''')
            _v = VFFSL(SL,"r2rKindForDimensionAndDirection",False)(dimRep.name, direction) # u'${r2rKindForDimensionAndDirection(dimRep.name, direction)}' on line 148, col 22
            if _v is not None: write(_filter(_v, rawExpr='${r2rKindForDimensionAndDirection(dimRep.name, direction)}')) # from line 148, col 22.
            write(''';
''')
        write('''
_fftw_''')
        _v = VFFSL(SL,"direction",True) # u'${direction}' on line 151, col 7
        if _v is not None: write(_filter(_v, rawExpr='${direction}')) # from line 151, col 7.
        write('''_plan = ''')
        _v = VFFSL(SL,"fftwPrefix",True) # u'${fftwPrefix}' on line 151, col 27
        if _v is not None: write(_filter(_v, rawExpr='${fftwPrefix}')) # from line 151, col 27.
        write('''_plan_guru_r2r(
  _transform_sizes_index, _transform_sizes,
  _loop_sizes_index, _loop_sizes,
  reinterpret_cast<real*>(_data_in), reinterpret_cast<real*>(''')
        _v = VFFSL(SL,"dataOut",True) # u'$dataOut' on line 154, col 62
        if _v is not None: write(_filter(_v, rawExpr='$dataOut')) # from line 154, col 62.
        write('''),
  _r2r_kinds, ''')
        _v = VFFSL(SL,"planType",True) # u'${planType}' on line 155, col 15
        if _v is not None: write(_filter(_v, rawExpr='${planType}')) # from line 155, col 15.
        _v = VFFSL(SL,"flags",True) # u'${flags}' on line 155, col 26
        if _v is not None: write(_filter(_v, rawExpr='${flags}')) # from line 155, col 26.
        write('''
);
if (!_fftw_''')
        _v = VFFSL(SL,"direction",True) # u'${direction}' on line 157, col 12
        if _v is not None: write(_filter(_v, rawExpr='${direction}')) # from line 157, col 12.
        write('''_plan)
  _LOG(_ERROR_LOG_LEVEL, "(%s: %i) Unable to create ''')
        _v = VFFSL(SL,"direction",True) # u'${direction}' on line 158, col 53
        if _v is not None: write(_filter(_v, rawExpr='${direction}')) # from line 158, col 53.
        write(''' r2r plan.\\n", __FILE__, __LINE__);

''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def mainBegin(self, dict, **KWS):



        ## CHEETAH: generated from @def mainBegin($dict) at line 163, col 1.
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
        
        write("""// load wisdom
#if CFG_OSAPI == CFG_OSAPI_POSIX // Don't load wisdom on windows
""")
        _v = VFFSL(SL,"loadWisdom",True) # u'${loadWisdom}' on line 166, col 1
        if _v is not None: write(_filter(_v, rawExpr='${loadWisdom}')) # from line 166, col 1.
        write('''#endif // POSIX
''')
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def loadWisdom(self, **KWS):



        ## CHEETAH: generated from @def loadWisdom at line 171, col 1.
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
        write('''{
  char _hostName[256];
  gethostname(_hostName, 256);
  _hostName[255] = \'\\0\'; // just in case
  
  string _pathToWisdom = getenv("HOME");
  _pathToWisdom += "/.xmds/wisdom/";
  
  string _wisdomFileName = _hostName;
  _wisdomFileName += ".wisdom";
  _wisdomFileName += "''')
        _v = VFFSL(SL,"wisdomExtension",True) # u'${wisdomExtension}' on line 183, col 23
        if _v is not None: write(_filter(_v, rawExpr='${wisdomExtension}')) # from line 183, col 23.
        write('''";
  
  FILE *_fp = NULL;
  
  _fp = fopen(_pathToWisdom.c_str(), "r");
  if (_fp) {
    fclose(_fp);
  } else {
    int _result = mkdir((string(getenv("HOME")) + "/.xmds").c_str(), S_IRWXU);
    if (mkdir(_pathToWisdom.c_str(), S_IRWXU)) {
      // We failed to create the ~/.xmds/wisdom directory
      _LOG(_WARNING_LOG_LEVEL, "Warning: Cannot find enlightenment, the path to wisdom ~/.xmds/wisdom doesn\'t seem to exist and we couldn\'t create it.\\n"
                               "         I\'ll use the current path instead.\\n");
      _pathToWisdom = ""; // present directory
    }
    
  }
  
  _fftwWisdomPath = _pathToWisdom + _wisdomFileName;
  
  FILE *_wisdomFile = NULL;
  if ( (_wisdomFile = fopen(_fftwWisdomPath.c_str(), "r")) != NULL) {
    _LOG(_SIMULATION_LOG_LEVEL, "Found enlightenment... (Importing wisdom)\\n");
    ''')
        _v = VFFSL(SL,"fftwPrefix",True) # u'${fftwPrefix}' on line 206, col 5
        if _v is not None: write(_filter(_v, rawExpr='${fftwPrefix}')) # from line 206, col 5.
        write('''_import_wisdom_from_file(_wisdomFile);
    fclose(_wisdomFile);
  }
}
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def saveWisdom(self, **KWS):



        ## CHEETAH: generated from @def saveWisdom at line 214, col 1.
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
        write('''{
  FILE *_wisdomFile = NULL;
  if ( (_wisdomFile = fopen(_fftwWisdomPath.c_str(), "w")) != NULL) {
    ''')
        _v = VFFSL(SL,"fftwPrefix",True) # u'${fftwPrefix}' on line 219, col 5
        if _v is not None: write(_filter(_v, rawExpr='${fftwPrefix}')) # from line 219, col 5.
        write('''_export_wisdom_to_file(_wisdomFile);
    fclose(_wisdomFile);
  }
}
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def mainEnd(self, dict, **KWS):



        ## CHEETAH: generated from @def mainEnd($dict) at line 227, col 1.
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
        write('''
// Save wisdom
#if CFG_OSAPI == CFG_OSAPI_POSIX
''')
        _v = VFFSL(SL,"saveWisdom",True) # u'${saveWisdom, autoIndent=True}' on line 232, col 1
        if _v is not None: write(_filter(_v, autoIndent=True, rawExpr='${saveWisdom, autoIndent=True}')) # from line 232, col 1.
        write('''#endif // POSIX

''')
        _v = VFFSL(SL,"fftwPrefix",True) # u'${fftwPrefix}' on line 235, col 1
        if _v is not None: write(_filter(_v, rawExpr='${fftwPrefix}')) # from line 235, col 1.
        write('''_cleanup();
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
        # FourierTransformFFTW3.tmpl
        # 
        # Created by Graham Dennis on 2007-08-23.
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

    planType = "FFTW_MEASURE"

    supportsInPlaceOperation = True

    _mainCheetahMethod_for_FourierTransformFFTW3= 'writeBody'

## END CLASS DEFINITION

if not hasattr(FourierTransformFFTW3, '_initCheetahAttributes'):
    templateAPIClass = getattr(FourierTransformFFTW3, '_CHEETAH_templateClass', Template)
    templateAPIClass._addCheetahPlumbingCodeToClass(FourierTransformFFTW3)


# CHEETAH was developed by Tavis Rudd and Mike Orr
# with code, advice and input from many other volunteers.
# For more information visit http://www.CheetahTemplate.org/

##################################################
## if run from command line:
if __name__ == '__main__':
    from Cheetah.TemplateCmdLineIface import CmdLineIface
    CmdLineIface(templateObj=FourierTransformFFTW3()).run()


