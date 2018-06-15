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
from xpdeint.Geometry.SplitUniformDimensionRepresentation import SplitUniformDimensionRepresentation
from xpdeint.CallOnceGuards import callOnceGuard
from itertools import chain

##################################################
## MODULE CONSTANTS
VFFSL=valueFromFrameOrSearchList
VFSL=valueFromSearchList
VFN=valueForName
currentTime=time.time
__CHEETAH_version__ = '2.4.4'
__CHEETAH_versionTuple__ = (2, 4, 4, 'development', 0)
__CHEETAH_genTime__ = 1484975071.895736
__CHEETAH_genTimestamp__ = 'Sat Jan 21 16:04:31 2017'
__CHEETAH_src__ = '/home/mattias/xmds-2.2.3/admin/staging/xmds-2.2.3/xpdeint/HDF5.tmpl'
__CHEETAH_srcLastModified__ = 'Mon Apr 23 13:26:13 2012'
__CHEETAH_docstring__ = 'Autogenerated by Cheetah: The Python-Powered Template Engine'

if __CHEETAH_versionTuple__ < RequiredCheetahVersionTuple:
    raise AssertionError(
      'This template was compiled with Cheetah version'
      ' %s. Templates compiled before version %s must be recompiled.'%(
         __CHEETAH_version__, RequiredCheetahVersion))

##################################################
## CLASSES

class HDF5(ScriptElement):

    ##################################################
    ## CHEETAH GENERATED METHODS


    def __init__(self, *args, **KWs):

        super(HDF5, self).__init__(*args, **KWs)
        if not self._CHEETAH__instanceInitialized:
            cheetahKWArgs = {}
            allowedKWs = 'searchList namespaces filter filtersLib errorCatcher'.split()
            for k,v in list(KWs.items()):
                if k in allowedKWs: cheetahKWArgs[k] = v
            self._initCheetahInstance(**cheetahKWArgs)
        

    @callOnceGuard
    def includes(self, **KWS):



        ## CHEETAH: generated from @def includes at line 31, col 1.
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
        _v = super(HDF5, self).includes()
        if _v is not None: write(_filter(_v))
        # 
        write("""#define H5_USE_16_API
#include <hdf5.h>

#if !defined(HAVE_H5LEXISTS)
htri_t H5Lexists(hid_t loc_id, const char *name, hid_t lapl_id)
{
  H5E_auto_t error_func;
  void* error_client_data;
  // Squelch errors generated by H5Gget_objinfo. It will report errors when it can't find an object
  // but that's the purpose of calling it.
  H5Eget_auto(&error_func, &error_client_data);
  H5Eset_auto(NULL, NULL);
  herr_t err = H5Gget_objinfo(loc_id, name, false, NULL);
  H5Eset_auto(error_func, error_client_data);
  if (err >= 0)
    return true;
  else
    return false;
}
#endif

#define H5T_NATIVE_REAL H5T_NATIVE_""")
        _v = {'double': 'DOUBLE', 'single': 'FLOAT'}[VFFSL(SL,"precision",True)] # u"${{'double': 'DOUBLE', 'single': 'FLOAT'}[$precision]}" on line 56, col 36
        if _v is not None: write(_filter(_v, rawExpr="${{'double': 'DOUBLE', 'single': 'FLOAT'}[$precision]}")) # from line 56, col 36.
        write('''
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def processData(self, dict, **KWS):



        ## CHEETAH: generated from @def processData(dict) at line 60, col 1.
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
        
        field = dict['field']
        basis = dict['basis']
        operation = dict['operation']
        assert operation in ['read', 'write']
        variables = dict['variables']
        dimensionOffsets = dict.get('dimensionOffsets', {})
        componentCount = 0
        for variable in VFFSL(SL,"variables",True): # generated from line 68, col 3
            componentCount += len(VFFSL(SL,"variable.vector.components",True))
            if VFFSL(SL,"variable.vector.type",True) == 'complex': # generated from line 70, col 5
                componentCount += len(VFFSL(SL,"variable.vector.components",True))
        dict['componentCount'] = componentCount
        write('''/* Create the data space */
''')
        dimensionCount = len(field.dimensions)
        #  File dim reps must be in dimension order as that is the order we desire for write-out
        fileDimReps = [dim.inBasis(basis) for dim in field.dimensions]
        memDimReps = field.inBasis(basis)
        #  Construct a list of (fileDimNum, memDimNum, dimRep) tuples. This is necessary
        #  for the case where we are using a distributed MPI driver with FFT's
        #  and the first two dimensions are transformed. In this situation, the
        #  first and second dimensions are transposed.
        # 
        dimRepOrdering = [(fileDimNum, memDimReps.index(dimRep), dimRep)                          for fileDimNum, dimRep in enumerate(fileDimReps)]
        # 
        write('''hsize_t file_start[''')
        _v = VFFSL(SL,"dimensionCount",True) # u'${dimensionCount}' on line 88, col 20
        if _v is not None: write(_filter(_v, rawExpr='${dimensionCount}')) # from line 88, col 20.
        write('''] = {''')
        _v = ', '.join(dimRep.localOffset for dimRep in fileDimReps) # u"${', '.join(dimRep.localOffset for dimRep in fileDimReps)}" on line 88, col 42
        if _v is not None: write(_filter(_v, rawExpr="${', '.join(dimRep.localOffset for dimRep in fileDimReps)}")) # from line 88, col 42.
        write('''};
hsize_t mem_dims[''')
        _v = VFFSL(SL,"dimensionCount",True)+1 # u'${dimensionCount+1}' on line 89, col 18
        if _v is not None: write(_filter(_v, rawExpr='${dimensionCount+1}')) # from line 89, col 18.
        write('''] = {''')
        _v = ', '.join(chain((dimRep.localLattice for dimRep in memDimReps), ['1'])) # u"${', '.join(chain((dimRep.localLattice for dimRep in memDimReps), ['1']))}" on line 89, col 42
        if _v is not None: write(_filter(_v, rawExpr="${', '.join(chain((dimRep.localLattice for dimRep in memDimReps), ['1']))}")) # from line 89, col 42.
        write('''};
hsize_t mem_start[''')
        _v = VFFSL(SL,"dimensionCount",True)+1 # u'${dimensionCount+1}' on line 90, col 19
        if _v is not None: write(_filter(_v, rawExpr='${dimensionCount+1}')) # from line 90, col 19.
        write('''] = {''')
        _v = ', '.join(['0']*(dimensionCount+1)) # u"${', '.join(['0']*(dimensionCount+1))}" on line 90, col 43
        if _v is not None: write(_filter(_v, rawExpr="${', '.join(['0']*(dimensionCount+1))}")) # from line 90, col 43.
        write('''};
hsize_t mem_stride[''')
        _v = VFFSL(SL,"dimensionCount",True)+1 # u'${dimensionCount+1}' on line 91, col 20
        if _v is not None: write(_filter(_v, rawExpr='${dimensionCount+1}')) # from line 91, col 20.
        write('''] = {''')
        _v = ', '.join(['1']*(dimensionCount+1)) # u"${', '.join(['1']*(dimensionCount+1))}" on line 91, col 44
        if _v is not None: write(_filter(_v, rawExpr="${', '.join(['1']*(dimensionCount+1))}")) # from line 91, col 44.
        write('''};
hsize_t mem_count[''')
        _v = VFFSL(SL,"dimensionCount",True)+1 # u'${dimensionCount+1}' on line 92, col 19
        if _v is not None: write(_filter(_v, rawExpr='${dimensionCount+1}')) # from line 92, col 19.
        write('''] = {''')
        _v = ', '.join(chain((dimRep.localLattice for dimRep in memDimReps), ['1'])) # u"${', '.join(chain((dimRep.localLattice for dimRep in memDimReps), ['1']))}" on line 92, col 43
        if _v is not None: write(_filter(_v, rawExpr="${', '.join(chain((dimRep.localLattice for dimRep in memDimReps), ['1']))}")) # from line 92, col 43.
        write('''};

''')
        for fileDimNum, memDimNum, dimRep in dimRepOrdering: # generated from line 94, col 3
            if dimRep.name in dimensionOffsets and not isinstance(dimRep, SplitUniformDimensionRepresentation): # generated from line 95, col 5
                offset = dimensionOffsets[dimRep.name]
                write('''if (file_start[''')
                _v = VFFSL(SL,"fileDimNum",True) # u'${fileDimNum}' on line 97, col 16
                if _v is not None: write(_filter(_v, rawExpr='${fileDimNum}')) # from line 97, col 16.
                write('''] < ''')
                _v = VFFSL(SL,"offset",True) # u'${offset}' on line 97, col 33
                if _v is not None: write(_filter(_v, rawExpr='${offset}')) # from line 97, col 33.
                write(''') {
  if (mem_count[''')
                _v = VFFSL(SL,"memDimNum",True) # u'${memDimNum}' on line 98, col 17
                if _v is not None: write(_filter(_v, rawExpr='${memDimNum}')) # from line 98, col 17.
                write('''] < ''')
                _v = VFFSL(SL,"offset",True) # u'${offset}' on line 98, col 33
                if _v is not None: write(_filter(_v, rawExpr='${offset}')) # from line 98, col 33.
                write(''' - file_start[''')
                _v = VFFSL(SL,"fileDimNum",True) # u'${fileDimNum}' on line 98, col 56
                if _v is not None: write(_filter(_v, rawExpr='${fileDimNum}')) # from line 98, col 56.
                write('''])
    mem_count[''')
                _v = VFFSL(SL,"memDimNum",True) # u'${memDimNum}' on line 99, col 15
                if _v is not None: write(_filter(_v, rawExpr='${memDimNum}')) # from line 99, col 15.
                write('''] = 0;
  else {
    mem_count[''')
                _v = VFFSL(SL,"memDimNum",True) # u'${memDimNum}' on line 101, col 15
                if _v is not None: write(_filter(_v, rawExpr='${memDimNum}')) # from line 101, col 15.
                write('''] -= ''')
                _v = VFFSL(SL,"offset",True) # u'${offset}' on line 101, col 32
                if _v is not None: write(_filter(_v, rawExpr='${offset}')) # from line 101, col 32.
                write(''' - file_start[''')
                _v = VFFSL(SL,"fileDimNum",True) # u'${fileDimNum}' on line 101, col 55
                if _v is not None: write(_filter(_v, rawExpr='${fileDimNum}')) # from line 101, col 55.
                write('''];
    mem_start[''')
                _v = VFFSL(SL,"memDimNum",True) # u'${memDimNum}' on line 102, col 15
                if _v is not None: write(_filter(_v, rawExpr='${memDimNum}')) # from line 102, col 15.
                write('''] += ''')
                _v = VFFSL(SL,"offset",True) # u'${offset}' on line 102, col 32
                if _v is not None: write(_filter(_v, rawExpr='${offset}')) # from line 102, col 32.
                write(''' - file_start[''')
                _v = VFFSL(SL,"fileDimNum",True) # u'${fileDimNum}' on line 102, col 55
                if _v is not None: write(_filter(_v, rawExpr='${fileDimNum}')) # from line 102, col 55.
                write('''];
  }
  file_start[''')
                _v = VFFSL(SL,"fileDimNum",True) # u'${fileDimNum}' on line 104, col 14
                if _v is not None: write(_filter(_v, rawExpr='${fileDimNum}')) # from line 104, col 14.
                write('''] = 0;
} else {
  file_start[''')
                _v = VFFSL(SL,"fileDimNum",True) # u'${fileDimNum}' on line 106, col 14
                if _v is not None: write(_filter(_v, rawExpr='${fileDimNum}')) # from line 106, col 14.
                write('''] -= ''')
                _v = VFFSL(SL,"offset",True) # u'${offset}' on line 106, col 32
                if _v is not None: write(_filter(_v, rawExpr='${offset}')) # from line 106, col 32.
                write(''';
}

if (mem_count[''')
                _v = VFFSL(SL,"memDimNum",True) # u'$memDimNum' on line 109, col 15
                if _v is not None: write(_filter(_v, rawExpr='$memDimNum')) # from line 109, col 15.
                write('''] > file_dims[''')
                _v = VFFSL(SL,"fileDimNum",True) # u'$fileDimNum' on line 109, col 39
                if _v is not None: write(_filter(_v, rawExpr='$fileDimNum')) # from line 109, col 39.
                write(''']) {
  mem_count[''')
                _v = VFFSL(SL,"memDimNum",True) # u'$memDimNum' on line 110, col 13
                if _v is not None: write(_filter(_v, rawExpr='$memDimNum')) # from line 110, col 13.
                write('''] = file_dims[''')
                _v = VFFSL(SL,"fileDimNum",True) # u'$fileDimNum' on line 110, col 37
                if _v is not None: write(_filter(_v, rawExpr='$fileDimNum')) # from line 110, col 37.
                write('''];
}
''')
        write('''
hid_t mem_dataspace;
''')
        for variable in variables: # generated from line 116, col 3
            components = VFFSL(SL,"variable.separatedComponents",True)
            write('''mem_dims[''')
            _v = VFFSL(SL,"dimensionCount",True) # u'${dimensionCount}' on line 118, col 10
            if _v is not None: write(_filter(_v, rawExpr='${dimensionCount}')) # from line 118, col 10.
            write('''] = ''')
            _v = VFFSL(SL,"len",False)(components) # u'${len(components)}' on line 118, col 31
            if _v is not None: write(_filter(_v, rawExpr='${len(components)}')) # from line 118, col 31.
            write(''';
mem_dataspace = H5Screate_simple(''')
            _v = VFFSL(SL,"dimensionCount",True)+1 # u'${dimensionCount+1}' on line 119, col 34
            if _v is not None: write(_filter(_v, rawExpr='${dimensionCount+1}')) # from line 119, col 34.
            write(''', mem_dims, NULL);
mem_stride[''')
            _v = VFFSL(SL,"dimensionCount",True) # u'${dimensionCount}' on line 120, col 12
            if _v is not None: write(_filter(_v, rawExpr='${dimensionCount}')) # from line 120, col 12.
            write('''] = ''')
            _v = VFFSL(SL,"len",False)(components) # u'${len(components)}' on line 120, col 33
            if _v is not None: write(_filter(_v, rawExpr='${len(components)}')) # from line 120, col 33.
            write(''';

''')
            ## START CAPTURE REGION: _69257830 writeLoopContents at line 122, col 5 in the source.
            _orig_trans_69257830 = trans
            _wasBuffering_69257830 = self._CHEETAH__isBuffering
            self._CHEETAH__isBuffering = True
            trans = _captureCollector_69257830 = DummyTransaction()
            write = _captureCollector_69257830.response().write
            for offset, componentName in components: # generated from line 123, col 7
                write('''mem_start[''')
                _v = VFFSL(SL,"dimensionCount",True) # u'${dimensionCount}' on line 124, col 11
                if _v is not None: write(_filter(_v, rawExpr='${dimensionCount}')) # from line 124, col 11.
                write('''] = ''')
                _v = VFFSL(SL,"offset",True) # u'$offset' on line 124, col 32
                if _v is not None: write(_filter(_v, rawExpr='$offset')) # from line 124, col 32.
                write(''';
H5Sselect_hyperslab(mem_dataspace, H5S_SELECT_SET, mem_start, mem_stride, mem_count, NULL);
''')
                #  
                #   This looks like a typo because 'mem_stride' and 'mem_count' are used with 'file_start' are used here.
                #   But it isn't a typo. The idea here is that the selection we want to make in the file has the same
                #   number of elements in each dimension and the same stride as in memory (but ignoring the last dimension).
                #   The only difference is the starting position for the selection.
                # 
                if fileDimReps: # generated from line 132, col 9
                    #  We can only do a selection in the file if the output data is more than zero-dimensional
                    write('''H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, file_start, mem_stride, mem_count, NULL);
''')
                write('''
if (dataset_''')
                _v = VFFSL(SL,"componentName",True) # u'${componentName}' on line 137, col 13
                if _v is not None: write(_filter(_v, rawExpr='${componentName}')) # from line 137, col 13.
                write(''')
  H5D''')
                _v = VFFSL(SL,"operation",True) # u'${operation}' on line 138, col 6
                if _v is not None: write(_filter(_v, rawExpr='${operation}')) # from line 138, col 6.
                write('''(dataset_''')
                _v = VFFSL(SL,"componentName",True) # u'${componentName}' on line 138, col 27
                if _v is not None: write(_filter(_v, rawExpr='${componentName}')) # from line 138, col 27.
                write(''', H5T_NATIVE_REAL, mem_dataspace, file_dataspace, H5P_DEFAULT, ''')
                _v = VFFSL(SL,"variable.arrayName",True) # u'${variable.arrayName}' on line 138, col 106
                if _v is not None: write(_filter(_v, rawExpr='${variable.arrayName}')) # from line 138, col 106.
                write(''');
''')
            trans = _orig_trans_69257830
            write = trans.response().write
            self._CHEETAH__isBuffering = _wasBuffering_69257830 
            writeLoopContents = _captureCollector_69257830.response().getvalue()
            del _orig_trans_69257830
            del _captureCollector_69257830
            del _wasBuffering_69257830
            # 
            #  Permit the driver to modify the writeLoopContents
            featureOrdering = ['Driver']
            dict = {'writeLoopContents': writeLoopContents,                  'dimRepOrdering': dimRepOrdering}
            VFFSL(SL,"insertCodeForFeatures",False)('writeDataHDF5ModifyLoopContents', featureOrdering, dict)
            writeLoopContents = dict['writeLoopContents']
            # 
            #  The object passed as the first argument to this next call is a 
            write('''// Select hyperslabs of memory and file data spaces for data transfer operation
''')
            _v = VFFSL(SL,"splitUniformDataSelect",False)(dimRepOrdering[:], writeLoopContents, dimensionOffsets) # u'${splitUniformDataSelect(dimRepOrdering[:], writeLoopContents, dimensionOffsets)}' on line 151, col 1
            if _v is not None: write(_filter(_v, rawExpr='${splitUniformDataSelect(dimRepOrdering[:], writeLoopContents, dimensionOffsets)}')) # from line 151, col 1.
            write('''
H5Sclose(mem_dataspace);
''')
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def splitUniformDataSelect(self, remainingDimReps, writeLoopContents, dimensionOffsets, **KWS):



        ## CHEETAH: generated from @def splitUniformDataSelect(remainingDimReps, writeLoopContents, dimensionOffsets) at line 157, col 1.
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
        
        if not remainingDimReps: # generated from line 158, col 3
            _v = VFFSL(SL,"writeLoopContents",True) # u'${writeLoopContents}' on line 159, col 1
            if _v is not None: write(_filter(_v, rawExpr='${writeLoopContents}')) # from line 159, col 1.
        else: # generated from line 160, col 3
            fileDimNum, memDimNum, dimRep = remainingDimReps.pop(0)
            if isinstance(dimRep, SplitUniformDimensionRepresentation): # generated from line 162, col 5
                write('''for (bool _positive_''')
                _v = VFFSL(SL,"dimRep.name",True) # u'${dimRep.name}' on line 163, col 21
                if _v is not None: write(_filter(_v, rawExpr='${dimRep.name}')) # from line 163, col 21.
                write(''' = true; ; _positive_''')
                _v = VFFSL(SL,"dimRep.name",True) # u'${dimRep.name}' on line 163, col 56
                if _v is not None: write(_filter(_v, rawExpr='${dimRep.name}')) # from line 163, col 56.
                write(''' = false) {
  if (_positive_''')
                _v = VFFSL(SL,"dimRep.name",True) # u'${dimRep.name}' on line 164, col 17
                if _v is not None: write(_filter(_v, rawExpr='${dimRep.name}')) # from line 164, col 17.
                write(''') {
    mem_start[''')
                _v = VFFSL(SL,"memDimNum",True) # u'$memDimNum' on line 165, col 15
                if _v is not None: write(_filter(_v, rawExpr='$memDimNum')) # from line 165, col 15.
                write('''] = 0;
    if (''')
                _v = VFFSL(SL,"dimRep.localOffset",True) # u'$dimRep.localOffset' on line 166, col 9
                if _v is not None: write(_filter(_v, rawExpr='$dimRep.localOffset')) # from line 166, col 9.
                write(''' >= ((''')
                _v = VFFSL(SL,"dimRep.globalLattice",True) # u'$dimRep.globalLattice' on line 166, col 34
                if _v is not None: write(_filter(_v, rawExpr='$dimRep.globalLattice')) # from line 166, col 34.
                write('''-1)/2 +1)) // No positive values are stored in this rank.
      continue;
    mem_count[''')
                _v = VFFSL(SL,"memDimNum",True) # u'$memDimNum' on line 168, col 15
                if _v is not None: write(_filter(_v, rawExpr='$memDimNum')) # from line 168, col 15.
                write('''] = MIN(((''')
                _v = VFFSL(SL,"dimRep.globalLattice",True) # u'$dimRep.globalLattice' on line 168, col 35
                if _v is not None: write(_filter(_v, rawExpr='$dimRep.globalLattice')) # from line 168, col 35.
                write('''-1)/2 +1) - ''')
                _v = VFFSL(SL,"dimRep.localOffset",True) # u'$dimRep.localOffset' on line 168, col 68
                if _v is not None: write(_filter(_v, rawExpr='$dimRep.localOffset')) # from line 168, col 68.
                write(''', ''')
                _v = VFFSL(SL,"dimRep.localLattice",True) # u'$dimRep.localLattice' on line 168, col 89
                if _v is not None: write(_filter(_v, rawExpr='$dimRep.localLattice')) # from line 168, col 89.
                write(''');
    file_start[''')
                _v = VFFSL(SL,"fileDimNum",True) # u'$fileDimNum' on line 169, col 16
                if _v is not None: write(_filter(_v, rawExpr='$fileDimNum')) # from line 169, col 16.
                write('''] = file_dims[''')
                _v = VFFSL(SL,"fileDimNum",True) # u'$fileDimNum' on line 169, col 41
                if _v is not None: write(_filter(_v, rawExpr='$fileDimNum')) # from line 169, col 41.
                write(''']/2 + ''')
                _v = VFFSL(SL,"dimRep.localOffset",True) # u'$dimRep.localOffset' on line 169, col 58
                if _v is not None: write(_filter(_v, rawExpr='$dimRep.localOffset')) # from line 169, col 58.
                write(''';
''')
                if dimRep.name in dimensionOffsets: # generated from line 170, col 7
                    write('''    if (''')
                    _v = VFFSL(SL,"dimRep.localOffset",True) # u'$dimRep.localOffset' on line 171, col 9
                    if _v is not None: write(_filter(_v, rawExpr='$dimRep.localOffset')) # from line 171, col 9.
                    write(''' > ((file_dims[''')
                    _v = VFFSL(SL,"fileDimNum",True) # u'$fileDimNum' on line 171, col 43
                    if _v is not None: write(_filter(_v, rawExpr='$fileDimNum')) # from line 171, col 43.
                    write(''']-1)/2 +1))
      continue;
    mem_count[''')
                    _v = VFFSL(SL,"memDimNum",True) # u'$memDimNum' on line 173, col 15
                    if _v is not None: write(_filter(_v, rawExpr='$memDimNum')) # from line 173, col 15.
                    write('''] = MIN(mem_count[''')
                    _v = VFFSL(SL,"memDimNum",True) # u'$memDimNum' on line 173, col 43
                    if _v is not None: write(_filter(_v, rawExpr='$memDimNum')) # from line 173, col 43.
                    write('''], ((file_dims[''')
                    _v = VFFSL(SL,"fileDimNum",True) # u'$fileDimNum' on line 173, col 68
                    if _v is not None: write(_filter(_v, rawExpr='$fileDimNum')) # from line 173, col 68.
                    write(''']-1)/2 +1) - ''')
                    _v = VFFSL(SL,"dimRep.localOffset",True) # u'$dimRep.localOffset' on line 173, col 92
                    if _v is not None: write(_filter(_v, rawExpr='$dimRep.localOffset')) # from line 173, col 92.
                    write(''');
''')
                write('''  } else {
    if ((''')
                _v = VFFSL(SL,"dimRep.globalLattice",True) # u'$dimRep.globalLattice' on line 176, col 10
                if _v is not None: write(_filter(_v, rawExpr='$dimRep.globalLattice')) # from line 176, col 10.
                write('''-1)/2+1 < ''')
                _v = VFFSL(SL,"dimRep.localOffset",True) # u'$dimRep.localOffset' on line 176, col 41
                if _v is not None: write(_filter(_v, rawExpr='$dimRep.localOffset')) # from line 176, col 41.
                write(''') // Only negative values are stored in this rank.
      mem_start[''')
                _v = VFFSL(SL,"memDimNum",True) # u'$memDimNum' on line 177, col 17
                if _v is not None: write(_filter(_v, rawExpr='$memDimNum')) # from line 177, col 17.
                write('''] = 0;
    else
      mem_start[''')
                _v = VFFSL(SL,"memDimNum",True) # u'$memDimNum' on line 179, col 17
                if _v is not None: write(_filter(_v, rawExpr='$memDimNum')) # from line 179, col 17.
                write('''] = (''')
                _v = VFFSL(SL,"dimRep.globalLattice",True) # u'$dimRep.globalLattice' on line 179, col 32
                if _v is not None: write(_filter(_v, rawExpr='$dimRep.globalLattice')) # from line 179, col 32.
                write('''-1)/2+1 - ''')
                _v = VFFSL(SL,"dimRep.localOffset",True) # u'$dimRep.localOffset' on line 179, col 63
                if _v is not None: write(_filter(_v, rawExpr='$dimRep.localOffset')) # from line 179, col 63.
                write(''';
    file_start[''')
                _v = VFFSL(SL,"fileDimNum",True) # u'$fileDimNum' on line 180, col 16
                if _v is not None: write(_filter(_v, rawExpr='$fileDimNum')) # from line 180, col 16.
                write('''] = mem_start[''')
                _v = VFFSL(SL,"memDimNum",True) # u'$memDimNum' on line 180, col 41
                if _v is not None: write(_filter(_v, rawExpr='$memDimNum')) # from line 180, col 41.
                write('''] + ''')
                _v = VFFSL(SL,"dimRep.localOffset",True) # u'$dimRep.localOffset' on line 180, col 55
                if _v is not None: write(_filter(_v, rawExpr='$dimRep.localOffset')) # from line 180, col 55.
                write(''' - ((''')
                _v = VFFSL(SL,"dimRep.globalLattice",True) # u'$dimRep.globalLattice' on line 180, col 79
                if _v is not None: write(_filter(_v, rawExpr='$dimRep.globalLattice')) # from line 180, col 79.
                write('''-1)/2+1);
    if (''')
                _v = VFFSL(SL,"dimRep.localLattice",True) # u'$dimRep.localLattice' on line 181, col 9
                if _v is not None: write(_filter(_v, rawExpr='$dimRep.localLattice')) # from line 181, col 9.
                write(''' <= mem_start[''')
                _v = VFFSL(SL,"memDimNum",True) # u'$memDimNum' on line 181, col 43
                if _v is not None: write(_filter(_v, rawExpr='$memDimNum')) # from line 181, col 43.
                write(''']) // No negative values are stored in this rank.
      break; // end loop over this dimension
    mem_count[''')
                _v = VFFSL(SL,"memDimNum",True) # u'$memDimNum' on line 183, col 15
                if _v is not None: write(_filter(_v, rawExpr='$memDimNum')) # from line 183, col 15.
                write('''] = ''')
                _v = VFFSL(SL,"dimRep.localLattice",True) # u'$dimRep.localLattice' on line 183, col 29
                if _v is not None: write(_filter(_v, rawExpr='$dimRep.localLattice')) # from line 183, col 29.
                write(''' - mem_start[''')
                _v = VFFSL(SL,"memDimNum",True) # u'$memDimNum' on line 183, col 62
                if _v is not None: write(_filter(_v, rawExpr='$memDimNum')) # from line 183, col 62.
                write(''']; // To the end of this dimension
''')
                if dimRep.name in dimensionOffsets: # generated from line 184, col 7
                    write('''    if ((long)file_start[''')
                    _v = VFFSL(SL,"fileDimNum",True) # u'$fileDimNum' on line 185, col 26
                    if _v is not None: write(_filter(_v, rawExpr='$fileDimNum')) # from line 185, col 26.
                    write('''] > ''')
                    _v = VFFSL(SL,"dimensionOffsets",True)[dimRep.name] # u'${dimensionOffsets[dimRep.name]}' on line 185, col 41
                    if _v is not None: write(_filter(_v, rawExpr='${dimensionOffsets[dimRep.name]}')) # from line 185, col 41.
                    write(''')
      file_start[''')
                    _v = VFFSL(SL,"fileDimNum",True) # u'$fileDimNum' on line 186, col 18
                    if _v is not None: write(_filter(_v, rawExpr='$fileDimNum')) # from line 186, col 18.
                    write('''] -= ''')
                    _v = VFFSL(SL,"dimensionOffsets",True)[dimRep.name] # u'${dimensionOffsets[dimRep.name]}' on line 186, col 34
                    if _v is not None: write(_filter(_v, rawExpr='${dimensionOffsets[dimRep.name]}')) # from line 186, col 34.
                    write(''';
    else {
      mem_start[''')
                    _v = VFFSL(SL,"memDimNum",True) # u'$memDimNum' on line 188, col 17
                    if _v is not None: write(_filter(_v, rawExpr='$memDimNum')) # from line 188, col 17.
                    write('''] += ''')
                    _v = VFFSL(SL,"dimensionOffsets",True)[dimRep.name] # u'${dimensionOffsets[dimRep.name]}' on line 188, col 32
                    if _v is not None: write(_filter(_v, rawExpr='${dimensionOffsets[dimRep.name]}')) # from line 188, col 32.
                    write(''' - file_start[''')
                    _v = VFFSL(SL,"fileDimNum",True) # u'$fileDimNum' on line 188, col 78
                    if _v is not None: write(_filter(_v, rawExpr='$fileDimNum')) # from line 188, col 78.
                    write('''];
      if (mem_count[''')
                    _v = VFFSL(SL,"memDimNum",True) # u'$memDimNum' on line 189, col 21
                    if _v is not None: write(_filter(_v, rawExpr='$memDimNum')) # from line 189, col 21.
                    write('''] > ''')
                    _v = VFFSL(SL,"dimensionOffsets",True)[dimRep.name] # u'${dimensionOffsets[dimRep.name]}' on line 189, col 35
                    if _v is not None: write(_filter(_v, rawExpr='${dimensionOffsets[dimRep.name]}')) # from line 189, col 35.
                    write(''' - file_start[''')
                    _v = VFFSL(SL,"fileDimNum",True) # u'$fileDimNum' on line 189, col 81
                    if _v is not None: write(_filter(_v, rawExpr='$fileDimNum')) # from line 189, col 81.
                    write('''])
        mem_count[''')
                    _v = VFFSL(SL,"memDimNum",True) # u'$memDimNum' on line 190, col 19
                    if _v is not None: write(_filter(_v, rawExpr='$memDimNum')) # from line 190, col 19.
                    write('''] -= ''')
                    _v = VFFSL(SL,"dimensionOffsets",True)[dimRep.name] # u'${dimensionOffsets[dimRep.name]}' on line 190, col 34
                    if _v is not None: write(_filter(_v, rawExpr='${dimensionOffsets[dimRep.name]}')) # from line 190, col 34.
                    write(''' - file_start[''')
                    _v = VFFSL(SL,"fileDimNum",True) # u'$fileDimNum' on line 190, col 80
                    if _v is not None: write(_filter(_v, rawExpr='$fileDimNum')) # from line 190, col 80.
                    write('''];
      else
        break; // end loop over this dimension
      file_start[''')
                    _v = VFFSL(SL,"fileDimNum",True) # u'$fileDimNum' on line 193, col 18
                    if _v is not None: write(_filter(_v, rawExpr='$fileDimNum')) # from line 193, col 18.
                    write('''] = 0;
    }
''')
                write('''  }
  
  ''')
                _v = VFFSL(SL,"splitUniformDataSelect",False)(remainingDimReps, writeLoopContents, dimensionOffsets) # u'${splitUniformDataSelect(remainingDimReps, writeLoopContents, dimensionOffsets), autoIndent=True}' on line 198, col 3
                if _v is not None: write(_filter(_v, autoIndent=True, rawExpr='${splitUniformDataSelect(remainingDimReps, writeLoopContents, dimensionOffsets), autoIndent=True}')) # from line 198, col 3.
                write('''  
  if (!_positive_''')
                _v = VFFSL(SL,"dimRep.name",True) # u'${dimRep.name}' on line 200, col 18
                if _v is not None: write(_filter(_v, rawExpr='${dimRep.name}')) # from line 200, col 18.
                write(''')
    break;
}
''')
            else: # generated from line 203, col 5
                _v = VFFSL(SL,"splitUniformDataSelect",False)(remainingDimReps, writeLoopContents, dimensionOffsets) # u'${splitUniformDataSelect(remainingDimReps, writeLoopContents, dimensionOffsets)}' on line 204, col 1
                if _v is not None: write(_filter(_v, rawExpr='${splitUniformDataSelect(remainingDimReps, writeLoopContents, dimensionOffsets)}')) # from line 204, col 1.
        
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
        # HDF5.tmpl
        # 
        # Created by Graham Dennis on 2008-03-28.
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

    uselib = ['hdf5']

    _mainCheetahMethod_for_HDF5= 'writeBody'

## END CLASS DEFINITION

if not hasattr(HDF5, '_initCheetahAttributes'):
    templateAPIClass = getattr(HDF5, '_CHEETAH_templateClass', Template)
    templateAPIClass._addCheetahPlumbingCodeToClass(HDF5)


# CHEETAH was developed by Tavis Rudd and Mike Orr
# with code, advice and input from many other volunteers.
# For more information visit http://www.CheetahTemplate.org/

##################################################
## if run from command line:
if __name__ == '__main__':
    from Cheetah.TemplateCmdLineIface import CmdLineIface
    CmdLineIface(templateObj=HDF5()).run()


