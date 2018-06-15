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
from xpdeint.SimulationDrivers.MPIMultiPathDriver import MPIMultiPathDriver

##################################################
## MODULE CONSTANTS
VFFSL=valueFromFrameOrSearchList
VFSL=valueFromSearchList
VFN=valueForName
currentTime=time.time
__CHEETAH_version__ = '2.4.4'
__CHEETAH_versionTuple__ = (2, 4, 4, 'development', 0)
__CHEETAH_genTime__ = 1484975072.577072
__CHEETAH_genTimestamp__ = 'Sat Jan 21 16:04:32 2017'
__CHEETAH_src__ = '/home/mattias/xmds-2.2.3/admin/staging/xmds-2.2.3/xpdeint/SimulationDrivers/AdaptiveMPIMultiPathDriver.tmpl'
__CHEETAH_srcLastModified__ = 'Mon Nov 18 20:57:44 2013'
__CHEETAH_docstring__ = 'Autogenerated by Cheetah: The Python-Powered Template Engine'

if __CHEETAH_versionTuple__ < RequiredCheetahVersionTuple:
    raise AssertionError(
      'This template was compiled with Cheetah version'
      ' %s. Templates compiled before version %s must be recompiled.'%(
         __CHEETAH_version__, RequiredCheetahVersion))

##################################################
## CLASSES

class AdaptiveMPIMultiPathDriver(MPIMultiPathDriver):

    ##################################################
    ## CHEETAH GENERATED METHODS


    def __init__(self, *args, **KWs):

        super(AdaptiveMPIMultiPathDriver, self).__init__(*args, **KWs)
        if not self._CHEETAH__instanceInitialized:
            cheetahKWArgs = {}
            allowedKWs = 'searchList namespaces filter filtersLib errorCatcher'.split()
            for k,v in list(KWs.items()):
                if k in allowedKWs: cheetahKWArgs[k] = v
            self._initCheetahInstance(**cheetahKWArgs)
        

    def description(self, **KWS):



        ## Generated from @def description: Adaptive MPI Multipath Simulation Driver at line 25, col 1.
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
        
        write('''Adaptive MPI Multipath Simulation Driver''')
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def seedOffset(self, dict, **KWS):



        ## CHEETAH: generated from @def seedOffset($dict) at line 31, col 1.
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
        write(''' + _rank + (_size * _batches_run)''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def segment0_loop(self, **KWS):



        ## CHEETAH: generated from @def segment0_loop at line 37, col 1.
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
        write('''for (long _i0 = ''')
        _v = VFFSL(SL,"pathLoopStart",True) # u'${pathLoopStart}' on line 39, col 17
        if _v is not None: write(_filter(_v, rawExpr='${pathLoopStart}')) # from line 39, col 17.
        write('''; _i0 < ''')
        _v = VFFSL(SL,"pathLoopEnd",True) # u'${pathLoopEnd}' on line 39, col 41
        if _v is not None: write(_filter(_v, rawExpr='${pathLoopEnd}')) # from line 39, col 41.
        write('''; _i0+=''')
        _v = VFFSL(SL,"pathLoopStep",True) # u'${pathLoopStep}' on line 39, col 62
        if _v is not None: write(_filter(_v, rawExpr='${pathLoopStep}')) # from line 39, col 62.
        write(''') {
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def runningSimulationCode(self, **KWS):



        ## CHEETAH: generated from @def runningSimulationCode at line 43, col 1.
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
        write('''if (_size > 1){
  if (_rank == 0)
    _master();
  else
    _slave();
  
  _reduce();
}
else
{
  _local_schedule = _n_paths;
  _segment0();
}
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def segment0End(self, **KWS):



        ## CHEETAH: generated from @def segment0End at line 61, col 1.
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
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def functionPrototypes(self, **KWS):



        ## CHEETAH: generated from @def functionPrototypes at line 66, col 1.
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
        _v = super(AdaptiveMPIMultiPathDriver, self).functionPrototypes()
        if _v is not None: write(_filter(_v))
        # 
        write('''void _master();
void *_mslave(void *ptr); 
void _slave();

void _reduce();
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def globals(self, **KWS):



        ## CHEETAH: generated from @def globals at line 78, col 1.
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
        _v = super(AdaptiveMPIMultiPathDriver, self).globals()
        if _v is not None: write(_filter(_v))
        # 
        write('''
pthread_mutex_t tasklock;    /*Ensures mutual exclusion when assigning tasks*/
pthread_mutex_t finlock;     /*Lock to synchronize completion of thread and master*/

int paths_assigned=0;        /*number of full paths assigned or completed*/

long _local_schedule;        /*current batch size for a slave*/
long _batches_run=0;         /*number of batches a slave has run*/
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def topLevelSegmentFunctionImplementation(self, **KWS):



        ## CHEETAH: generated from @def topLevelSegmentFunctionImplementation at line 93, col 1.
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
        write('''void _segment0()
{
  _LOG(_PATH_LOG_LEVEL, "Running %li paths\\n", ''')
        _v = VFFSL(SL,"pathLoopEnd",True) # u'${pathLoopEnd}' on line 97, col 48
        if _v is not None: write(_filter(_v, rawExpr='${pathLoopEnd}')) # from line 97, col 48.
        write(''');
''')
        #  And now insert the code for the features that apply in the top level sequence
        featureOrdering = ['ErrorCheck', 'Stochastic']
        dict = {'extraIndent': 0}
        write('''  ''')
        _v = VFFSL(SL,"insertCodeForFeatures",False)('topLevelSequenceBegin', featureOrdering, dict) # u"${insertCodeForFeatures('topLevelSequenceBegin', featureOrdering, dict), autoIndent=True}" on line 101, col 3
        if _v is not None: write(_filter(_v, autoIndent=True, rawExpr="${insertCodeForFeatures('topLevelSequenceBegin', featureOrdering, dict), autoIndent=True}")) # from line 101, col 3.
        extraIndent = dict['extraIndent']
        write('''  
  ''')
        _v = VFFSL(SL,"topLevelSegmentPathLoop",True) # u'${topLevelSegmentPathLoop, autoIndent=True, extraIndent=extraIndent}' on line 104, col 3
        if _v is not None: write(_filter(_v, autoIndent=True, extraIndent=extraIndent, rawExpr='${topLevelSegmentPathLoop, autoIndent=True, extraIndent=extraIndent}')) # from line 104, col 3.
        write('''  
  ''')
        _v = VFFSL(SL,"insertCodeForFeaturesInReverseOrder",False)('topLevelSequenceEnd', featureOrdering, dict) # u"${insertCodeForFeaturesInReverseOrder('topLevelSequenceEnd', featureOrdering, dict), autoIndent=True}" on line 106, col 3
        if _v is not None: write(_filter(_v, autoIndent=True, rawExpr="${insertCodeForFeaturesInReverseOrder('topLevelSequenceEnd', featureOrdering, dict), autoIndent=True}")) # from line 106, col 3.
        write('''  
  ''')
        _v = VFFSL(SL,"segment0End",True) # u'${segment0End}' on line 108, col 3
        if _v is not None: write(_filter(_v, rawExpr='${segment0End}')) # from line 108, col 3.
        write('''
}
''')
        # 
        
        ########################################
        ## END - generated method body
        
        return _dummyTrans and trans.response().getvalue() or ""
        

    def functionImplementations(self, **KWS):



        ## CHEETAH: generated from @def functionImplementations at line 113, col 1.
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
        _v = super(AdaptiveMPIMultiPathDriver, self).functionImplementations()
        if _v is not None: write(_filter(_v))
        # 
        write('''
void _reduce()
{
  ''')
        _v = VFFSL(SL,"segment0ReduceBlock",True) # u'${segment0ReduceBlock}' on line 120, col 3
        if _v is not None: write(_filter(_v, rawExpr='${segment0ReduceBlock}')) # from line 120, col 3.
        write('''
}

void _master() 
{  
  int outstanding = 0;        /*number of slaves that are still processing tasks*/
  int *schedule = new int[_size];        /*Batch size scheduled for each slave [reset every iteration]*/
  double *timing = new double[_size];       /*Timing function to determine computation to communication ratio*/
  int *partitions = new int[_size];      /*Batch size scheduled for each slave [resetted after completion]*/

  int i, j;                   /*indexes*/

  MPI_Status *stats = new MPI_Status[_size];    /*MPI Structures*/
  MPI_Request *reqs = new MPI_Request[_size];
  int *indices = new int[_size];
  int ndone;
  double *bufs = new double[_size];         /*MPI Input buffer*/
  
  int *slave_stat = new int[_size];
  double *throughput = new double[_size];
  double *commave = new double[_size];

  double schedtime=0.0;      /*time spent deciding and dispatching schedules*/
  double commtime=0.0;       /*index for communication latency*/
  double totaltime=0.0;      /*index for seconds per schedule*/
  double totalcommtime=0.0;  /*total communication latency*/
  double paratime=0.0;       /*total parallel walltime for slaves excluding mslave*/

  /************* Scheduling Parameters **************/
  double calpha = 0.2;        /*weighting for communication average*/
  double talpha = 0.2;        /*weighting for throughput average*/

  double epsilon = 0.005;     /*maximum tolerated communication overhead*/
  double lower = 2.0;         /*minimum tolerated resolution in seconds*/
  double upper = 10.0;        /*maximum tolerated resolution seconds*/
  /***************************************************/

  double tp1, tp2;

  //Initialise slave status arrays
  for (i=0; i<_size; i++){
    slave_stat[i]=0;
    partitions[i]=0;
    commave[i]=0.0;
    throughput[i]=0.0;
  }

  //pthread is always busy doing something
  slave_stat[0] = 1;

  /************* PThread Initialization **************/
  pthread_t helper;
  
  //Initialise mutual exclusion mechanism
  pthread_mutex_init(&tasklock, NULL);
  pthread_mutex_init(&finlock, NULL);
  pthread_mutex_lock(&finlock);

  //Create a thread to act as a slave
  if (pthread_create(&helper, NULL, _mslave, NULL)!=0)
    _LOG(_ERROR_LOG_LEVEL, "Thread creation failed\\n");
  
  //Listen for messages from all slaves
  for (i=0; i<_size; i++){
    MPI_Irecv(&bufs[i], 1, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, &reqs[i]);
  }
  
  //Loop until all paths are finished, and all results recieved
  //      -Test for messages from slaves
  //      -Determine path schedule for idle slaves
  //      -Send schedule to slaves
  
  /**************LISTEN FOR "TASKS COMPLETED" MESSAGES FROM SLAVES*******************/
  
  while (paths_assigned < _n_paths || outstanding > 0){
    
    //Wait for messages from slaves
    if (outstanding > 0){
      MPI_Waitsome(_size, reqs, &ndone, indices, stats);
      
      for (i=0; i<ndone; i++){
        //Deal with incoming messages
        j = indices[i];

        //Dynamically determine bandwidth and throughput
        totaltime = MPI_Wtime() - timing[j];
        commtime = totaltime - bufs[j];

        //Calculate average communication time and average throughput
        if (commave[j] == 0.0){
          commave[j] = commtime;
          throughput[j] = partitions[j]/totaltime;
        }
        else {
          commave[j] = commave[j] * (1 - calpha) + (commtime * calpha);
          throughput[j] = throughput[j] * (1 - talpha) + (partitions[j] / totaltime) * talpha;
        }
        
        totalcommtime += commtime;
        paratime += totaltime;
        slave_stat[j] = 0;
        outstanding--;
        MPI_Irecv(&bufs[j], 1, MPI_DOUBLE, j, MPI_ANY_TAG, MPI_COMM_WORLD, &reqs[j]);
      }
    }
    
    // If no more tasks need to be assigned continue to listen for messages from slaves
    if (paths_assigned >= _n_paths)
      continue;
    
    /********************SCHEDULE MORE TASKS FOR IDLE SLAVES***********************/
    
    for (i=0; i<_size; i++){
      schedule[i]=0;
    }
    tp1 = MPI_Wtime();

    //allocate tasks to free processors
    //scheduling must be mutually exclusive as the slave thread
    //also modifies the global variables below for self-scheduling
    pthread_mutex_lock(&tasklock);
    
    for (i=1; i<_size; i++){
      if (paths_assigned >= _n_paths)
        break;
      
      //only allocate more tasks to slaves that are idle
      if (slave_stat[i] == 0){
        slave_stat[i]=1;
        
        //Determine new batch size based on slave throughput and
        //communication overhead. Preferable estimated computing times
        //for each schedule is high enough to reduce comm overhead
        //and between upper and lower
        partitions[i] = (int) (MAX((commave[i]*throughput[i])/epsilon, throughput[i]*lower));
        partitions[i] = (int) (MIN(partitions[i], throughput[i]*upper));
        partitions[i] = (int) (MAX(partitions[i], 1));
        
        if (paths_assigned + partitions[i] > _n_paths){
          partitions[i] = _n_paths - paths_assigned;
        }
        
        schedule[i] = partitions[i];
        paths_assigned += partitions[i];
      }
    }
    
    pthread_mutex_unlock(&tasklock);
    
    /**************************SEND SCHEDULE TO SLAVE(S)********************/
    for (i=1; i<_size; i++){
      if (schedule[i]>0){
        timing[i] = MPI_Wtime();
        MPI_Send(&schedule[i], 1, MPI_INT, i, 1, MPI_COMM_WORLD);
        outstanding++;
      }
    }
    tp2 = MPI_Wtime() - tp1;
    schedtime += tp2;
  }
  
  //Block until the thread slave has completed processing
  pthread_mutex_lock(&finlock);
  
  //Tell slave processes to Reduce then exit
  for (i=1; i<_size; i++){
    MPI_Send(NULL, 0, MPI_INT, i, 0, MPI_COMM_WORLD);
  }

  //Kill slave thread
  pthread_cancel(helper);
  
  delete[] schedule;
  delete[] timing;
  delete[] partitions;
  delete[] stats;
  delete[] reqs;
  delete[] indices;
  delete[] bufs;
  delete[] slave_stat;
  delete[] throughput;
  delete[] commave;
}

void *_mslave(void *ptr) 
{
  double thr_throughput=0.0; 
  double thr_time_per_batch=2.0; 
  double thr_talpha=0.2; 
  
  int i; 
  double tp1, tp2;
  
  _local_schedule = 1; 
  
  while(paths_assigned < _n_paths)
  { 
    //Self schedule more tasks to process
    tp1 = MPI_Wtime();
    
    /********************SCHEDULE MORE TASKS*************************/
    pthread_mutex_lock(&tasklock);
    
    if (paths_assigned >= _n_paths){
      pthread_mutex_unlock(&tasklock);
      break;
    }
    
    if (paths_assigned + _local_schedule > _n_paths)
      _local_schedule = _n_paths - paths_assigned;
    
    paths_assigned += _local_schedule;
    
    pthread_mutex_unlock(&tasklock);
    /*****************************************************************/
    
    _segment0();
    
    _batches_run++;
    
    tp2 = MPI_Wtime() - tp1;
    
    /********************CALCULATE NEW BATCH SIZE*********************/
    
    if (thr_throughput == 0.0)
      thr_throughput = _local_schedule/tp2;
    else
      thr_throughput = (1-thr_talpha)*thr_throughput + thr_talpha * (_local_schedule/tp2);
    
    _local_schedule = MAX(1, (int) (thr_throughput * thr_time_per_batch));
    
    /*****************************************************************/ 
  }
  
  //Unlocking indicates that the thread slave has finished processing 
  pthread_mutex_unlock(&finlock);
  
  return NULL;
}

void _slave() 
{
  MPI_Status stat;
  
  double tp1, tp2;
  
  while(1) {
    //Wait for initial communication from master
    MPI_Recv(&_local_schedule, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
    
    tp1 = MPI_Wtime();    
    
    if (stat.MPI_TAG == 0)
      break;
    
    _segment0();
    
    _batches_run++;
    
    tp2 = MPI_Wtime() - tp1;
    
    //Send completion notice to the master
    MPI_Send(&tp2, 1, MPI_DOUBLE, 0, stat.MPI_TAG, MPI_COMM_WORLD);
  }
  
  //Tell master that slave is done
  MPI_Send(NULL, 0, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
}
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
        # AdaptiveMPIMultiPathDriver.tmpl
        # 
        # Created by Liam Madge on 2013-09-20.
        # Modified by Gregory Bogomiagkov on 2013-10-12
        # 
        # Copyright (c) 2008-2013, Graham Dennis
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

    pathLoopStart = '0'

    pathLoopStep = '1'

    pathLoopEnd = '_local_schedule'

    _mainCheetahMethod_for_AdaptiveMPIMultiPathDriver= 'writeBody'

## END CLASS DEFINITION

if not hasattr(AdaptiveMPIMultiPathDriver, '_initCheetahAttributes'):
    templateAPIClass = getattr(AdaptiveMPIMultiPathDriver, '_CHEETAH_templateClass', Template)
    templateAPIClass._addCheetahPlumbingCodeToClass(AdaptiveMPIMultiPathDriver)


# CHEETAH was developed by Tavis Rudd and Mike Orr
# with code, advice and input from many other volunteers.
# For more information visit http://www.CheetahTemplate.org/

##################################################
## if run from command line:
if __name__ == '__main__':
    from Cheetah.TemplateCmdLineIface import CmdLineIface
    CmdLineIface(templateObj=AdaptiveMPIMultiPathDriver()).run()


