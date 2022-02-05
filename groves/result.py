# -*- coding: utf-8 -*-

from pathlib import PurePath,Path
import json
from json import JSONEncoder
import os
import abc

import numpy as np
import pandas as pd
import tifffile as tif


class ReaderWriter(abc.ABC):
    '''Abstract base class for data reader/writer, so that you can 
    write the processed file and read that file afterwards for farther processing,
    using the same class/instance.
    Abstractmethods must be overridden in the subclasses.
    '''
    def __init__(self,path,mode='rw'):
        """
        Args:
            path (str|PurePath|Path): path to the file
            mode (str, optional): ['rw','r','w']. specify if read-only or write-only.
                Defaults to 'rw'.
        """        
        self._assertpath(path)
        self.data=None
        assert mode in ['rw','r','w']
        self.mode = mode

    def set(self,data):
        """Set data to the field self.data
        If data is not the expected type, _assertdata() method will raise error

        Args:
            data (any): will be specified in subclasses.

        Raises:
            PermissionError: if read-only
        """        ''''''
        if self.mode == 'r':
            raise PermissionError('read-only')
        self._assertdata(data)
        self.data = data

    def get(self):
        """Returns stored data.

        Raises:
            PermissionError: if write-only

        Returns:
            self.data: data being stored in this object
        """        
        if self.mode == 'w':
            raise PermissionError('write-only')
        if self.data is None:
            print('no data: ',self.path)
        return self.data

    def read(self):
        """Read the file at self.path and store as self.data.

        Raises:
            PermissionError: if write-only
            FileNotFoundError: if self.path does not exist.
        """
        if self.mode == 'w':
            raise PermissionError('write-only')
        if not os.path.exists(self.path):
            raise FileNotFoundError(self.path)
        self._readmain()

    def write(self,force=False):
        """Write the data being stored in this object to the self.path.

        Args:
            force (bool, optional): You will be asked if you overwrite the existing data in the command line.
            Defaults to False.

        Raises:
            PermissionError: if read-only
        """        
        if self.mode == 'r':
            raise PermissionError('read-only')
        self._assertdata(self.data)
        if (not force) and os.path.exists(self.path):
            a = input(self.path + '\n overwrite? [y/n]:')
            if a!='y':
                return
        self._writemain()
    
    def readget(self):
        '''utility method. read() then returns get()'''
        self.read()
        return self.get()
    
    def setwrite(self,data,force=False):
        '''utility method. set(data) then write(force).'''
        self.set(data)
        self.write(force=force)

    def _assertpath(self,path):
        """assert the path and store its contents as fields.

        Args:
            path (str|PurePath|Path): path to the file

        Raises:
            TypeError: if path type is invalid
            FileNotFoundError: if parent directory does not exist.
        """        
        if isinstance(path,PurePath):
            self.path = str(path)
        elif isinstance(path,str):
            self.path = path
        else:
            raise TypeError('invalid path type, neither PurePath or str')

        ext = os.path.splitext(self.path)[1]
        self._assertext(ext)
        direc = os.path.split(self.path)[0]
        if not Path(str(direc)).exists():
            raise FileNotFoundError(f'directory {direc} not found')
        
        self.ext = ext
        self.direc = direc

    @abc.abstractmethod
    def _assertext(self,ext):
        """Only called internally. This method asserts the file extention.

        Args:
            ext (str): extension ('.***')

        """        
        pass
    @abc.abstractmethod
    def _assertdata(self,data):
        """Only called internally. This method asserts the data before set and write.

        Args:
            ext (any): data to test
        """
        pass
    @abc.abstractmethod
    def _readmain(self):
        """Only called internally. Read the data.
        """        
        pass
    @abc.abstractmethod
    def _writemain(self):
        """Only called internaly. Write the data.
        """        
        pass



class Null(ReaderWriter):
    """Null ReaderWriter doing nothing.
    """
    def _assertext(self, ext):
        pass
    def _assertdata(self, data):
        pass
    def _readmain(self):
        pass
    def _writemain(self):
        pass

class ImBinary(ReaderWriter):
    """Binary ndarray(bool) - tif image file.
    """
    def _assertext(self, ext):
        assert ext == '.tif'
    def _assertdata(self, data):
        assert isinstance(data,np.ndarray)
        assert data.dtype is np.dtype(np.bool_)
    def _readmain(self):
        self.data = tif.imread(self.path).astype(bool)
    def _writemain(self):
        tif.imwrite(self.path,self.data.astype(bool))

class ImUint8(ReaderWriter):
    """ndarray(uint8) - uint8 tif image file.
    """
    def _assertext(self, ext):
        assert ext == '.tif'
    def _assertdata(self, data):
        assert isinstance(data,np.ndarray)
        assert data.dtype is np.dtype(np.uint8)
    def _readmain(self):
        self.data = tif.imread(self.path).astype(np.uint8)
    def _writemain(self):
        tif.imwrite(self.path,self.data)

class ImUint16(ReaderWriter):
    """ndarray(uint16) - uint16 tif image file.
    """
    def _assertext(self, ext):
        assert ext == '.tif'
    def _assertdata(self, data):
        assert isinstance(data,np.ndarray)
        assert data.dtype is np.dtype(np.uint16)
    def _readmain(self):
        self.data = tif.imread(self.path).astype(np.uint16)
    def _writemain(self):
        tif.imwrite(self.path,self.data)

class ImInt16(ReaderWriter):
    """ndarray(int16) - int16 tif image file.
    """
    def _assertext(self, ext):
        assert ext == '.tif'
    def _assertdata(self, data):
        assert isinstance(data,np.ndarray)
        assert data.dtype is np.dtype(np.int16)
    def _readmain(self):
        self.data = tif.imread(self.path).astype(np.int16)
    def _writemain(self):
        tif.imwrite(self.path,self.data)

class ImInt32(ReaderWriter):
    """ndarray(int32) - int32 tif image file.
    """
    def _assertext(self, ext):
        assert ext == '.tif'
    def _assertdata(self, data):
        assert isinstance(data,np.ndarray)
        assert data.dtype is np.dtype(np.int32)
    def _readmain(self):
        self.data = tif.imread(self.path).astype(np.int32)
    def _writemain(self):
        tif.imwrite(self.path,self.data)

class ImFloat32(ReaderWriter):
    """ndarray(float32) - float32 tif image file.
    """
    def _assertext(self, ext):
        assert ext == '.tif'
    def _assertdata(self, data):
        assert isinstance(data,np.ndarray)
        assert data.dtype is np.dtype(np.float32)
    def _readmain(self):
        self.data = tif.imread(self.path).astype(np.float32)
    def _writemain(self):
        tif.imwrite(self.path,self.data)

class ImFloat64(ReaderWriter):
    """ndarray(float64) - float64 tif image file.
    """
    def _assertext(self, ext):
        assert ext == '.tif'
    def _assertdata(self, data):
        assert isinstance(data,np.ndarray)
        assert data.dtype is np.dtype(np.float64)
    def _readmain(self):
        self.data = tif.imread(self.path).astype(np.float64)
    def _writemain(self):
        tif.imwrite(self.path,self.data)

class DFrame(ReaderWriter):
    """DataFrame - csv file.
    """
    def _assertext(self, ext):
        assert ext == '.csv'
    def _assertdata(self, data):
        assert isinstance(data,pd.DataFrame)
    def _readmain(self):
        self.data = pd.read_csv(self.path)
    def _writemain(self):
        if len(self.data)==0:
            return
        self.data.to_csv(self.path,index=False)

class NumpyArrayEncoder(JSONEncoder):
    """Json encoder for ndarray -> list conversion.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int64):
            return int(obj)
        return JSONEncoder.default(self, obj)

class Dict(ReaderWriter):
    """dict - json file (.json or .txt).
    """
    def _assertext(self, ext):
        assert ext == '.json' or ext=='.txt'
    def _assertdata(self, data):
        assert isinstance(data,dict)
    def _readmain(self):
        with open(self.path,mode='r') as f:
            self.data = json.loads(f.read())
    def _writemain(self):
        with open(self.path, mode='w') as f:
            t = json.dumps(self.data,cls=NumpyArrayEncoder,indent=4)
            f.write(t)
