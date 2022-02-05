# -*- coding: utf-8 -*-

import json
import os
import xml.etree.ElementTree as ET
import abc
import copy
from pathlib import Path,PurePath

import numpy as np
import tifffile as tif
import pandas as pd



class DataReader(abc.ABC):
    def __init__(self,path,read=False):
        """Abstract base class for reading the source files.    
        read() method reads the file and store in the scope.
        _readmain() method must be overridden to read the file at self.path in the subclasses.
        "get" methods will be implemented in the subclasses.

        Args:
            path (str|PurePath|Path): path for the file
            read (bool, optional): run read() after instanciated if True for convenience. Defaults to False.

        Raises:
            TypeError: path is invalid type
        """
        self.readflag = False

        if isinstance(path,PurePath):
            self.path = str(path)
        elif isinstance(path,str):
            self.path = path
        else:
            raise TypeError('invalid path type, neither Path or str')

        if read:
            self.read()

    def read(self):
        """Read file and store as a field. Internally this just calls self._readmain().
        changes self.readflag to True.

        Raises:
            FileNotFoundError: if file not existing.
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError('no file:' + self.path)
        self._readmain()
        self.readflag = True

    @abc.abstractmethod
    def _readmain(self):
        pass




class RawImage(DataReader):
    '''Scope 8 tiff file.
    read as TCYX ndarray (np.uint16).
    '''
    def _readmain(self):
        self.im = tif.imread(self.path)
        assert isinstance(self.im, np.ndarray)
        assert self.im.dtype is np.dtype(np.uint16)
        # t,c,y,x

    def get_im(self):
        """returns image ndarray

        Returns:
            ndarray: TCYX 4 dimensional.
                The reurned ndarray has full frames.
                It will contain duplicates if any frames are skipped.
                e.g. [0--3--6--] becomes [000333666]

        Raises:
            ValueError: if file not read yet.
        """
        if not self.readflag:
            raise ValueError("file not yet read")
        return self.im

class RawImageStack(DataReader):
    '''Scope 8 saved as separate tiff files.
    read as TCYX ndarray (np.uint16).
    path argument should be the directory to the folder containing images and txt.
    requires "display_and_coments.txt" in the same directory.
    '''
    def _readmain(self):
        self._readtxt()
        self._readim()
    def get_im(self):
        """get image

        Returns:
            ndarray: TCYX

        Raises:
            ValueError: if file not read yet.
        """
        if not self.readflag:
            raise ValueError("file not yet read")
        return self.im
    def _readtxt(self):
        tpath = os.path.join(self.path,'display_and_comments.txt')
        if not os.path.exists(tpath):
            raise FileNotFoundError('display_and_comments.txt not existing')
        with open(tpath,'r') as f:
            t = f.read()
        self.d = json.loads(t)
        l = self.d['Channels'] 
        self.names = [d['Name'] for d in l]
    def _readim(self):
        fnlist = os.listdir(self.path)
        imfn = {n:{} for n in self.names}
        for fn in fnlist:
            l = fn.split('_')
            if l[0]!='img':
                continue
            imfn[l[2]][int(l[1])]=fn
        lastf = 0
        for sub in imfn.values():
            la = max(list(sub.keys()))
            if la > lastf:
                lastf = la
        for sub in imfn.values():
            for i in range(lastf+1):
                if i in sub.keys():
                    cur = sub[i]
                else:
                    sub[i] = cur
        self.im = np.zeros((lastf+1,len(self.names),512,512),dtype=np.uint16)
        for i in range(self.im.shape[1]):
            name = self.names[i]
            for j in range(self.im.shape[0]):
                path = os.path.join(self.path,imfn[name][j])
                self.im[j,i,:,:]=tif.imread(path)


class MetaData(DataReader):
    '''Scope 8 MDA metadata (.txt, json format).
    So far this class does not compile every information in the original file.
    More methods may be added in the future if necessary.
    '''
    def get_eltime(self):
        """get elapsed time in ms unit.

        Returns:
            pd.DataFrame: colomns = Channel names (str) in the original order.
                raws = frames (int)
                contains Elapsed Time (in ms unit, np.float) as written in the metadata file.
                The skipped frame-channel contain np.nan object.

        Raises:
            ValueError: if file not read yet.
        """
        if not self.readflag:
            raise ValueError("file not yet read")
        return self.eltimedf
    def get_skipbool(self):
        """get skipped frame/channel

        Returns:
            pd.DataFrame: 
                colomns = Channel names (str) in the original order.
                raws = frames (int)
                contains bool. True if the frame-channel is not acquired, and elapsed time df contains np.nan.

        Raises:
            ValueError: if file not read yet.
        """
        if not self.readflag:
            raise ValueError("file not yet read")
        return self.skipbool
    def get_channels(self):
        """get channel names

        Returns:
            np.ndarray(str): channel names in the original order

        Raises:
            ValueError: if file not read yet.
        """        
        if not self.readflag:
            raise ValueError("file not yet read")
        return self.channels

    def _readmain(self):
        self.metadata = self._readdict()
        self.eltimedf = self._extracteltime()
        self.skipbool = np.isnan(self.eltimedf)
        self.channels = self.eltimedf.columns.to_numpy()
        self._assertdata()
    def _readdict(self):
        with open(self.path, 'r') as f:
            txtdata=f.read()
        dictdata = json.loads(txtdata)
        return dictdata
    def _extracteltime(self):
        dlist = []
        for k,v in self.metadata.items():
            if "FrameIndex" in v and "ChannelIndex" in v and "ElapsedTime-ms" in v:
                l=[v["FrameIndex"],v["ChannelIndex"],v["ElapsedTime-ms"]]
                dlist.append(l)
            elif k=="Summary":
                channels = v["ChNames"]
            else:
                print("Eltime elements not found in Tree: " + k)
                return
        df = pd.DataFrame(dlist,columns=["FrameIndex", "ChannelIndex", "ElapsedTime-ms"])
        df = df.pivot(columns="ChannelIndex",index="FrameIndex",values="ElapsedTime-ms")
        df.columns = channels
        return df
    def _assertdata(self):
        assert isinstance(self.eltimedf,pd.DataFrame)
        assert isinstance(self.skipbool,pd.DataFrame)
        assert isinstance(self.channels,np.ndarray)






class TrackMateOutput(DataReader):
    '''
    TrackMate output .xml file.
    Returns data as DataFrame or other simple types.
    This class may depend on TrackMate versions.
    So far this class does not read and compile every information in the xml file.
    More methods may be added in the future if necessary.
    '''
    def get_tree(self) -> ET.ElementTree:
        """Get whole tree as ElementTree object.

        Returns:
            ET.ElementTree: whole xml tree object.
        """        
        return self.tree

    def get_spots(self):
        """get spot information as dataframe

        Returns:
            pd.DataFrame: 
                coloumns: Spot features specified in the xml file.
                rows: arbitrary index.
                'id' column is spot ID as integer.

        Raises:
            ValueError: if file not read yet.
        """        
        if not self.readflag:
            raise ValueError("file not yet read")
        return self.spotdf
    def get_tracks(self):
        """get track information as dataframe

        Returns:
            pd.DataFrame: 
                coloumns: Track features specified in the xml file.
                rows: arbitrary index.
                'track_id' column is track ID as integer.
                Spots can be specified using edges data. 
                Note that single spots are not included as tracks nor edges

        Raises:
            ValueError: if file not read yet.
            ValueError: if file not containing track data.
        """        
        if not self.readflag:
            raise ValueError("file not yet read")
        if self.trackdf is None:
            raise ValueError('No track data was read from the xml file')
        return self.trackdf
    def get_edges(self):
        """get edge information as dataframe

        Returns:
            pd.DataFrame: 
                coloumns: Edge features specified in the xml file.
                rows: arbitrary index.
                'track_id' column is track ID, added for clarity.
                'spot_source_id' and 'spot_target_id' can be used for specifying.
                Note that single spots are not included as tracks nor edges

        Raises:
            ValueError: if file not read yet.
            ValueError: if file not containing edge data.
        """        
        if not self.readflag:
            raise ValueError("file not yet read")
        if self.edgedf is None:
            raise ValueError('No edge data was read from the xml file')
        return self.edgedf

    def get_nspots(self):
        """get total number of spots

        Returns:
            int: the number of spots in total

        Raises:
            ValueError: if file not read yet.
        """        
        if not self.readflag:
            raise ValueError("file not yet read")
        return self.nspots

    def get_spots_with_track_id(self):
        """Get the spots DF containing track id. Singletons are separated.

        Raises:
            ValueError: if no edge data was read.

        Returns:
            pd.DataFrame: Singletons spot dataframe. Just screened spot df.
            pd.DataFrame: Spots with track_id assigned. New column 'track_id' are appended.
        """        
        if self.edgedf is None:
            raise ValueError('No edge data was read from the xml file')

        # singletons have -1 track_id at a time
        new_spotdf = copy.deepcopy(self.spotdf)
        new_spotdf['track_id'] = -1
        id_series = new_spotdf['id']

        source_df = self.edgedf[['spot_source_id','track_id']].rename(columns={'spot_source_id':'id'})
        target_df = self.edgedf[['spot_target_id','track_id']].rename(columns={'spot_target_id':'id'})

        source_nanfill = pd.merge(id_series,source_df,'left',on='id')
        target_nanfill = pd.merge(id_series,target_df,'left',on='id')

        new_spotdf.update(source_nanfill['track_id'])
        new_spotdf.update(target_nanfill['track_id'])
        new_spotdf['track_id'] = new_spotdf['track_id'].astype(int)
        new_spotdf = new_spotdf.sort_values(by='track_id').reset_index(drop=True)

        singletons = new_spotdf.loc[new_spotdf['track_id']==-1,:]
        singletons = singletons.drop('track_id',axis=1)
        with_track = new_spotdf.loc[new_spotdf['track_id']!=-1,:]

        return singletons,with_track

    def _readmain(self):
        self.tree = ET.parse(self.path)
        self.root = self.tree.getroot()
        self.tm_version = self.root.attrib['version'] # str
        self._check_version()
        self.log = self.root.find('Log').text
        self.settings = self.root.find('Settings')
        self.settings_imagedata = self.settings.find('ImageData').attrib # dict
        self.settings_detector = self.settings.find('DetectorSettings').attrib # dict
        self.settings_tracker = self.settings.find('TrackerSettings').attrib # dict

        Model = self.root.find('Model')
        AllSpots = Model.find('AllSpots')
        self.nspots = int(AllSpots.attrib['nspots'])
        FeatureDeclarations = Model.find('FeatureDeclarations')
        AllTracks = Model.find('AllTracks')
        self._features(FeatureDeclarations)
        self.spotdf = self._getspots(AllSpots)
        self.trackdf, self.edgedf = self._gettracks(AllTracks)
    
    def _check_version(self):
        if self.tm_version == '6.0.1':
            return
        else:
            print(f'TrackMate version {self.tm_version} is not tested')
            return
    
    def _features(self,fd):
        self.sf = fd.find('SpotFeatures')
        self.nsf = len(self.sf)
        self.sflist = []
        self.sfisint = []
        for f in self.sf:
            self.sflist.append(f.get('feature').lower())
            isint = f.get('isint')
            if isint=='true':
                self.sfisint.append(True)
            else:
                self.sfisint.append(False)

        self.tf = fd.find('TrackFeatures')
        self.ntf = len(self.sf)
        self.tflist = []
        self.tfisint = []
        for f in self.tf:
            self.tflist.append(f.get('feature').lower())
            isint = f.get('isint')
            if isint=='true':
                self.tfisint.append(True)
            else:
                self.tfisint.append(False)

        self.ef = fd.find('EdgeFeatures')
        self.nef = len(self.ef)
        self.eflist = []
        self.efisint = []
        for f in self.ef:
            self.eflist.append(f.get('feature').lower())
            isint = f.get('isint')
            if isint=='true':
                self.efisint.append(True)
            else:
                self.efisint.append(False)

    def _getspots(self,AS):
        darray = np.zeros(self.nspots,dtype=object)
        for i,spot in enumerate(AS.iter('Spot')):
            darray[i] = spot.attrib
        df = pd.DataFrame(list(darray))
        df.columns = df.columns.str.lower()

        df['id'] = df['id'].astype(int)
        float_col = [n for i,n in enumerate(self.sflist) if not self.sfisint[i]]
        int_col = [n for i,n in enumerate(self.sflist) if self.sfisint[i]]
        float_col = [n for n in float_col if n in df.columns]
        int_col = [n for n in int_col if n in df.columns]
        df[float_col] = df[float_col].astype(float)
        df[int_col] = df[int_col].astype(int)
        return df
    
    def _gettracks(self,AT):
        trackarr = np.zeros(len(AT),dtype=object)
        edgelist = []
        for i,track in enumerate(AT.findall('Track')):
            trackarr[i] = track.attrib
            for j,edge in enumerate(track.findall('Edge')):
                d = copy.deepcopy(edge.attrib)
                d.update({'track_id':track.get('TRACK_ID')})
                edgelist.append(d)
        if len(trackarr)==0:
            return None,None
        
        tdf = pd.DataFrame(list(trackarr))
        tdf.columns = tdf.columns.str.lower()
        edf = pd.DataFrame(edgelist)
        edf.columns = edf.columns.str.lower()

        tdf['track_id'] = tdf['track_id'].astype(int)
        edf['track_id'] = edf['track_id'].astype(int)
        float_col = [n for i,n in enumerate(self.tflist) if not self.tfisint[i]]
        int_col = [n for i,n in enumerate(self.tflist) if self.tfisint[i]]
        float_col = [n for n in float_col if n in tdf.columns]
        int_col = [n for n in int_col if n in tdf.columns]
        tdf[float_col] = tdf[float_col].astype(float)
        tdf[int_col] = tdf[int_col].astype(int)

        float_col = [n for i,n in enumerate(self.eflist) if not self.efisint[i]]
        int_col = [n for i,n in enumerate(self.eflist) if self.efisint[i]]
        float_col = [n for n in float_col if n in edf.columns]
        int_col = [n for n in int_col if n in edf.columns]
        edf[float_col] = edf[float_col].astype(float)
        edf[int_col] = edf[int_col].astype(int)
        return tdf, edf
    


def TrackMateOutputFactory(path,read=False,force=False) -> TrackMateOutput:
    '''This is a factory function for generating TrackMateOutput class for different TrackMate versions.
    Currently only 6.0.1 is supported.
    If unsupported version is given, this raises ValueError.
    If force is True, this returns ver6.0.1 reader even for different versions.'''
    if isinstance(path,PurePath):
        path = str(path)
    elif isinstance(path,str):
        path = path
    else:
        raise TypeError('invalid path type, neither Path or str')
    tree = ET.parse(path)
    root = tree.getroot()
    tm_version = root.attrib['version'] # str
    if tm_version == '6.0.1':
        return TrackMateOutput(path,read)
    else:
        if force:
            return TrackMateOutput(path,read)
        raise ValueError(f'TrackMate version {tm_version} is not tested. \n If you force to get ver6.0.1 reader, call this function with force=True.')



# UtilityClass

class Stream(object):
    def __init__(self,ri:RawImage,md:MetaData,read=False):
        """Utility class to read RawImage and MetaData in combination.

        Args:
            ri (RawImage): RawImage object that already finished read(), 
                or will finish read() separately.
            md (MetaData): MetaData object. Same as ri.

        """ 
        self.ri = ri
        self.md = md
        if read:
            self.read()
    def read(self):
        '''read both RawImage and MetaData'''
        self.ri.read()
        self.md.read()
    def get_im_channel(self,channel,skip=False):
        """get image with specified channel name(s)

        Args:
            channel (str|iterable[str]): channel(s) to get

        Returns:
            np.ndarray[np.uint16]: TCYX or TYX
                The reurned ndarray has full frames.
                It will contain duplicates if any frames are skipped.
        """        
        if len(self.get_channels())==1 and self.get_image().ndim==3:
            print('single channel file')
            if channel not in self.get_channels():
                print('channel not found')
                return
            return self.get_image()
        if not self._checkaxes():
            print('axes configuration not consistent in Stream')
            return
        channels = self.get_channels()
        if isinstance(channel,str):
            channel=[channel]
        index=[]
        for ch in channel:
            if ch in channels:
                index.append(list(channels).index(ch))
                continue
            text='channel {ch} does not exist in Stream.'.format(ch=channel)
            print(text)
            return
        image=self.get_image()
        selected = np.squeeze(image[:,index,:,:])
        if skip:
            skb = self.get_skipbool()[ch]
            selected = selected[~skb,:,:]

        return selected

    def get_channels(self):
        """call MetaData.get_channels()

        Returns:
            np.ndarray[str]: channels
        """        
        return self.md.get_channels()
    def get_image(self):
        """call RawImage.get_channels()

        Returns:
            np.ndarray[np.uint16]: TCYX
        """        
        return self.ri.get_im()
    def get_eltime(self):
        """call MetaData.get_eltime()

        Returns:
            pd.DataFrame: elapsed time (ms)
        """        
        return self.md.get_eltime()
    def get_skipbool(self):
        """call MetaData.get_skipbool().

        Returns:
            pd.DataFrame: skipped frame/channel
        """        
        return self.md.get_skipbool()
    def _checkaxes(self):
        """check if frames and channel length are the same between image and metadata.

        Returns:
            bool: True if okay
        """        
        imshape = self.get_image().shape
        metashape = self.get_eltime().shape
        return imshape[0]==metashape[0] and imshape[1]==metashape[1]










# Utility tools for path handling.


from pathlib import Path

class StreamPath:
    def __init__(self,root:str,imglobkey=None,metaglobkey=None):
        self.p = Path(root)
        self.search(imglobkey,metaglobkey)
    def search(self,imglobkey=None,metaglobkey=None):
        if imglobkey is None:
            self.rawim = sorted(self.p.glob('*.ome.tif'))[0]
        else:
            self.rawim = sorted(self.p.glob(imglobkey))[0]
        if metaglobkey is None:
            self.meta = sorted(self.p.glob('*metadata.txt'))[0]
        else:
            self.meta = sorted(self.p.glob(metaglobkey))[0]
        self.processed = self.p/'processed'
    def generate_stream(self,read=False):
        '''returns Stream object'''
        ri = RawImage(self.rawim)
        md = MetaData(self.meta)
        st = Stream(ri,md,read)
        return st

class ChamberPath():
    def __init__(self,root:str):
        self.p = Path(root)
        self.streams = []
        self.search()
    def search(self):
        for ch in self.p.iterdir():
            if not ch.is_dir():
                continue
            self.streams.append(StreamPath(str(ch)))
    def generate_streampath(self) -> list:
        '''returns list of StreamPath object'''
        return self.streams
        
            
