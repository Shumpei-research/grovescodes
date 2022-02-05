# Groves Lab data analysis package

Last updated: 2022/02/03  
This is packaged modules for data analysis.  
Currently i/o for scope8 and TrackMate is the main purpose.  
by Shumpei Morita

## Package Content

Package name:  
- groves  

Modules:  
- data  
- result  

## Tested environment
The codes are written for cross-platform, but tested only in Windows 10.  
python 3.7

## Dependency
python 3.7 or later  
numpy  
pandas  
tifffile  

## How to use

Make and activate your virtual environment for safety.  
Install necessary libraries.  

At terminal, navigate to the package folder (so that you see setup.py), run the following
```
pip install .
```
Now you installed the package.


In the python script, import the modules by:
```python
from groves import data, result
```
Now you imported the modules. ('data' and 'result' in the global name space)

### data module, scope 8 output

If you wanna load the scope8 outputs
```
./image.ome.tif
./image_metadata.txt  
```
Do like this:

```python
image_reader = data.RawImage('image.ome.tif')
meta_reader = data.MetaData('image_metadata.txt')

image_reader.read()
image = image_reader.get_im()
# image contains image data as np.ndarray[np.uint16], TCYX

meta_reader.read()
channels = meta_reader.get_channels()
# channels contains channel names as np.ndarray[str]
elapsed_time = meta_reader.get_eltime()
# elapsed_time contains elapsed time at each acquisition as pd.DataFrame

ricm_ix = np.nonzero(elapsed_time=='RICM')
ricm_image = image[:,ricm_ix,:,:]
# RICM channel images, TYX
```

Here, you are instanciaing RawImage class and MetaData class with the file path.  
These instances reads the file at the file path when .read() is called.  
The data is stored in a field of them.
You can obtain the read data by calling several 'getter' methods.
By calling RawImage.get_im() method, you can get the whole image file as ndarray in TCYX order.
By calling MetaData.get_channels(), you can get a ndarray of str containing channel names, such as ndarray(['RICM','tirf488','epi561'])  
You can select a specific channel using these loaded data.
'RICM' channel is extracted in this example.
Several getter methods are available. You can check them by reading the source code.  


There is a utility class to combine RawImage and MetaData, because they are always together.
```python
image_reader = data.RawImage('image.ome.tif')
meta_reader = data.MetaData('image_metadata.txt')

stream = data.Stream(image_reader,meta_reader)
stream.read() # internally calling image_reader.read() and meta_reader.read()
ricm_image = stream.get_im_channel('RICM')
# ricm_image contains images at 'RICM' channel as ndarray[np.uint16], TYX order

elapsed_time = stream.get_eltime()
channels = stream.get_channels()
# Stream also inherits getter methods from RawImage and MetaData
```

Here, you are combining RawImage and MetaData objects into a Stream object, 
which deals with both data types.  
Stream.read() method reads both data.  
Maybe the most convenient method is Stream.get_im_channel(channel) to get a specific channel images.  
You can see the image selection of 'RICM' channel is shortened in one line,
where all the steps are internally performed.  
Stream class inherits the other getter methods from RawImage and MetaData,
so you only need this Stream class to for most of your demands.

There's an additional layer of utility class for the path handling.  
Because scope8 automatically saves image file and metadata in a consistent naming rule,
you can just specify the folder name and perform automatic path search.
```python
stream_path = data.StreamPath('.')
stream = stream_path.generate_stream()
stream.read()
ricm_image = stream.get_im_channel('RICM')
```

StreamPath object automatically finds image path and meta path 
by internally calling glob('*.ome.tif') and glob('*metadata.txt').  
If you wanna specify other keys for glob(), you can add arguments like this:
```python
data.StreamPath('.',imglobkey='*main.ome.tif',metaglobkey='*main_metadata.ome.tif')
```

You have to explicitly call .read() method and you might feel lazy.  
This is intentinal, because reading large files takes time and memory,
and you might want to pend .read() in some situations.  
However, you can add 'read' argument to make them call read() upon instanciation.
```python
image_reader = data.RawImage('image.ome.tif',read=True)
image = image_reader.get_im()
```
or
```python
image_reader = data.RawImage('image.ome.tif')
meta_reader = data.MetaData('image_metadata.txt')

stream = data.Stream(image_reader,meta_reader,read=True)
ricm_image = stream.get_im_channel('RICM')
```
or
```python
stream_path = data.StreamPath('.')
stream = stream_path.generate_stream(read=True)
ricm_image = stream.get_im_channel('RICM')
```

### data module, TrackMate output

If you want to read TrackMate output,
```
./tracks.xml
```
you can do this.

```python
tm_reader = TrackMateOutputFactory('tracks.xml')
# TrackMateOutput object

tm_reader.read()
spots = tm_reader.get_spots()
# spots data as pd.DataFrame
singleton_spots, spots_with_tracks = tm_reader.get_spots_with_track_id()
# spots data with trackID at 'track_id' column. 
# spots without tracks, singletons, are compiled in another df without 'track_id' column.
```

Here, TrackMateOutputFactory function returns TrackMateOutput object.  
TrackMate has several versions, and its output format are not unified among them.  
TrackMateOutputFactory function is designed to instanciate different reader class for different TrackMate versions,
so that you don't have to care which version you are using.  
However, currently only the version 6.0.1 is supported.  

There are some other getter methods you can check at source code.


### result module

This result module is designed to help you save and load the analysis results in several data types.  
These save/load operations can be easily done with many available libraries, so this module is pretty optional.  

The classes provided inherit ReaderWriter base class, and implement four essential methods.  
```
read()
write()
get()
set()
```

For example, if you wanna save the binary image data as ndarray[bool],
at the path './binary_segment.tif'
```python
image = np.ones((100,100),dtype=bool)

rw = result.ImBinary('binary_segment.tif')
rw.set(image)
rw.write()
```
Here, you are instanciating the ImBinary class, setting the image data to its field,
and saving the data by calling write()

If you are lazy to call set() and write() separately,
.setwrite() utility method is available
```python
image = np.ones((100,100),dtype=bool)
rw = result.ImBinary('binary_segment.tif')
rw.setwrite(image)
```

After saving the data, you sometimes wanna load that data in the following analyses.  
You can load the saved file using the same class,
```python
rw = result.ImBinary('binary_segment.tif')
rw.read()
image = rw.get()
# ndarray[np.bool]
```
or
```python
rw = result.ImBinary('binary_segment.tif')
image = rw.readget()
# ndarray[np.bool]
```

One benefit using these ReaderWriter subclasses is you can be consistent about the types.  
If you save and load binary ndarray with normal libraries, you probably get ndarray[np.uint8].  
You sometimes need to convert it to bool by yourself.  
It is internally done with these classes.


There are several ReaderWriter subclasses available.

|Class    |Type                  |file extension|
|---      |---                   |---           |
|Dict     |dict                  |.json OR .txt|
|DFrame   |pd.DataFrame          |.csv|
|ImBinary |np.ndarray[np.bool]   |.tif|
|ImUint8  |np.ndarray[np.uint8]  |.tif|
|ImUint16 |np.ndarray[np.uint8]  |.tif|
|ImInt16  |np.ndarray[np.int16]  |.tif|
|ImInt32  |np.ndarray[np.int32]  |.tif|
|ImFloat32|np.ndarray[np.float32]|.tif|
|ImFloat64|np.ndarray[np.float64]|.tif|


If you want, you can constrain the 'reader' or 'writer' mode by adding an argument mode,
'r' or 'w' or 'rw':
```python
rw = result.ImBinary('binary_segment.tif',mode='r')
image = rw.readget()

newimage = np.zeros((100,100),dtype=bool)

# this raises PermissionError
rw.set(newimage)
# this raises PermissionError
rw.write()
# this raises PermissionError
rw.setwrite(newimage)
```