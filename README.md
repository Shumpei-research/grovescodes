# Groves Lab data analysis package

Last updated: 2022/02/03  
This is packaged modules for data analysis.  
Currently i/o for scope8 and TrackMate is the main purpose.  
by Shumpei Morita

## Package Content

package name:  
groves  

modules:  
data  
result  


## How to use

Make and activate your virtual environment for safety.  
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


If you wanna load the scope8 