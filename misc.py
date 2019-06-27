from contextlib import contextmanager
import os
import pylab as pl
import numpy as np
import ffmpeg_wrap

from collections import defaultdict, OrderedDict, Callable
class OrderedDefaultDict(OrderedDict):
    # Source: http://stackoverflow.com/a/6190500/562769
    def __init__(self, default_factory=None, *a, **kw):
        if (default_factory is not None and
           not isinstance(default_factory, Callable)):
            raise TypeError('first argument must be callable')
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = self.default_factory,
        return type(self), args, None, None, self.items()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        import copy
        return type(self)(self.default_factory,
                          copy.deepcopy(self.items()))

    def __repr__(self):
        return 'OrderedDefaultDict(%s, %s)' % (self.default_factory,
                                               OrderedDict.__repr__(self))

from collections import deque
class frameCache:
    """Keep a cache of 100 frames (and no more).
    """
    def __init__(self, vidfile, size_xy, **kwargs):
        self.video = ffmpeg_wrap.FfmpegFrameReader(vidfile, size_xy, **kwargs)
        self.frame_cache = OrderedDefaultDict(lambda: np.zeros([size_xy[1], size_xy[0], 3], dtype='uint8'))
        self.frame_back = 100
        
        f =  self.video.get_next_frame()
        assert f is not None, "Video file %s has no first frame"%vidfile
        self.frame_cache[0] = f

            
    def get_frame(self, at_idx):
        at_idx = int(at_idx)
        
        #first, last
        kmin = next(iter(self.frame_cache))
        kmax = deque(iter(self.frame_cache), maxlen=1).pop()
                  
        for i in range(kmin, at_idx-self.frame_back):
            try: 
                del self.frame_cache[i]
            except KeyError: pass
        
        for i in range(kmax+1, at_idx+1):
            f = self.video.get_next_frame()
            if f is not None:
                self.frame_cache[i] = f
        
        return self.frame_cache[at_idx]

class dotdict(dict):    
    """A dot-able dictionary for easy access to items. Note stay clear from
    keys that clashes with dict internals like: copy, fromkeys,
    get, items, keys, pop, popitem, setdefault, update, and values.
    ...
    
    Examples
    --------
    >>> a = dotdict(val1=1)
    >>> a.val2 = 2
    >>> a
    {'val1': 1, 'val2': 2}
    >>> a['val1']
    1
    >>> a.val1
    1
    """
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


def add_file_posttext(filename, posttext):
    return os.path.splitext(filename)[0] + posttext + os.path.splitext(filename)[1]

def remove_file_posttext(filename, posttext):
    mainname, ext = os.path.splitext(filename)
    if mainname.lower().endswith(posttext):    
        return  mainname[:len(mainname)-len(posttext)]+ext
    else:
        raise ValueError('could not find %s in file %s'%(posttext, filename))

def dirfiles(dirname, fullpath=True):
    _, _, files = next(os.walk(dirname))
    files.sort()
    if fullpath:
        files = [os.path.join(dirname, file) for file in files]
    return files
        
@contextmanager
def allowTemporaryExistance(filepath):
    try: yield None
    finally: os.remove(filepath)

def hamming_2d(shape):
    i,j = shape
    a = np.r_[[np.hamming(i)]*j].T
    b = np.c_[[np.hamming(j)]*i]
    return a*b

def blob_pattern(shape):
    i,j = shape
    ii = int(i//2); jj = int(j//2)
    ham2d = hamming_2d([ii,jj])
    arr = np.zeros([i,j])
    arr[:ii,:jj] = ham2d
    arr[:ii,-jj:] = -ham2d
    arr[-ii:,:jj] = -ham2d
    arr[-ii:,-jj:] = ham2d
    return arr

def cmd_str(cmd):
    return '"'+('" "'.join(cmd))+'"'

def roundint(input):
    return np.round(input).astype('int')

def scaled_audio_write(fname, freq, xl, xr=None):
    from scipy.io import wavfile
    if xr is None:
        xr = xl
        
    scaled = np.r_[[xl], [xr]] * 1.0
    scaled = np.int16(np.round((scaled/np.max(scaled))*32767))
    
    wavfile.write(fname, freq, scaled.T)