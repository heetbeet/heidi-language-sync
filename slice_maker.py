import numpy as np

class ddict(dict):    
    """A dot-able dictionary for easy access to items. Note stay clear from
    keys that clashes with dict internals like: copy, fromkeys,
    get, items, keys, pop, popitem, setdefault, update, and values.
    ...
    
    Examples
    --------
    >>> a = ddict(val1=1)
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


def slice_maker(xlen,
                nwindows=None,
                wsize=None,
                **kwargs):
    """
    Generator function to slice an array into overlapping 
    parts.

    xlen: 
        The length of the array to be sliced

    nwindows (default 10) or wsize:
       (a) breakup into nwindows, or
       (b) a window size wsize (minus overlap).

    **kwargs:
        overlap or overlap_frac (default 0.1):
            (a) overlap, the overlapping window size, or
            (b) calculate overlap as overlap_frac*wsize.

        expand or expand_frac (default 0):
            (a) expand, the maximum wsize expansion (expands maximum this
                many samples randomly per each window), or
            (b) calculate expand as expand_frac*wsize

        endsize or endsize_frac (default 0.4):
            (a) endsize, the minimum allowed size for the last window, or
            (b) calculate endsize as endsize_frac*wsize


    Returns each iteration:
        (i,j): from-and-to indices and
        
        params: a dot dictionary with keys
            nwindows: number of windows expected
            i: index of current yield
            is_first: bool indicates first (i,j)
            is_last: bool indicates last (i,j)
            dampwin: scaling window overlap dampening (n np.array of size wsize)
            
    For example:
    
    0                                                                         xlen
    |______________                                                             |
    |    window    `-,                                                          |
    |    1 of 4    |  \  |                                                      |
    |______________|___`-,______________________________________________________|
    |<-  wsize 1 ->|     |                                                      |
    |<- totsize 1 ------>|_____________|                                        |
    |              |   ,-`   window    `-,                                      |
    |              |  /  |   2 of 4    |  \  |                                  |
    |              ,-`___|_____________|___`-,                                  |
    |              |     |<- wsize 2 ->|     |                                  |
    |              |<------ totsize 2 ------>|_____________|                    |
    |              |     |             |   ,-`   window    `-,                  |
    |              |     |             |  /  |   3 of 4    |  \  |              |
    |              |<over|             ,-`___|_____________|___`-,              |
    |              | lap>|             |     |<- wsize 3 ->|     |              |
    |              |     |             |<------ totsize 3 ------>|______________|
    |              |     |             |     |             |   ,-`   window     |
    |              |     |             |     |             |  /  |   4 of 4     |
    i(1) ------------> j(1)            |     |             ,-`___|______________|
    |              |                   |     |             |     |<- wsize 4  ->|
    |            i(2) -------------------> j(2)            |<------ totsize 4 ->|
    |                                  |                   |     |              |
    |                                i(3) -------------------> j(3)             |
    |                                                      |                    |
                                                         i(4) --------------> j(4)
                                                          
    Note, the wsizes will differ randomly if expand > 0, but the overlap is 
    kept constant.
    
    """
    kwargs = ddict(**kwargs)
    defaults = ddict(overlap_frac=0.1,
                     expand_frac=0,
                     endsize_frac=0.4)
    
    if nwindows is None and wsize is None:
        raise ValueError("Input either nwindows or wsize, not both.")
    elif wsize is None:
        if nwindows is None:
            nwindows = 10
        wsize = xlen/(nwindows)
    
    for key in defaults:
        if key[:-5] in kwargs and key in kwargs: 
            raise ValueError("Input either %s or %s, not both."%(key, key[:-5]))
        else:
            if key[:-5] not in kwargs and key not in kwargs:
                kwargs[key] = defaults[key]
            if key[:-5] not in kwargs:
                kwargs[key[:-5]] = kwargs[key]*wsize
    
    #Do 15 times and keep healthiest last window size - closest to wsize
    #TODO: replace this tactic with a cleaver np.linspace implimentation
    points_maybes = []
    for i in range(15):
        points = [0]
        while points[-1] < xlen:
            point = points[-1] + wsize + np.random.rand()*kwargs.expand
            point = min(xlen, point)
            points.append(min(xlen, point))
            
        #Fix if last window size will be smaller than minsize
        if len(points) > 2:
            if points[-1] - points[-2] < kwargs.endsize:
                points = points[:-2]+points[-1:] #remove second last point
                                
        points_maybes.append(points)
        
        if kwargs.expand ==0:
            break

    idx = np.argmin([np.abs(wsize - (points[-1]-points[-2])) for points in points_maybes])
    points = points_maybes[idx]
      
    
    #  1 |          ___|  (Overlap damping
    #    |       ,-`   |   cosine curve)
    #    |      /      |
    #  0 |___,-`       |
    #    0 - - - - - overlap 
    kwargs.overlap = int(round(kwargs.overlap)) #integer
    curve = (np.cos(np.linspace(np.pi, 0, kwargs.overlap+2))*0.5+0.5)[1:-1]
    
    
    lovlap = np.floor(kwargs.overlap/2) # loverlap + roverlap = overlap
    rovlap = np.ceil(kwargs.overlap/2)
    
    p = ddict(nwindows=len(points)-1)
    for ii, (i,j) in enumerate(zip(points[:-1],points[1:])):
        p = ddict(**p)
        p.i = ii
        p.is_first = ii==0
        p.is_last  = ii==len(points)-2
        
        i = i if p.is_first else int(i-lovlap)
        j = j if p.is_last  else int(j+rovlap)
        
        p.dampwin = np.ones(j-i).astype('float')
        if kwargs.overlap > 0:
            if not p.is_first: p.dampwin[:kwargs.overlap]  *= curve
            if not p.is_last : p.dampwin[-kwargs.overlap:] *= curve[::-1]
            
        yield (i,j), p