import os
import sys
import subprocess

import fft
import ffmpeg_wrap

import pylab as plt
import numpy as np
from scipy import ndimage
from scipy import signal
from scipy import interpolate
import misc

from slice_maker import slice_maker

#Generate some filenames to be used throughout the program
def generate_filenames(discard_video,     #eg. Afrikaans LQ
                       desired_video,     #eg. Japanese HQ
                       desired_audio,     #eg. Afrikaans language
                       output='./output', #output dirname
                       id='',             #eg. file number to prepend (0000, 0001...)
                      ):
    if isinstance(id, int):
        id = "%03d"%id
    else:
        id = str(id)
    
    fnames = misc.dotdict()
    fnames.id = id
    fnames.discard_video = discard_video
    fnames.desired_video = desired_video
    fnames.desired_audio = desired_audio
        
    discard_vname = os.path.splitext(os.path.basename(discard_video))[0]
    desired_vname = os.path.splitext(os.path.basename(desired_video))[0]
    desired_aname = os.path.splitext(os.path.basename(desired_audio))[0]
    
    def tdir(dname):
        outdir = os.path.join(output, dname)
        os.makedirs(outdir, exist_ok=True)
        return outdir

    fnames.tmp = tdir('temp')

    outname = id+'--'+desired_vname+'--'+desired_aname
    
    fnames.output_video = os.path.join(tdir('video'), outname+'.mp4')
    fnames.output_wav   = os.path.join(tdir('audio'),  outname+'.wav')
    fnames.output_aac   = os.path.join(tdir('audio'),  outname+'-aac.mp4')
    fnames.input_audio  = os.path.join(tdir('audio'),  id+'-input-audio-'+discard_vname+'.npy')
    
    fnames.desired_signal = os.path.join(tdir('data'), id+'-desiredsig--'+desired_vname+'.npy')
    fnames.discard_signal = os.path.join(tdir('data'), id+'-discardsig--'+discard_vname+'.npy')
    
    fnames.desired_frames = os.path.join(tdir('data'), id+'-desiredframes--'+desired_vname+'.npy')
    fnames.discard_frames = os.path.join(tdir('data'), id+'-discardframes--'+discard_vname+'.npy')
    
    fnames.dmap_dumps = tdir('data/dmaps/'+outname)
    fnames.dmap = os.path.join(tdir('data'), outname+'-dmap.npy')
    fnames.dmap_smooth = os.path.join(tdir('data'), outname+'-dmap-smooth.npy')
    
    return fnames
    

def read_2D_video_signal(filename):
    '''Read both videos and output then as a 1D signal'''
    v = [];
    frames = [] 
    blob = misc.blob_pattern((80, 80))
    with ffmpeg_wrap.FfmpegFrameReader(filename, (80, 80)) as r:
        f = r.get_next_frame()
        while f is not None:
            v.append(np.sum(blob*f[:,:,0]))
            frames.append(f)
            
            f = r.get_next_frame()

    #somewhat normalising the signal
    v = np.array(v)
    v = v - np.mean(v)
    v = v / np.percentile(v, 96)
    return v, frames

def roundint(input):
    return np.round(input).astype('int')

def blackmanned(input):
    return input*np.blackman(len(input))

    
def padtopow2(input):
    output = np.zeros(int(2**np.ceil(np.log2(len(input)))), dtype=input.dtype)
    output[:len(input)] = input
    return output


def find_mapping(v1, v2, vmap=None, bordercontrol=(0, 0), show_result=False):
    #Apply a high-pass filter in order for the signals to be more "correlatable"
    #plot these to see the difference
    fs = 25
    v1 = (fft.butter_bandpass_filter(v1, 2, 7, fs, order=3))*1.0
    v2 = (fft.butter_bandpass_filter(v2, 2, 7, fs, order=3))*1.0
    
    #get border params
    lii = bordercontrol[0]           #start index (include)
    rii = len(v1)-bordercontrol[1]   #end index (exclude)
    uselen = rii-lii  #len of signal

    #Stretch signal 2 in order for sampling from mapping to be applied with a degree of subsampling
    upx = 100
    lpow = roundint(2**np.ceil(np.log2(len(v2))))
    v2_ups = signal.resample(np.r_[v2,np.zeros(lpow-len(v2))], lpow*upx)*1.0

    #                       
    #(1) dmap is the graph: 
    #              |_,-----,__
    # delay(v2:v1) |__________  
    #               i index of v1
    # so that v2[i+delay] ~= v1[i]
    #
    #(2) vmap is the graph:
    #           |    /    
    # index(v2) |  /
    #           |/________
    #            index(v1)
    # so that v2[vmap] ~= v1   
    #
    # eg. if vmap = [0,2,3,4,5,6,7], then
    #
    #           vmap:
    #         _ _ _ _ _ _
    #       7|_|_|_|_| |x|              dmap:
    #       6|_|_|_|_|x|_|           _ _ _ _ _ _
    #       5|_|_|_|x|_|_|         3|_|_|_|_|_|_|                   
    # index 4|_|_|x|_|_|_|   delay 2|_|_|x|x|x|x|
    #  (v2) 3|_|_|_|_|_|_|    (v2) 1|_|x|_|_|_|_|
    #       2|_|x|_|_|_|_|         0|x|_|_|_|_|_|
    #       1|_|_|_|_|_|_|           0 1 2 3 4 5
    #       0|x|_|_|_|_|_|            index(v1)
    #         0 1 2 3 4 5 
    #          index(v1)
    #
    # v1[0,1,2,3,4,5] ~= v2[vmap] = v2[0,2,3,4,5,6,7]
    
    dmap = np.zeros(len(v1))*1.0
    def to_vmap(dmap):
        return dmap+np.arange(len(dmap))

    #Cut signal into pieces and find each piece's phase shifts,
    dmap_old = dmap
    for nwindows in np.r_[1, 3, 4, np.linspace(0, 20, 25) + np.random.rand(25)*5]:
        
        # Overlapping windows: overlap by roughly half the window size
        # [win 1 ]
        #     [win 2 ]
        #         [win 3 ] ...
        for (i0, i1), p in slice_maker(uselen, #total length 
                                       nwindows=nwindows, #into how many windows?
                                       overlap_frac=0.5,  #overlap with neighbour
                                       expand_frac=0.15): #windows may enlarge by 15%
            i0, i1 = (i0+lii, i1+lii)
            
            sub1 = blackmanned(v1[i0:i1])
            try:
                sub2 = blackmanned(v2_ups[roundint(to_vmap(dmap_old)[i0:i1]*upx)])
            except:
                print('Oops out of bounds...')
                continue
            
            delay, _ = fft.phasedelay(sub1, sub2, subsample_rate=50)
            
            #only allow the delay if shifted 5% of snippet or less
            if np.abs(delay) > (i1-i0)*0.05:
                continue

            dmap[i0:i1] += (delay)*p.dampwin

        dmap[:lii] = dmap[lii]   # drag into the lhs border
        dmap[rii:] = dmap[rii-1] # drag into the rhs border
        
        #revert to old dmap if this new dmap is worse
        if (np.max(fft.xcorr(v1, v2_ups[roundint(to_vmap(dmap_old)[lii: rii]*upx)])) > 
            np.max(fft.xcorr(v1, v2_ups[roundint(to_vmap(dmap    )[lii: rii]*upx)]))):
            dmap = dmap_old

    return(dmap)

def fix_dmap_warps(dmap,
                   fnames=None,
                   speed_threshold = 0.4,#
                   fix_length = 60*24*2  #1 minutes
                  ):
    from scipy.ndimage.filters import gaussian_filter
    
    ################################################################33
    # extra stuff to take audio signal into account at cut points
    if fnames is not None:
        #info from the archives
        xl, xr = np.load(fnames.input_audio).T
        freq = ffmpeg_wrap.get_sample_rate(fnames.discard_video)
        fps = ffmpeg_wrap.get_framerate(fnames.discard_video)
        
        def find_audio_silent_spot(framei, framej):
            a0, a1 = (int((i/fps)*freq) for i in [framei, framej])
            
            #downsample 200x, and smooth out
            sound = gaussian_filter(gaussian_filter(np.abs((xl+xr)[a0: a1]), 4)[::200], 15)
            sps = freq/200 #sound per second

            silence = 1/sound
            silence = silence*signal.gaussian(len(silence), len(silence)/10)

            mid_soundsnip = np.argmax(silence)
            mid = framei + roundint((mid_soundsnip/sps)*fps)
            
            frame = np.sum(np.load(fnames.discard_frames)[mid], axis=2)
            
            misc.scaled_audio_write(fnames.tmp+'/'+fnames.id+'--%5d_to_%5d_l.wav'%(framei, framej),
                                   freq,
                                   xl[int(framei*freq/fps):int(mid*freq/fps)])
            misc.scaled_audio_write(fnames.tmp+'/'+fnames.id+'--%5d_to_%5d_r.wav'%(framei, framej),
                                   freq,
                                   xl[int(mid*freq/fps):int(framej*freq/fps)])


            plt.figure(figsize=(12,2))
            plt.plot()

            plt.xlabel(os.path.split(fnames.desired_video)[-1])
            
            return mid
    
        
    dmap = np.array(dmap)
    n = len(dmap)
    
    #Blur the crap out of the dmap to avoid jerky audio spikes
    dmap_blurred = gaussian_filter(dmap, sigma=1000)
    
    #derivative: slight blur to get smooth speed estimate
    speed = (gaussian_filter(dmap, sigma=100)[1:]-     
             gaussian_filter(dmap, sigma=100)[:-1])*24 #(??) don't know what units

    
    toofast = np.abs(speed)>0.4
   
    #Find the middle of speedbumps and "split" the signal there
    #There are better alternative ways, such as finding local maximas
    splits = []
    i=0
    while i < n-2:
        i+=1
        if toofast[i] and not toofast[i-1]:
            for j in range(i,len(toofast)-1):
                if toofast[j] and not toofast[j+1]:
                    m = i+np.argmax(np.abs(speed[i:j]))
                    if fnames is not None:
                        m = find_audio_silent_spot(i,j)
                        
                    splits.append((i,m,j))
                    
                    i=j-1 # Loop termination and 
                    break # progression

    #beginning and endpoints for these splits
    splits = [(None,0,1)] + splits + [(n-1,n,None)]
    
    plt.figure(figsize=(12,4))
    plt.plot(speed)
    for i in splits: plt.plot(list(i), [0]*len(i), 'o')
        
    print('splitpoints:', splits)
    #Break signal into chunks and smooth out these chunks
    chunks = []
    for (i0,m0,j0), (i1,m1,j1) in zip(splits[:-1], splits[1:]):
        chunk = dmap[m0:m1]
        idx = m0 #shift
        
        chunk[:j0-idx] = chunk[j0-idx]
        chunk[i1-idx:] = chunk[i1-idx]
        
        chunks.append(gaussian_filter(chunk, sigma=1200))
    
    dmap_fixed = np.concatenate(chunks)
    return dmap_fixed



def warp_audio(fnames):
    dmap = np.load(fnames.dmap_smooth)
    
    fps1 = ffmpeg_wrap.get_framerate(fnames.discard_video) #eg. afrikaans = v1
    fps2 = ffmpeg_wrap.get_framerate(fnames.desired_video) #eg. japanese  = v2
    
    vmap = misc.dotdict()
    vmap.idx1 = np.arange(len(dmap))
    vmap.idx2 = dmap + vmap.idx1
    vmap.sec1 = vmap.idx1/fps1
    vmap.sec2 = vmap.idx2/fps2

    #signal and timestamps for audio 1: we have this audio
    #arr, myfreq = ffmpeg_wrap.read_as_audio(fnames.desired_audio)
    
    x1l, x1r = np.load(fnames.input_audio).T
    myfreq = ffmpeg_wrap.get_sample_rate(fnames.desired_audio)
    
    aud_t1 = np.arange(len(x1l))/myfreq
        
    #timestamps for audio 2: we generate this audio
    runtime = ffmpeg_wrap.get_duration(fnames.desired_video) 
    aud_t2 = np.arange(runtime*myfreq)/myfreq  


    #This is from the scipy docs as examples:
    #f = interpolate.interp1d(x, y)
    #xnew = np.arange(0, 9, 0.1)
    #ynew = f(xnew)   # use interpolation function returned by `interp1d`
        
    aud_t1_new = interpolate.interp1d(vmap.sec2, #x = timestamps at v2
                                      vmap.sec1, #y = timestamps at v1
                                      fill_value=(0,0),
                                      bounds_error=False)(   aud_t2   )
    
    #div by sample delay -> indxes
    aud_idx1_new = aud_t1_new*myfreq
    aud_idx1_new[aud_idx1_new < 0] = 0
    aud_idx1_new[aud_idx1_new > len(x1l)-1] = len(x1l)-1

    #print('Resample left audio stream')
    #upsample to 5x and interpolate the rest of the way
    upx = 1 #too slow, skip this step
    
    # Too expensive to do at once :(, process as wth pieces    
    
    x2l, x2r = (np.zeros(len(aud_idx1_new), dtype='float'),
                np.zeros(len(aud_idx1_new), dtype='float'))
    
    
    print('Audio snippet', end='', flush=True)        
    for (ii, jj), p in slice_maker(len(aud_idx1_new),  #total length 
                                   wsize=48000*30   ,  #into 30 second snippets
                                   overlap_frac=0.05,  #overlap with neighbour
                                   endsize_frac=0.15): #endsize at least 15% wsize
        
        i1, j1 = (roundint(np.min(aud_idx1_new[ii:jj])),
                  roundint(np.max(aud_idx1_new[ii:jj])))
        
        powlen = j1-i1 #int(2**np.ceil(np.log2(j1-i1)))
        lsnip, rsnip = np.zeros(powlen, dtype='float'), np.zeros(powlen, dtype='float')
        lsnip[:j1-i1] = x1l[i1:j1]
        rsnip[:j1-i1] = x1r[i1:j1]
        
        if len(rsnip):
            print(('%d'%np.ceil((p.i+1)*100/p.nwindows)), end='% ', flush=True)

            lsnip, rsnip = [
                interpolate.interp1d(np.arange(len(snip))/upx  , # x: gotcha upx
                                     snip                      , # y: upsample upx times
                                     kind='quadratic'          ,
                                     fill_value=(0,0)          ,
                                     bounds_error=False)( aud_idx1_new[ii:jj]-i1 ) # newx: gotcha i1 -> 0
                for snip in (lsnip, rsnip)]

            for snip in lsnip, rsnip:
                snip*=p.dampwin

            #note that overlap windows sum to one :)
            x2l[ii:jj] += lsnip 
            x2r[ii:jj] += rsnip 
    
    print('Write the audio to '+os.path.abspath(fnames.output_wav), flush=True)
    misc.scaled_audio_write(fnames.output_wav, myfreq, x2l, x2r)

    
def mux_audio_to_video(fnames, lang='afr'):
    '''
    print("Converting wav to aac")
    cmd = ["ffmpeg",
           "-y",
           "-i", fnames.output_wav,
           "-codec:a", "aac",
           fnames.output_aac]
    
    print(misc.cmd_str(cmd))
    subprocess.call(cmd)
    '''
    
    #lang_orig = ["-metadata:s:a:0", "language=%s"%lang_orig] if lang_orig else []
    print("Muxing in afrikaans to "+os.path.abspath(fnames.output_video))
    cmd = ["ffmpeg",
           "-y",
           "-i", fnames.desired_video,
           "-i", fnames.output_wav,
           "-map", "0",
           "-map", "1:a",
           "-metadata:s:a:1", "language=%s"%lang,
           "-c:v", "copy",
           "-c:a", "aac",
           fnames.output_video]
    
    print(misc.cmd_str(cmd))
    subprocess.call(cmd)
    