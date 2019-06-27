from sys import stdin
from sys import stderr
import numpy as np
import subprocess

sp = subprocess
 
def YUVtoRGB(yuv):
    yuv = np.array(yuv)
    # according to ITU-R BT.709
    yuv[:,:, 0] = yuv[:,:, 0].clip(16, 235).astype(yuv.dtype) - 16
    yuv[:,:,1:] = yuv[:,:,1:].clip(16, 240).astype(yuv.dtype) - 128
     
    A = np.array([[1.164,  0.000,  1.793],
                  [1.164, -0.213, -0.533],
                  [1.164,  2.112,  0.000]])
     
    # our result
    rgb = np.dot(yuv, A.T).clip(0, 255).astype('uint8')
     
    return rgb

def to_string(input):
    return input.decode("utf-8") 

def get_ffmpeg_info(filename):
    cmd = ['ffprobe', '-show_streams', filename]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)#, close_fds=True)
    output = to_string(p.stdout.read())
    p.stdin.close()
    p.stdout.close()
    
    return output 

def get_shape(filename):
    probestr = get_ffmpeg_info(filename)
    shape =[int(probestr.split('height=')[1].split('\n')[0]),
            int(probestr.split('width=')[1].split('\n')[0])]
    return shape

def get_framerate(filename):
    info = get_ffmpeg_info(filename)
    # print info
    if 'tbr,' in info:
        framerate = info.split('tbr,')[0].split('\n')[-1].split(',')[-1].strip()
    elif 'fps,' in info:
        framerate = info.split('fps,')[0].split('\n')[-1].split(',')[-1].strip()
    else:
        return None
    return float(framerate)


def get_duration(filename):
    duration = get_ffmpeg_info(filename).split('Duration:')[1].split(',')[0]
    duration = np.sum([float(duration.split(':')[-(i + 1)]) * (60 ** i) for i in range(len(duration.split(':')))])
    return duration


def get_channels(filename):
    channels = int(get_ffmpeg_info(filename).split('channels=')[1].split('\n')[0].strip())
    return channels


def get_sample_rate(filename):
    channels = int(get_ffmpeg_info(filename).split('sample_rate=')[1].split('\n')[0].strip())
    return channels


def write_as_audio(filename, signals, sampefreq=44100, codecs='copy'):
    pipe = sp.Popen(['ffmpeg',
                     '-y',
                     '-f', 's16le',
                     # "-acodec", "pcm_s16le",
                     '-ar', '%d' % sampefreq,
                     '-ac', '%d' % len(signals),
                     '-i', '-',
                     '-strict', 'experimental',
                     '-vn',
                     '-c:a', codecs,
                     filename],
                    stdin=sp.PIPE, stdout=sp.PIPE)
    signal = (np.vstack(signals).T).flatten()
    signal.astype("int16").tofile(pipe.stdin)


def read_as_audio(filename):
    channels = get_channels(filename)
    samplefreq = get_sample_rate(filename)
    cmd = ['ffmpeg',
           '-i', filename,
           # '-strict', 'experimental',
           '-f', 's16le',
           '-acodec', 'pcm_s16le',
           '-ar', '%d' % samplefreq,
           '-ac', '%d' % channels,
           '-']
    print(' '.join(cmd))
    pipe = sp.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 ** 8)
    raw_audio = pipe.stdout.read()

    audio_channels = np.frombuffer(raw_audio, dtype="int16")
    audio_channels = audio_channels.reshape((int(len(audio_channels) / channels), channels))

    return audio_channels, samplefreq


import time
class FfmpegFrameReader:
    def __init__(self, filename, framesize_xy=None, as_RGB=False):
        self.as_RGB = as_RGB
        if framesize_xy is None:
            framesize_xy = get_shape(filename)[::-1]
            
        cmd = ['ffmpeg',
               '-y',
               '-i', '%s'%filename,
               '-f', 'image2pipe',
               '-s', '%dx%d'%(framesize_xy[0], framesize_xy[1]),
               '-pix_fmt', 'rgb24' if as_RGB else 'yuv444p',
               '-vcodec', 'rawvideo',
               '-']
        
        self.pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 ** 8)
        self.f = self.pipe.stdout
        
        self.frame_shape = [framesize_xy[1], framesize_xy[0]]
        self.frame_size = framesize_xy[0]*framesize_xy[1]
        self.frame_no = 0
    
    def get_next_frame(self):
        
        self.frame_no += 1
        frame_bytes = np.frombuffer(self.f.read(self.frame_size*3), dtype='uint8')
        
        if len(frame_bytes) == 0:
            self.close_file()
            return None

        if not self.as_RGB: frame_bytes = frame_bytes.reshape([3,-1]).T
            
        return frame_bytes.reshape([self.frame_shape[0], self.frame_shape[1], 3])

    def close_file(self):
        try:
            self.f
        except:
            pass
        else:
            if self.f != stdin and self.f is not None:
                if not self.f.closed:
                    while self.pipe.poll is None:
                        print('1',end='')
                        time.sleep(0.000001)
                    self.f.close()

    def __del__(self):
        self.close_file()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close_file()
        return False



class FfmpegFrameWriter:
    def __init__(self, filename, framesize_xy=None, fps=24):
        self.framesize_xy = framesize_xy
        self.fps = fps
        cmd = ['ffmpeg',
               '-y', # (optional) overwrite output file if it exists
               '-f', 'rawvideo',
               '-vcodec','rawvideo',
               '-s', '%dx%d'%(framesize_xy[0], framesize_xy[1]), # size of one frame
               '-pix_fmt', 'rgb24',
               '-r', '%d'%fps, # frames per second
               '-i', '-', # The input comes from a pipe
               '-an', # Tells FFMPEG not to expect any audio
               '-vcodec', 'mpeg4',
               filename]
        
        self.pipe = subprocess.Popen(cmd, stdin=sp.PIPE)
        self.f = self.pipe.stdin
    
    def write_next_frame(self, array):
        
        try:
            assert  array.dtype == np.dtype('uint8') 
        except:
            stderr.write("Error array must be of dtype 'uint8'.\n")
            raise
        
        #try:
        #    assert list(array.shape) == [self.framesize_xy[1], self.framesize_xy[0], [3]]
        #except: 
        #    stderr.write("Error array must be of shape [%d,%d,%d].\n"%(self.framesize_xy[0], self.framesize_xy[1], 3))
        #    raise

        self.f.write(array.tostring())
        
    def close_file(self):
        self.f.close()
        
    def __del__(self):
        self.close_file()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close_file()
        return False
    
#***************************************************************************************
# Some tests
#
#***************************************************************************************
import unittest
import os

examples_path = os.path.abspath(os.path.join(__file__, '..', 'examples'))
video_signal_path = os.path.join(examples_path, 'video_signal.mp4')
video_output_path = os.path.join(examples_path, 'video_output.mp4')
audio_signal_path = os.path.join(examples_path, 'audio_signal.mp3')


class TestFfmpegFrameReader(unittest.TestCase):
    def test_get_ffmpeg_info(self):
        self.assertTrue('Duration' in get_ffmpeg_info(video_signal_path))

    def test_get_duration(self):
        self.assertAlmostEqual(get_duration(video_signal_path), 8.33, 1)

    def test_get_framerate(self):
        self.assertEqual(get_framerate(video_signal_path), 24)

    def test_get_shape(self):
        self.assertEqual(get_shape(video_signal_path), [100,100])
        
    def test_get_channels(self):
        self.assertEqual(get_channels(audio_signal_path), 2)

    def test_get_sample_rate(self):
        self.assertEqual(get_sample_rate(audio_signal_path), 44100)

    def test_read_as_audio(self):
        a, freq = read_as_audio(audio_signal_path)
        self.assertEqual(len(a), 88751)
        self.assertEqual(len(a[0]), 2)

    def test_FfmpegFrameReader(self):
        with FfmpegFrameReader(video_signal_path, framesize_xy=(4,4)) as r:
            f = r.get_next_frame()
            for i in range(3):
                f = r.get_next_frame()
                self.assertEqual(list(f.shape), [4,4,3])
                
    def test_FfmpegFrameWriter(self):
        with FfmpegFrameWriter(video_output_path, framesize_xy=(120,100)) as w:
            for i in range(200):
                w.write_next_frame((np.random.random([100,120,3])*128).astype('uint8'))

if __name__ == "__main__":
    unittest.main()
    
    '''
    tester = TestFfmpegFrameReader()
    method_list = [func for func in dir(tester) if callable(getattr(tester, func)) and func.startswith("test_")]
    import time
    for f in method_list:
        print('----------------------------------------')
        print('')
        print('Testing: '+f[5:])
        print('')
        print('----------------------------------------')
        exec('tester.'+f+'()')
        time.sleep(0.01)
    '''
    