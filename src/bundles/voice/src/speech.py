# vim: set expandtab shiftwidth=4 softtabstop=4:

#  === UCSF ChimeraX Copyright ===
#  Copyright 2022 Regents of the University of California.
#  All rights reserved.  This software provided pursuant to a
#  license agreement containing restrictions on its disclosure,
#  duplication and use.  For details see:
#  https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
#  This notice must be embedded in or attached to all copies,
#  including partial copies, of the software or any revisions
#  or derivations thereof.
#  === UCSF ChimeraX Copyright ===
import atexit
import os
import threading
import tempfile
import warnings
import wave

# We don't really care that Numba has a problem with how whisper is using it.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import whisper
import pyaudio

import numpy as np

TOKENS_TO_SUPPRESS = [
    13 # periods
]

CHUNK_SIZE = 1024
CHANNELS = 2
RATE = 44100

# Adapted from code from Vispy's example directory. While Vispy is (at this time, July 2023) 
# licensed under the BSD license, their license states that all code in their example directory
# is public domain. Thanks, Vispy!
# See https://github.com/vispy/vispy/blob/main/examples/demo/scene/oscilloscope.py
class SpeechRecorder:
    def __init__(self, rate=RATE, chunksize=CHUNK_SIZE):
        self.rate = rate
        self.chunksize = chunksize
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.chunksize,
                                  stream_callback=self.new_frame)
        self.lock = threading.Lock()
        self.stop = False
        self.frames = []
        atexit.register(self.close)

    def new_frame(self, data, frame_count, time_info, status):
        data = np.fromstring(data, 'int16')
        with self.lock:
            self.frames.append(data)
            if self.stop:
                return None, pyaudio.paComplete
        return None, pyaudio.paContinue

    def get_frames(self):
        with self.lock:
            frames = self.frames
            self.frames = []
            return frames

    def record(self):
        self.stream.start_stream()

    def close(self):
        with self.lock:
            self.stop = True
        self.stream.close()
        self.p.terminate()

class SpeechDecoder:
    def __init__(self, frames) -> None:
        self.frames = frames 

    def decode(self):
        model = whisper.load_model("base.en")
        f = tempfile.NamedTemporaryFile(suffix=".wav")
        with wave.open(f, "wb") as wav:
            wav.setnchannels(CHANNELS)
            wav.setframerate(RATE)
            wav.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
            wav.setnframes(len(self.frames))
            for frame in self.frames:
                wav.writeframesraw(frame.tobytes())
        # We have no way of knowing whether a CPU supports F16 or F32, and since 
        # whisper seems to have no problem picking on its own, it doesn't really 
        # need to bother our users by telling them now does it
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.transcribe(f.name, suppress_tokens=TOKENS_TO_SUPPRESS)
        del f
        return SpeechResult(result)
    
class SpeechResult:
    def __init__(self, result):
        self.text = result['text']
        self.segments = result['segments']

    def getText(self):
        return self.text.strip().strip(".").lower()
    
    def getSegments(self):
        return self.segments
    
    
if __name__ == "__main__":
    decoder = SpeechDecoder("src/bundles/voice/test/data/open 5y5s.m4a")
    result = decoder.decode()
    print(result.getText()) 
    print(result.getSegments()) 