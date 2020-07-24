from pathlib import Path
from pydub import AudioSegment
from pydub.silence import split_on_silence

import numpy as np

import os
import argparse

# *********************************************************************************************
#                                         first method
# *********************************************************************************************
def segment_method1(input_path):
    wav = AudioSegment.from_wav(input_path)
    # min_silence_len <ms>, silence_thresh <dB>
    chunks = split_on_silence (wav, min_silence_len = 800,silence_thresh = -62.5)
    i = 0
    for chunk in chunks:
        if len(chunk) > 800:
            print("Exporting chunk{0}.wav".format(i))
            chunk.export(".//sentence_{0}.wav".format(i),format = "wav")
            i += 1

# *********************************************************************************************

# *********************************************************************************************
#                                        second method 
# *********************************************************************************************
def windows(signal, window_size, step_size):
    if type(window_size) is not int:
        raise AttributeError("Window size must be an integer.")
    if type(step_size) is not int:
        raise AttributeError("Step size must be an integer.")
    for i_start in range(0, len(signal), step_size):
        i_end = i_start + window_size
        if i_end >= len(signal):
            break
        yield signal[i_start:i_end]

# find energy of sound using sum of square
def energy(samples):
    return np.sum(np.power(samples, 2.)) / float(len(samples))

# find index that is a rising edge then append to generator
def rising_edges(binary_signal):
    previous_value = 0
    index = 0
    for x in binary_signal:
        if x and not previous_value:
            yield index
        previous_value = x
        index += 1

# didn't use
# def remove_silence(data,sampwidth):
#     # Remove pauses using an energy threshold = 50% of the median energy:
#     segments = data / 2**((8*sampwidth)-1)
#     energies = [(s**2).sum() / len(s) for s in segments]
#     # (attention: integer overflow would occure without normalization here!)
#     thres = 0.5 * np.median(energies)
#     index_of_segments_to_keep = (np.where(energies > thres)[0])
#     # get segments that have energies higher than a the threshold:
#     no_silence_data = segments[index_of_segments_to_keep]
#     return no_silence_data

def end_silence_trim(data,rate,sampwidth):
    window_duration = 0.1
    step_duration = window_duration
    window_size = int(window_duration * rate)
    step_size = int(step_duration * rate)
    signal_windows = windows(
    signal=(data/2**((8*sampwidth)-1)),
    window_size=window_size,
    step_size=step_size
    )
    window_energy = [energy(w) for w in signal_windows]
    energies = np.array(window_energy)
    silence_thresh = 9e-5
    window_silence = [silence_thresh > e for e in window_energy]
    index = len(window_silence)-1
    while(window_silence[index] and index > 0):
        index -= 1
    stop = int(index * step_duration * rate) + (2*window_size)
    return data[:stop]

def start_silence_trim(data,rate,sampwidth):
    window_duration = 0.1
    step_duration = window_duration
    window_size = int(window_duration * rate)
    step_size = int(step_duration * rate)
    signal_windows = windows(
    signal=(data/2**((8*sampwidth)-1)),
    window_size=window_size,
    step_size=step_size
    )
    window_energy = [energy(w) for w in signal_windows]
    energies = np.array(window_energy)
    silence_thresh = 9e-5
    window_silence = [silence_thresh > e for e in window_energy]
    index = 0
    while(window_silence[index] and index < len(window_silence)-1):
        index += 1
    start = int(index * step_duration * rate) - window_size
    return data[start:]

def segment_method2(input_path,output_dir):
    # read audio file
    audio = AudioSegment.from_file(input_path)
    channels = audio.channels
    sampwidth = audio.sample_width        # <byte>
    rate = audio.frame_rate
    samples = audio.get_array_of_samples()
    data = np.array(samples)
    data = data.reshape(-1, channels)

    # get algorithm's parameter
    window_duration = 1     # min silence length <seconds>
     # The amount of time to step forward in the input file after calculating energy.
     # Smaller value = slower, but more accurate silence detection. 
    step_duration = window_duration / 10        # <second>
    window_size = int(window_duration * rate)   # <samples>
    step_size = int(step_duration * rate)       # <samples>

    # start algorithm
    signal_windows = windows(
    signal=(data/2**((8*sampwidth)-1)),
    window_size=window_size,
    step_size=step_size
    )

    window_energy = [energy(w) for w in signal_windows]
    energies = np.array(window_energy)

    # silence_thresh = np.mean(speech_energies)/4
    # silence_thresh = np.median(energies)/6
    silence_thresh = np.mean(energies)/10

    # find starting time in video timestamp
    window_silence = (e > silence_thresh for e in window_energy)
    cut_times = (r * step_duration for r in rising_edges(window_silence))

    # get -> at this start timestamp what sample number it is
    cut_samples = [int(t * rate) for t in cut_times]
    cut_samples.append(-1)

    # get index, start sample, stop sample to make a video
    cut_ranges = [(i, cut_samples[i], cut_samples[i+1]) for i in range(len(cut_samples) - 1)]
    output_filename = os.path.splitext(os.path.basename(input_path))
    for i, start, stop in cut_ranges:
        output_path = "{}_{:02d}.wav".format(output_dir+output_filename[0],i+1)
        # end_trimmed_data = end_silence_trim(data[start:stop],rate,sampwidth)
        # trimmed_data = start_silence_trim(end_trimmed_data,rate,sampwidth)
        # print("Writing file {}".format(output_path))
        audio_segment = AudioSegment(
            data[start:stop].tobytes(),
            frame_rate=rate,
            sample_width=sampwidth,
            channels=channels
        )
        audio_segment.export(output_path, format='wav')


# *********************************************************************************************

parser = argparse.ArgumentParser()
parser.add_argument('--input', action='store',type=str, required=True)
parser.add_argument('--output', action='store',type=str, required=True)
args = parser.parse_args()

input_dir = Path(args.input)
output_dir = Path(args.output)

for i, input_path in enumerate(input_dir.iterdir()):
    input_path = str(input_path)
    folder_name = os.path.splitext(os.path.basename(input_path))
    print('spliting {}{}'.format(folder_name[0],folder_name[1]))
    output_dir = str(output_dir) + '/' + folder_name[0] + '/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    segment_method2(input_path,output_dir)




