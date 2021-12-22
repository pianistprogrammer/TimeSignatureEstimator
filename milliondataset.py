import hdf5_getters
import os


def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            filepath = root + os.sep + name
            if filepath.endswith(".h5"):
                r.append(os.path.join(root, name))
    return r


files = list_files('MillionSongSubset/')
for file in files:
    h5 = hdf5_getters.open_h5_file_read(file)
    duration = hdf5_getters.get_duration(h5)
    print(duration)
    h5.close()
