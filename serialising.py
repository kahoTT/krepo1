import pickle
import gzip
from pathlib import Path

class Serialising(object):
    def save(self, filename=None, path='/home/kaho/kepshot_run'):
        if filename:
            filename = filename+'.gz'
        outfile = gzip.open(Path(path)/filename, 'wb')
        pickle.dump(self, outfile)
        outfile.close()

    def load(filename, path='/home/kaho/kepshot_run'):
        if filename:
            filename = filename
        infile = gzip.open(Path(path)/filename, 'rb')
        self = pickle.load(infile)
        infile.close()
        return self