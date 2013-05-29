import os

def get_root_dir():
    fpath = os.path.abspath(__file__)
    (cwdir, fname) = os.path.split(fpath)
    rdir = os.path.abspath(os.path.join(cwdir, '..', '..', '..'))
    return rdir

ROOT_DIR = get_root_dir()

def get_image_path(name):
    return os.path.join(ROOT_DIR, 'images', name)
