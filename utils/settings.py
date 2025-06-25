from configparser import ConfigParser
import platform
import os

def load_config(config_path):
    root_directory = os.path.abspath('../')
    config_path = os.path.join(root_directory, config_path)
    if os.path.exists(config_path):
        config_path = os.path.abspath(config_path)
        configfile = ConfigParser()
        configfile.read(config_path)
    else:
        raise FileNotFoundError("No config file found. Use reset_config() and create_config() to make one.")

    if platform.system() == 'Darwin':
        NAS_config = configfile['DEFAULT']
        SSD_config = configfile['MAC_SSD']
        if os.path.exists(os.path.abspath(NAS_config['data'])):
            data_dir = NAS_config['data']
            analysis_dir = NAS_config['analysis']
            h5_dir = NAS_config['h5']
            meta_dir = NAS_config['meta']
            tags_dir = NAS_config['tags']
            user = NAS_config['user']
        elif os.path.exists(os.path.abspath(SSD_config['data'])):
            data_dir = SSD_config['data']
            analysis_dir = SSD_config['analysis']
            h5_dir = SSD_config['h5']
            meta_dir = SSD_config['meta']
            tags_dir = SSD_config['tags']
            user = SSD_config['user']
        else:
            raise FileNotFoundError("No NAS or SSD paths found, check that one of them is connected")

    else:
        NAS_config = configfile['WINDOWS_NAS']
        SSD_config = configfile['WINDOWS_SSD']
        if os.path.exists(os.path.abspath(NAS_config['data'])):
            data_dir = NAS_config['data']
            analysis_dir = NAS_config['analysis']
            h5_dir = NAS_config['h5']
            meta_dir = NAS_config['meta']
            tags_dir = NAS_config['tags']
            user = NAS_config['user']
        elif os.path.exists(os.path.abspath(SSD_config['data'])):
            data_dir = SSD_config['data']
            analysis_dir = SSD_config['analysis']
            h5_dir = SSD_config['h5']
            meta_dir = SSD_config['meta']
            tags_dir = SSD_config['tags']
            user = SSD_config['user']
        else:
            raise FileNotFoundError("No NAS or SSD paths found, check that one of them is connected")
    
    mea_config = {'data' : data_dir, 'analysis': analysis_dir,
                'h5': h5_dir, 'meta': meta_dir, 'tags': tags_dir,
                'user': user}

    return mea_config

def reset_config(config_path):
    os.remove(config_path)
    print("Config file successfully deleted")

def create_config(config_path, config_name,
                      data_dir, analysis_dir, 
                      h5_dir, meta_dir, tags_dir, username = 'drezeanu'):

    config = ConfigParser()
    config[config_name] = {'Analysis': analysis_dir,
                         'Data': data_dir,
                         'H5': h5_dir,
                         'Meta': meta_dir,
                         'Tags': tags_dir,
                         'User': username}

    if os.path.exists(config_path):
        with open(config_path, "a") as configfile:
            config.write(configfile)

    else:
        with open(config_path, "w") as configfile:
            config.write(configfile)

mea_config = load_config('utils/config.ini')

NAS_DATA_DIR = mea_config['data'] 
NAS_ANALYSIS_DIR = mea_config['analysis']
H5_DIR = mea_config['h5']
META_DIR = mea_config['meta']
TAGS_DIR = mea_config['tags']
USER = mea_config['user']
