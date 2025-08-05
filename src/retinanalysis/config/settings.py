from configparser import ConfigParser
import platform
import os
# import importlib.resources as ir
import importlib_resources as ir
import retinanalysis


def load_config(config_path):
    if os.path.exists(config_path):
        config_path = os.path.abspath(config_path)
        configfile = ConfigParser()
        configfile.read(config_path)
    else:
        print()
        raise FileNotFoundError(f"No config file found at {config_path}.\nUse reset_config() and create_config() to make one.")

    if platform.system() == 'Darwin':
        DEFAULT_config = configfile['DEFAULT']
        SECONDARY_config = configfile['SECONDARY']
    else:
        DEFAULT_config = configfile['WINDOWS_DEFAULT']
        SECONDARY_config = configfile['WINDOWS_SECONDARY']
    
    if os.path.exists(os.path.abspath(DEFAULT_config['data'])):
        use_config = DEFAULT_config
    elif os.path.exists(os.path.abspath(SECONDARY_config['data'])):
        use_config = SECONDARY_config
    else:
        use_config = {'data': '', 'analysis': '', 'h5': '', 'meta': '', 'tags': '', 'query': '', 'user': ''}
        print("No NAS or SSD paths found, check that one of them is connected")

    mea_config = {'data' : use_config['data'], 'analysis': use_config['analysis'],
                'h5': use_config['h5'], 'meta': use_config['meta'], 'tags': use_config['tags'],
                'query': use_config['query'], 'user': use_config['user']}

    return mea_config


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


def reset_config(config_path):
    os.remove(config_path)
    print("Config file successfully deleted")

# reset_config('/Users/racheloaks-leaf/Desktop/retinanalysis/src/retinanalysis/')
# create_config(config_path='/Users/racheloaks-leaf/Desktop/retinanalysis/src/retinanalysis/config.ini', 
#               config_name='NAS',
#               data_dir='/Volumes/data/data/sorted/',
#                 analysis_dir='/Volumes/data/data/analysis/',
#                 h5_dir='/Volumes/data/data/h5/',
#                 meta_dir='/Volumes/data/datajoint_testbed/mea/meta/',
#                 tags_dir='/Volumes/data/datajoint_testbed/mea/tags/',
#                 username='roaksleaf')

# create_config(config_path='/Users/racheloaks-leaf/Desktop/retinanalysis/src/retinanalysis/config.ini', 
#               config_name='MAC_SSD',
#               data_dir='/Volumes/RachelSSD/mea/sorted/',
#                 analysis_dir='/Volumes/RachelSSD/mea/analysis/',
#                 h5_dir='/Volumes/RachelSSD/mea/h5/',
#                 meta_dir='/Volumes/RachelSSD/mea/datajoint_testbed/mea/meta/',
#                 tags_dir='/Volumes/RachelSSD/mea/datajoint_testbed/mea/tags/',
#                 username='roaksleaf')

config_path = ir.files(retinanalysis) / os.path.join("config", "config.ini")
mea_config = load_config(config_path)

DATA_DIR = mea_config['data'] 
ANALYSIS_DIR = mea_config['analysis']
H5_DIR = mea_config['h5']
META_DIR = mea_config['meta']
TAGS_DIR = mea_config['tags']
QUERY_DIR = mea_config['query']
USER = mea_config['user']
