from PyQt6.QtWidgets import QFileDialog, QApplication
import sys
from configparser import ConfigParser
import os

def select_directory(title):
    directory = QFileDialog.getExistingDirectory(None, caption = title, directory = "/")
    
    if directory:
        print(f"Selected directory: {directory}")
    else:
        print("No directory selected.")
    
    return directory

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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    directory = select_directory(title = "Select H5 Directory")
    print(directory)
    