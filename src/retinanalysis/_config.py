"""A config class that holds live values for all necessary directories
(i.e. 'DATA_DIR', 'ANALYSIS_DIR', etc.). The class includes CLI and GUI
setup helpers, the ability to change the current active config, and 
the ability to create or delete new profiles such as 'ssd' or 'nas'.
"""
from pathlib import Path
from platformdirs import user_config_path
from sys import platform
import tomlkit
from warnings import warn
import os

class ConfigNotInitializedError(RuntimeError):
    pass

class RAConfig:
    """Config class, used to create a single 'config' instance
    used by retinanalysis to track the currently active config
    paths, swap config profiles, create new profiles, and set up
    a fresh config file during first use.
    """
    REQUIRED_KEYS = (
        'analysis',
        'data',
        'raw', 
        'h5',
        'vision',
        'meta',
        'tags',
        'user'
    )
    
    KEY_DICT = {
        'analysis' : 'Analysis directory',
        'data' : 'Sorted data directory',
        'raw' : 'Raw data directory',
        'h5' : 'H5 file directory',
        'vision': 'Path to Vision.jar',
        'meta' : 'Metadata directory',
        'tags' : 'Tags directory',
        'user' : 'Username',
    }

    def __init__(self):

        self.config_path = self._get_config_path()

        if self.config_path.is_file():
            with open(self.config_path, 'r') as f:
                self.config_file = tomlkit.load(f)
            self._initialized = True

            self.active_profile = self.config_file['active']['profile']
            if not os.path.exists(self.ANALYSIS_DIR):
                for profile in self.profiles:
                    self.set_profile(profile)
                    if os.path.exists(self.ANALYSIS_DIR):
                        break


            if not os.path.exists(self.ANALYSIS_DIR):
                warn(
                    'None of the available config paths currently connected. '
                    'Connect to SSD or NAS and run ra.config.reset()',
                    stacklevel=2,
                )
                self.active_profile=None
                self._initialized=False

        else:
            warn(
                f'No config file found. To configure, run:\n'
                    '\nOption 1, CLI setup: retinanalysis.config.setup()\n'
                    '\nOption 2, GUI setup: retinanalysis.config.setup_gui()\n'
                    f'\nOption 3, Manual setup: edit {self.config_path}',
                stacklevel=2,
            )
            self.active_profile = None
            self._initialized = False

    def reset(self):
        self.__init__()


    def setup(self, overwrite: bool = False) -> None:
        """CLI config setup method that prompts the user to create
        a new profile and input all of the required paths one by one.

        Args:
            overwrite: Optional, default False. If true will overwrite
                an existing working config file.

        Raises:
            ValueError: if a working config exists and overwrite = False
        """
        if (self._initialized) and (not overwrite):
            raise ValueError(
                'A config already exists. To overwrite this config, set '
                'overwrite = True.'
            )

        print('Input a config profile and all relevant paths. To skip a path, '
              'enter an empty string.')

        name = input('Config profile name: ')
        profile_paths = dict()

        for key in self.REQUIRED_KEYS:
            profile_paths[key] = input(f'{self.KEY_DICT[key]} : ')

        self._setup(name, profile_paths)

    def setup_gui(self, overwrite: bool = False):
        """GUI config setup method that uses tkinter to create a setup
        window that allows the use to input all of the required paths
        one by one interactively.

        Args:
            overwrite: Optional, default False. If true allows the user
                to overwrite an existing working config file.

        Raises:
            ValueError: if a working config file exists and overwrite = False.
        """
        if (self._initialized) and (not overwrite):
            raise ValueError(
                'A config already exists. To overwrite this config, set '
                'overwrite = True.'
            )

        import subprocess
        import tempfile
        import json
        import sys

        # Create an empty json in a temp directory, which will be used by
        # the GUI to write and return the values we need.
        fd, result_file = tempfile.mkstemp(suffix='.json')
        with os.fdopen(fd, 'w') as f:
            json.dump({}, f)

        gui_module = Path(__file__).parent / '_config_gui.py'

        subprocess.run([sys.executable, str(gui_module), result_file])

        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                result = json.load(f)

            if ('name' in result) and ('paths' in result):
                self._setup(
                    name=result['name'],
                    profile_paths=result['paths']
                )
                os.remove(result_file)
            else:
                warn('Setup cancelled - No config created')
                os.remove(result_file)
        else:
            warn('Setup cancelled - No config created')


    def set_profile(self, name) -> None:
        """Method for setting the current active profile. Note, this does
        NOT overwrite the default 'active' profile in the config.toml file.
        To set a new default profile, call ``set_default_profile`` method.

        Args:
            name: name of new profile as a string

        Raises:
            ConfigNotInitializedError if no config file is found.
            ValueError if 'name' does not match an available profile.
        """
        self._check_initialized('Cannot set new profile')

        if name not in self.profiles:
            raise ValueError(
                f'{name} is not a valid config profile. '
                f'Available profiles are {self.profiles}.'
            )

        self.config_file['active']['profile'] = name
        self.active_profile = self.config_file['active']['profile']

    def set_default_profile(self, name) -> None:
        """Method for setting the default active profile. This method
        overwrites the profile listed under 'active' in config.toml

        Args:
            name: name of new default profile to set

        Raises:
            ConfigNotInitializedError if no config file is found.
            ValueError if 'name' does not match an available profile.
        """
        self._check_initialized('Cannot set a new default profile')

        if name not in self.profiles:
            raise ValueError(
                f'{name} is not a valid config profile. '
                f'Available profiles are {self.profiles}. '
                'Use create_profile() or create_profile_gui() to add '
                'a new profile to the list.'
            )

        self.config_file['active']['profile'] = name
        self.active_profile = name

        with open(self.config_path, 'w') as f:
            tomlkit.dump(self.config_file, f)

    def create_profile(
        self,
        name: str,
        profile_paths: dict[str, str],
        overwrite: bool = False,
    ) -> None:
        """Method for creating a new config profile, which is 
        written to the config.toml file. Note that this does not
        set the new profile as active by default. To do so, call
        ``set_profile(name)`` or ``set_default_profile(name)``.

        Args:
            name: name of new profile to create
            profile_paths: dictionary that contains one of more of these keys. 
                - 'analysis' (OPTIONAL)
                - 'data' (OPTIONAL)
                - 'raw' (OPTIONAL)
                - 'h5' (OPTIONAL)
                - 'vision' (OPTIONAL)
                - 'meta' (OPTIONAL)
                - 'tags' (OPTIONAL)
                - 'user' (REQUIRED)
                If a key is not included, it will be filled with an empty string.
            overwrite: optional, default False. If true allows 
                create_profile() to overwrite an existing profile
                with the same ``name``.

        Raises:
            ValueError if no config file is found.
            ValueError: if overwrite is set to False and ``name`` is
                found in the list of existing profiles.
            Warning if extra keys are found in ``profile_paths``.
                Extra keys are ignored, per the warning message.

        """
        if not self.config_path.is_file():
            raise ValueError(
                'No config file found. Run ra.config.setup() '
                'or ra.config.setup_gui() to create the file and '
                'add your first profile.'
            )

        if (not overwrite) and (name in self.profiles):
            raise ValueError(
                f'{name} profile already exists and overwrite = False. '
                'To overwrite this profile set overwrite = True.\n'
            )


        extra_keys = list(set(profile_paths.keys()) - set(self.REQUIRED_KEYS))
        if extra_keys:
            warn(
                f'Only {self.REQUIRED_KEYS} allowed in config. '
                f'Ignoring unrecognized keys: {extra_keys} '
            )

        new_profile = {name : dict()}
        for key in self.REQUIRED_KEYS:
            if key not in profile_paths:
                profile_paths[key] = ""
            if key == 'user':
                if (not profile_paths[key]) or (profile_paths[key] is None):
                    raise ValueError(
                            f'Username is required'
                    )

            new_profile[name][key] = profile_paths[key]

        self.config_file['profiles'].update(new_profile)

        with open(self.config_path, 'w') as f:
            tomlkit.dump(self.config_file, f)

        self.reset()
    
    def create_profile_gui(self, overwrite = False):
        """Method for creating a new config profile interactively,
        using a graphical user interface. The new profile is appended 
        to the config.toml file. Note that this does not set the new
        profile as active by default. To do so, call ``set_profile(name)``
        or ``set_default_profile(name)`` after completing the setup.

        Args:
            overwrite: optional, default False. If true allows 
                create_profile_gui() to overwrite an existing profile
                with the same name.

        Raises:
            ValueError if no config file is found.
            ValueError: if overwrite is set to False and given ``name`` is
                found in the list of existing profiles.
        """
        import subprocess
        import tempfile
        import json
        import sys

        if not self.config_path.is_file():
            raise ValueError(
                'No config file found. Run ra.config.setup() '
                'or ra.config.setup_gui() to create the file and '
                'add your first profile.'
            )

        # Create an empty json in a temp directory, which will be used by
        # the GUI to write and return the values we need.
        fd, result_file = tempfile.mkstemp(suffix='.json')
        with os.fdopen(fd, 'w') as f:
            json.dump({}, f)

        gui_module = Path(__file__).parent / '_config_gui.py'

        subprocess.run([sys.executable, str(gui_module), result_file])

        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                result = json.load(f)

            if ('name' in result) and ('paths' in result):

                self.create_profile(
                    name=result['name'],
                    profile_paths=result['paths'],
                    overwrite=overwrite,
                )
                os.remove(result_file)
            else:
                warn('Profile creation cancelled - no profile created')
                os.remove(result_file)
        else:
            warn('Profile creation cancelled - no profile created')

    def edit_profile(
        self,
        name: str,
        profile_paths: dict[str, str],
    ) -> None:
        """Method for creating a new config profile, which is 
        written to the config.toml file. Note that this does not
        set the new profile as active by default. To do so, call
        ``set_profile(name)`` or ``set_default_profile(name)``.

        Args:
            name: name of profile to edit
            profile_paths: dictionary that contains all keys to be edited 
                and their new values. 
                Possible keys are:
                    - 'analysis'
                    - 'data'
                    - 'raw'
                    - 'h5'
                    - 'vision'
                    - 'meta'
                    - 'tags'
                    - 'user'

        Raises:
            ValueError if no config file is found.
            ValueError: if ``name`` does not match an existing profile.
            Warning if extra keys are found in ``profile_paths``.
                Extra keys are ignored, per the warning message.

        """
        if not self.config_path.is_file():
            raise ValueError(
                'No config file found. Run ra.config.setup() '
                'or ra.config.setup_gui() to create the file and '
                'add your first profile.'
            )

        if name not in self.profiles:
            raise ValueError(
                f'{name} profile does not exist in the current config. '
                'Use create_profile() or create_profile_gui() to make a new '
                'profile from scratch.'
            )


        extra_keys = list(set(profile_paths.keys()) - set(self.REQUIRED_KEYS))
        if extra_keys:
            warn(
                f'Only {self.REQUIRED_KEYS} allowed in config. '
                f'Ignoring unrecognized keys: {extra_keys} '
            )

        profile = {name : dict()}
        for key in self.REQUIRED_KEYS:
            if key not in profile_paths:
                profile_paths[key] = self.config_file['profiles'][name][key]
            if key == 'user':
                if (not profile_paths[key]) or (profile_paths[key] is None):
                    raise ValueError(
                            f'Username cannot be blank'
                    )

            profile[name][key] = profile_paths[key]

        self.config_file['profiles'].update(profile)

        with open(self.config_path, 'w') as f:
            tomlkit.dump(self.config_file, f)

        self.reset()


    def edit_profile_gui(self, name):
        """Method for editing an existing config profile interactively,
        using a graphical user interface. The profile is edited inside 
        the config.toml file. Note that this does not set the newly edited
        profile as active by default. To do so, call ``set_profile(name)``
        or ``set_default_profile(name)`` after finishing the edit.

        Args:
            name: The name of the profile to edit

        Raises:
            ValueError if no config file is found.
            ValueError: if ``name`` does not match an existing profile  
        """
        import subprocess
        import tempfile
        import json
        import sys

        if not self.config_path.is_file():
            raise ValueError(
                'No config file found. Run ra.config.setup() '
                'or ra.config.setup_gui() to create the file and '
                'add your first profile.'
            )

        if name not in self.profiles:
            raise ValueError(
                f'{name} is not an existing profile. To create it, '
                'call create_profile() or create_profile_gui()'
            )

        # Create an empty json in a temp directory, which will be used by
        # the GUI to write and return the values we need.
        input_dict = self.config_file['profiles'][name].copy()
        input_dict['name'] = name

        fd, result_file = tempfile.mkstemp(suffix='.json')
        with os.fdopen(fd, 'w') as f:
            json.dump(input_dict, f)

        gui_module = Path(__file__).parent / '_config_gui.py'

        subprocess.run([sys.executable, str(gui_module), result_file])

        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                result = json.load(f)

            if ('name' in result) and ('paths' in result):

                self.edit_profile(
                    name=result['name'],
                    profile_paths=result['paths'],
                )
                os.remove(result_file)
            else:
                warn(f'Profile edit cancelled - {name} hasn not been edited.')
                os.remove(result_file)
        else:
            warn(f'Profile edit cancelled - {name} hasn not been edited.')


    def remove_profile(self, name):
        self._check_initialized('Cannot remove profile')

        if name not in self.profiles:
            raise ValueError(
                f'{name} not in list of available profiles.\n'
                'Available profiles are:\n'
                f'{self.profiles}'
            )

        if name == self.active_profile:
            raise ValueError(
                f'{name} is the current active profile. '
                'Switch to a different profile before removing.'
            )

        if name == self.config_file['active']['profile']:
            raise ValueError(
                f'{name} is the current default profile. '
                'Use set_default_profile() to switch the default '
                f'before removing {name}.'
            )

        self.config_file['profiles'].pop(name)

        with open(self.config_path, 'w') as f:
            tomlkit.dump(self.config_file, f)


    @property
    def profiles(self):
        if not hasattr(self, 'config_file'):
            return ()
        return tuple(self.config_file['profiles'].keys())

    @property
    def ANALYSIS_DIR(self) -> str:
        self._check_initialized()
        return self.config_file['profiles'][self.active_profile]['analysis']


    @property
    def DATA_DIR(self) -> str:
        self._check_initialized()
        return self.config_file['profiles'][self.active_profile]['data']


    @property
    def RAW_DIR(self) -> str:
        self._check_initialized()
        return self.config_file['profiles'][self.active_profile]['raw']


    @property
    def H5_DIR(self) -> str:
        self._check_initialized()
        return self.config_file['profiles'][self.active_profile]['h5']

    @property
    def VISION_PATH(self) -> str:
        self._check_initialized()
        return self.config_file['profiles'][self.active_profile]['vision']

    @property
    def META_DIR(self) -> str:
        self._check_initialized()
        return self.config_file['profiles'][self.active_profile]['meta']

    @property
    def TAGS_DIR(self) -> str:
        self._check_initialized()
        return self.config_file['profiles'][self.active_profile]['tags']

    @property
    def USER(self) -> str:
        self._check_initialized()
        return self.config_file['profiles'][self.active_profile]['user']

    def _setup(
        self,
        name: str,
        profile_paths: dict[str, str],
    ) -> None:
        """Private helper function that does the actual setup. Creates a
        fresh config.toml file in the user's platform-specific config
        directory, populates the appropriate toml fields, and uses 
        user-provided name and profile_paths to create and populate the
        first config profile, then sets it as the default active profile.

        Args:
            name: name of profile to create
            profile_paths: dictionary that contains these keys: 
                - 'analysis' (OPTIONAL)
                - 'data' (OPTIONAL)
                - 'raw' (OPTIONAL)
                - 'h5' (OPTIONAL)
                - 'meta' (OPTIONAL)
                - 'tags' (OPTIONAL)
                - 'user' (REQUIRED)
                Missing keys will be filled with an empty string
        """
        # Create directories
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = True

        # Create dictionary structure for toml using tomlkit's built
        # in types
        self.config_file = tomlkit.document()
        active = tomlkit.table()
        active.add('profile', '')
        self.config_file.add('active', active)
        self.config_file.add('profiles', tomlkit.table())

        # Set active profile
        self.config_file['active']['profile'] = name
        self.active_profile = name

        # Write the empty scaffold so create_profile finds the file
        with open(self.config_path, 'w') as f:
            tomlkit.dump(self.config_file, f)

        # Create the profile by adding the paths
        self.create_profile(name, profile_paths)


    def _get_config_path(self) -> Path:
        """Private helper method that uses platformdirs to pull the user's
        platform-specific config directory and generate the config_path
        variable. Note: on Mac, we overwrite the default ``~/Library/Application Support``
        directory and use ``~/.config`` instead.
        """
        if platform == 'darwin':
            return Path.home() / '.config' / 'retinanalysis' / 'config.toml'
        else:
            return user_config_path('retinanalysis', appauthor=False) / 'config.toml'

    def _check_initialized(
        self,
        message: str = 'Run config.setup() to configure.',
    ) -> None:
        """Private helper method that checks if a config file was found
        and successfully initialized.
        
        Args:
            message: Optional message for the user that follows 'No active config.'
                Default: 'Run config.setup() to configure.'

        Raises:
            ConfigNotInitializedError if self._initialized = False
        """
        if not self._initialized:
            raise ConfigNotInitializedError(
                f"No active config. {message}."
            )

config = RAConfig()
