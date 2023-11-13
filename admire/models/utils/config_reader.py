import configparser

def read_config(file: str = 'config.ini') -> dict:
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Create a dictionary of all the sections
    config_dict = {}
    for section in config.sections():
        config_dict[section] = {}
        for option in config.options(section):
            # Add the lower and upper case versions of the key
            # Just for convenience
            config_dict[section][option.lower()] = config.get(section, option)
            config_dict[section][option.upper()] = config.get(section, option)
            
    return config_dict