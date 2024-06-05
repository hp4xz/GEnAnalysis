def Function_JSON(key, config_file='config.json'):
    import json

    with open(config_file, 'r') as file:
        config = json.load(file)
    
    return config.get(key)