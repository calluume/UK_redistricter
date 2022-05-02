import gdown, json
from os import makedirs
from os.path import exists, dirname

if __name__ == "__main__":
    with open('data/default_parameters.json', 'r') as defaults_file:
        files = json.load(defaults_file)['default_files']

    required = 0
    for file in files.values():
        if not exists(file['output_file']): required += 1
    
    if required > 0:
        print("{0} (of {1}) required files must be downloaded.".format(required, len(files.keys())))

        for file in files.values():
            if not exists(file['output_file']):
                print('\n'+file['name'].upper()+":")

                if not exists(dirname(file['output_file'])):
                    makedirs(dirname(file['output_file']))

                gdown.download(file['gdrive_link'], file['output_file'], quiet=False)
        
        print('All required files have finished downloaded!')
    else:
        print("All required files have already been downloaded!")
    
    print("  ↳ Run 'redistricter.py' to start the model.")