import os
import hashlib
import requests
import logging
import sys

def download_files(list_file_name, output_dir):
    # Set up logging
    log_file = os.path.join(output_dir, 'download.log')
    logging.basicConfig(filename=log_file, level=logging.DEBUG)

    with open(list_file_name+"_list.txt", 'w') as wf:
        with open(list_file_name, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                name, url, hash_value = line.strip().split(';')
                wf.write(url + "\n")
                # print(f'Downloading {name} from {url}')
                # response = requests.get(url)
                # file_data = response.content
                # if hashlib.md5(file_data).hexdigest() == hash_value:
                #     file_path = os.path.join(output_dir, name)
                #     with open(file_path, 'wb') as f:
                #         f.write(file_data)
                #     print(f'Downloaded {name} from {url}')
                # else:
                #     print(f'Hash mismatch for {name} from {url}')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} LIST_FILE OUTPUT_DIR')
        sys.exit(1)

    list_file_name = sys.argv[1]
    output_dir = sys.argv[2]
    
    download_files(list_file_name, output_dir)