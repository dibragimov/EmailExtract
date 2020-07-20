import argparse
import copy
import logging
import numpy as np
import os
import time
from nn_service import create_app


'''logging.basicConfig(filename='./results/app.log', filemode='a',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')'''

if __name__ == '__main__':
    appli = create_app()
    time.sleep(5)  # sleep for 5 seconds
    appli.run(host='0.0.0.0', use_reloader=False, threaded=True, debug=True, port=5000)