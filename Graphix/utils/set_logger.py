import logging
import sys
from logging.handlers import RotatingFileHandler

def set_logger(log_path, name='custom_logger', testing=False):
    logFormatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if testing:
        fileHandler = RotatingFileHandler('%s/log_test.txt' % (log_path), maxBytes=10*1024*1024, backupCount=5)
    else:
        fileHandler = RotatingFileHandler('%s/log_train.txt' % (log_path), maxBytes=10*1024*1024, backupCount=5)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    return logger

