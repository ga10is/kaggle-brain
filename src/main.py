from .common.logger import create_logger
from .tvp import train

if __name__ == '__main__':
    create_logger('log/brain.log')

    train()
