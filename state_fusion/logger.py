import logging


def getLogger(name,log_level=logging.DEBUG):

    # formateer
    formatter=logging.Formatter("[%(levelname)s] %(process)d %(thread)d [%(name)s] %(asctime)s | Msg: %(message)s ")

    # logger
    logger = logging.Logger(name)
    logger.setLevel(log_level)

    # Console log
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)  # A consola

    return logger

