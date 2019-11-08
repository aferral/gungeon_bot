from contextlib import contextmanager
import time
import pickle
import numpy as np
import struct
import socket
import os
import sys
from state_fusion.logger import getLogger

log=getLogger('msg_sockets')

chunk_size=4096*20000
send_buf_size = 2048
socket_type=socket.SOCK_STREAM
skip_delay = 0.05 


def load_msg(data_raw):
    data_dict = pickle.loads(data_raw)
    now = time.time()
    trec = data_dict['timestamp']
    delay = now - trec
    #log.info('Elapsed between send-rec: {0}'.format(delay))
    return data_dict,delay


def format_msg(data_dict):
    timestamp = time.time()
    data_dict['timestamp'] = timestamp
    data_raw = pickle.dumps(data_dict)
    data_len = len(data_raw)
    header_data=struct.pack('i', data_len)
    return header_data + data_raw



def socket_read_msg(name,verbose=False,skip_if_delay=False):
    """
    Unix domain socket ITERATOR for interprocess comunication (send dictionaries)
    :param name: name for the socket (must be the same for reader, sender)
    :param verbose: show debug msgs
    :param skip_if_delay: should reader deliver old msgs ??
    :return: iterator that returns msgs as dictionaries
    """
    try:
        os.unlink(name)
    except OSError:
        if os.path.exists(name):
            raise

    sock_read = socket.socket(socket.AF_UNIX, socket_type)
    if verbose: log.info('starting up on %s' % name)
    
    sock_read.bind(name)

    sock_read.listen(1) 
    header_size = struct.calcsize('i')

    all_delays = []
    
    if verbose: log.info('waiting for a connection')

    conn, client_address = sock_read.accept()


    st=time.time()
    try:
        temp_buffer = bytes()
        reading=False
        size_to_read=0
        skiped=0

        while True:
            
            #if len(temp_buffer) < 500:
            data = conn.recv(chunk_size) 
            #log.info("received {0} bytes".format(len(data)))
            temp_buffer += data

            if len(data) == 0 and len(temp_buffer) == 0:
                break
        
            if not reading:
                size_to_read = struct.unpack('i',temp_buffer[0:header_size])[0]
                #log.info('Reading msg of len {0}'.format(size_to_read))
                reading=True

            if len(temp_buffer) >= (size_to_read+header_size):
                msg_len = size_to_read+header_size
                data_msg = temp_buffer[header_size:msg_len]
                reading=False

                msg,delay = load_msg(data_msg)
                #log.debug('Delay: {0}'.format(delay))
                if skip_if_delay and (delay > skip_delay):
                    skiped +=1
                    temp_buffer = temp_buffer[msg_len:]
                    continue

                if verbose: log.info('Skiped {0}'.format(skiped))

                yield msg


                skiped=0
                all_delays.append(delay)

                temp_buffer = temp_buffer[msg_len:]
    finally:
        en=time.time()
        conn.close()
        sock_read.close()

        if verbose:
            log.info('Elapsed: {0}'.format(en-st))
            log.info('N messages rec: {0}'.format(len(all_delays)))
            log.info(np.array(all_delays).mean())

@contextmanager
def socket_sender(name,verbose=False):
    """
    Send data (anything pickable) to the unix domain socket named 'name'
    usage
    with socket_sender(name) as x:
        a=1
        x(1)
    It will automatically close the socket after all the data has been send

    :param name: name of the socket
    :param verbose: show debug msgs
    :return:
    """

    sock = socket.socket(socket.AF_UNIX, socket_type)
    sock.setsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF,send_buf_size)

    if verbose: log.info('connecting to %s' % name)

    log.info('WAITING FOR CONN')
    while True: # waiting for connection
        try:
            sock.connect(name)
            break
        except socket.error as msg:
            time.sleep(0.5)
    log.info('CONN READY')
    try:

        def send_fun(obj):
            data_to_send = format_msg(obj)
            if verbose: log.info('sending {0} bytes'.format(len(data_to_send)))
            sock.sendall(data_to_send)

        yield send_fun

    finally:
        time.sleep(1)
        if verbose: log.info('closing socket')
        sock.close()


if __name__ == '__main__':
    server_address = './uds_socket'

    argv = sys.argv
    modo = argv[1]

    if modo == '0':
        log.info('INIT READER')
        for msg in socket_read_msg(server_address,verbose=True):
            pass

    if modo == '1':
        log.info('INIT SENDER')

        n_msg = 100
        
        with socket_sender(server_address) as send_fun:

            for i in range(n_msg):
                img = np.random.rand(100,100,3).astype(np.float32)
                dict_test = {'test' : 32,'hola' : 33,'img': img}
                send_fun(dict_test)
