import pickle
import logging

def load_pickle(pickle_file):
    r"""
    Description:
    -----------
    Load pickle data.
    
    Parameters:
    -----------
    pickle_file: str
        File path.

    Returns:
    -----------
    pickle_data: any
        Pickle data.
    """
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data     = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data     = pickle.load(f, encoding='latin1')
    except Exception as e:
        logging.info('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

file_path = "datasets/sensor_graph/adj_mx_la.pkl"
adj_mx = load_pickle(file_path)[2]
edge = 0
for i in range(adj_mx.shape[0]):
    for j in range(adj_mx.shape[1]):
        if adj_mx[i][j] != 0:
            edge += 1

logging.info("==================== METR-LA ====================")
logging.info("# Node: {0}".format(adj_mx.shape[0]))
logging.info("# Edge: {0}".format(edge))

file_path = "datasets/sensor_graph/adj_mx_bay.pkl"
adj_mx = load_pickle(file_path)[2]
edge = 0
for i in range(adj_mx.shape[0]):
    for j in range(adj_mx.shape[1]):
        if adj_mx[i][j] != 0:
            edge += 1
logging.info("==================== PEMS-BAY ====================")
logging.info("# Node: {0}".format(adj_mx.shape[0]))
logging.info("# Edge: {0}".format(edge))

file_path = "datasets/sensor_graph/adj_mx_04.pkl"
adj_mx = load_pickle(file_path)
edge = 0
for i in range(adj_mx.shape[0]):
    for j in range(adj_mx.shape[1]):
        if adj_mx[i][j] != 0:
            edge += 1
logging.info("==================== PEMS04 ====================")
logging.info("# Node: {0}".format(adj_mx.shape[0]))
logging.info("# Edge: {0}".format(edge))

file_path = "datasets/sensor_graph/adj_mx_08.pkl"
adj_mx = load_pickle(file_path)
edge = 0
for i in range(adj_mx.shape[0]):
    for j in range(adj_mx.shape[1]):
        if adj_mx[i][j] != 0:
            edge += 1
logging.info("==================== PEMS08 ====================")
logging.info("# Node: {0}".format(adj_mx.shape[0]))
logging.info("# Edge: {0}".format(edge))

a = 1