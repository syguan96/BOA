from utils.data_preprocess import pw3d_extract, mpi_inf_3dhp_extract, h36m_eval_extract, h36m_train_extract
import config
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, choices=['3dpw', '3dhp', 'h36m'],help='process which dataset?')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.dataset == '3dpw':
        pw3d_extract(config.PW3D_ROOT, config.DATASET_NPZ_PATH, debug=False)
    elif args.dataset == '3dhp':
        mpi_inf_3dhp_extract(config.MPI_INF_3DHP_ROOT, config.DATASET_NPZ_PATH, 'test')
    elif args.dataset == 'h36m':
        h36m_train_extract(config.H36M_ROOT, config.DATASET_NPZ_PATH, extract_img=True)
        # h36m_extract(config.H36M_ROOT, config.DATASET_NPZ_PATH, protocol=1, extract_img=True)
        # h36m_extract(config.H36M_ROOT, config.DATASET_NPZ_PATH, protocol=2, extract_img=True)
    else:
        print('Not implemented.')
