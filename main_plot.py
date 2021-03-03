import utils
import numpy as np
from main_train import args

if __name__ == "__main__":
    utils.plot_analysis(plot_type='sparsity', max_epoch=args.epochs)
    utils.plot_train_curve_from_csv()
    utils.plot_batch_hist_from_csv(np.arange(1, args.epoches, args.epoches/9).astype(int))
