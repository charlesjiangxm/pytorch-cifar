import shutil
import os


def clear_dir(mk_new=True):
    rm_dir_name = ['plot', 'txt', 'plot_similarity', 'fmap_similarity_csv', 'fmap_sparsity_csv', 'plot_fmap',
                   'checkpoint', 'log']
    for name in rm_dir_name:
        shutil.rmtree(name, ignore_errors=True)
        os.mkdir(name) if mk_new else None


if __name__ == "__main__":
    clear_dir(mk_new=False)
