from matplotlib import pyplot as plt
import os
import warnings

narw_dclde_selmap_train = [
    ('NOPP6_EST_20090328', 'NOPP6_20090328_RW_upcalls.selections.txt'),
    ('NOPP6_EST_20090329', 'NOPP6_20090329_RW_upcalls.selections.txt'),
    ('NOPP6_EST_20090330', 'NOPP6_20090330_RW_upcalls.selections.txt'),
    ('NOPP6_EST_20090331', 'NOPP6_20090331_RW_upcalls.selections.txt'),
]

narw_dclde_selmap_test = [
    ('NOPP6_EST_20090401', 'NOPP6_20090401_RW_upcalls.selections.txt'),
    ('NOPP6_EST_20090402', 'NOPP6_20090402_RW_upcalls.selections.txt'),
    ('NOPP6_EST_20090403', 'NOPP6_20090403_RW_upcalls.selections.txt'),
]


def save_display_to_disk(specs_dict, outputroot, test_mod_name, aug_name):

    rows = specs_dict['Original'].shape[0]
    cols = len(specs_dict)

    fig, ax = plt.subplots(rows, cols, sharex='all', sharey='all')
    #print(rows, cols, ax.shape)

    for r_idx in range(rows):
        vmin = min([val[r_idx, ...].min() for _, val in specs_dict.items()])
        vmax = max([val[r_idx, ...].max() for _, val in specs_dict.items()])

        for c_idx, (key, val) in enumerate(specs_dict.items()):
            ax[r_idx, c_idx].imshow(
                val[r_idx, ...],
                origin='lower', interpolation='none', cmap=plt.cm.jet,
                vmin=vmin, vmax=vmax
            )
            ax[r_idx, c_idx].spines['top'].set_visible(False)
            ax[r_idx, c_idx].spines['right'].set_visible(False)
            ax[r_idx, c_idx].spines['bottom'].set_visible(False)
            ax[r_idx, c_idx].spines['left'].set_visible(False)

            if r_idx == 0:
                ax[r_idx, c_idx].set_title(key)

    fig.tight_layout()
    # fig.canvas.set_window_title(aug_name)
    # plt.show()
    outdir = os.path.join(outputroot, test_mod_name)
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, f'{aug_name}.png'), bbox_inches='tight')
    warnings.warn(
        f'{aug_name} outputs saved at {outdir} for manual validation.')
