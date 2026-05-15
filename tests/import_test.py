import retinanalysis as ra
import visionloader as vl
import visionwriter as vw
import matplotlib.pyplot as plt

if __name__ == '__main__':

    exp = '20260318C'
    chunk = 'chunk2'
    data = 'data013'

    summary = ra.get_exp_summary(exp)
    print(summary)

    pipeline = ra.create_mea_pipeline(exp, data, chunk, typing_file = 'dragos_kilosort2.5.classification.txt')
    
    axes = pipeline.plot_rfs(cell_types = ['OffM', 'OnM', 'OffP', 'OnP', 'SBC'])
    plt.show()
