import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

dataset = 'awa'
tfevents1 = glob.glob(f'./log/final/{dataset}_False_g/*')[0]
tfevents2 = glob.glob(f'./log/final/{dataset}_True_g_KL/*')[0]

def get_acc(tfevents):
    acc_list = []
    best_acc = 0
    for i, e in enumerate(tf.train.summary_iterator(tfevents)):
        for v in e.summary.value:
            print(v)
            if v.tag == 'H':
                acc_list.append(v.simple_value)
                if v.simple_value == best_acc:
                    print()
            elif v.tag == 'best_H':
                best_acc = v.simple_value
    return acc_list, best_acc

# for tfevents in tfevents2:
#     _, best_acc = get_acc(tfevents)
#     print(tfevents, best_acc)

def main():
    acc_list1, best_acc1 = get_acc(tfevents1)
    acc_list2, best_acc2 = get_acc(tfevents2)
    print(len(acc_list1),len( acc_list2))
    print(best_acc1, best_acc2)

    num = 100
    x = np.arange(num)
    plt.plot(x, acc_list1[:num])
    plt.plot(x, acc_list2[:num])
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('H')
    plt.legend(['GAN', 'GAN + MI'] ,loc='lower right')
    plt.savefig(f'curve/curve_{dataset}_g.jpg')
    plt.savefig(f'curve/curve_{dataset}_g.pdf')

main()
