from util import load_pickle
import matplotlib.pyplot as plt

svm = load_pickle('result-alpha0.1/svm/learning')
svm_lda_50 = load_pickle('result-alpha0.1/svm-lda/50/learning')
svm_lda_20 = load_pickle('result-alpha0.1/svm-lda/20/learning')
plt.plot(svm[0], svm[1], 'r')
plt.plot(svm_lda_50[0], svm_lda_50[1], 'b')
plt.plot(svm_lda_20[0], svm_lda_20[1], 'y')
plt.show()