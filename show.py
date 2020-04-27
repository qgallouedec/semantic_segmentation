import matplotlib.pyplot as plt
import json

plt.figure()

with open('results/result.json', 'r') as file:
    result1 = json.load(file)

with open('results/result_add.json', 'r') as file:
    result2 = json.load(file)

with open('results/result_max.json', 'r') as file:
    result3 = json.load(file)

with open('results/result_min.json', 'r') as file:
    result4 = json.load(file)

plt.plot(result1['train_loss'], color='red', label='UNET - train loss')
plt.plot(result1['test_loss'], color='red', ls='--', label='UNET - test loss')


plt.plot(result2['train_loss'], color='blue',
         label='UNET using ADD - train loss')
plt.plot(result2['test_loss'], color='blue', ls='--',
         label='UNET using ADD - test loss')


plt.plot(result3['train_loss'], color='orange',
         label='UNET using MAX - train loss')
plt.plot(result3['test_loss'], color='orange', ls='--',
         label='UNET using MAX - test loss')


plt.plot(result4['train_loss'], color='green',
         label='UNET using MIN - train loss')
plt.plot(result4['test_loss'], color='green', ls='--',
         label='UNET using MIN - test loss')


plt.legend(loc="upper right")
plt.grid()
plt.ylim(0, 1.3)
plt.xlim(-1, 31)
plt.xlabel("number of epochs")
plt.ylabel("cross-entropy loss")
plt.savefig("docs/question3.png")
plt.title("Replacing concatenation in decoding")
plt.show()
