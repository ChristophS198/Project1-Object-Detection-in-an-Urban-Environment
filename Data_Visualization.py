#import matplotlib
import json
import numpy as np
import matplotlib.pyplot as plt

# {1: 180065, 2: 54216, 4: 1400}
num_vehicles = 360331
num_pedestrians = 107443
num_cyclists = 2776
obj_count = num_vehicles + num_pedestrians + num_cyclists

plt.bar([1, 2, 3], height=[num_vehicles/obj_count, num_pedestrians/obj_count, num_cyclists/obj_count])
plt.xticks([1, 2, 3], ['Vehicles','Pedestrians','Cyclists'])
plt.ylabel('Relative frequency')
plt.title('Comparison of class occurreces based on 20.000 images')

plt.show()


with open('./ClassComp.json') as f:
    predictions = json.load(f)

filenames = []
vehicles = []
pedestrians = []
cyclists = []
labels = []
for record in predictions:
    filenames.append(record)
    labels.append(record.split('segment-')[-1][:10])
    vehicles.append(predictions[record]["1"])
    pedestrians.append(predictions[record]["2"])
    cyclists.append(predictions[record]["4"])
print(labels)
#num_ojbects = sum(vehicles) + sum(pedestrians) + sum(cyclists)
vehicles = np.array(vehicles)
pedestrians = np.array(pedestrians)
cyclists = np.array(cyclists)

men_means = [20, 35, 30, 35, 27]
women_means = [25, 32, 34, 20, 25]
men_std = [2, 3, 4, 1, 2]
women_std = [3, 5, 2, 3, 3]
width = 0.8       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()

ax.bar(labels, vehicles, width, label='vehicles',align='center')
ax.bar(labels, pedestrians, width, bottom=vehicles, label='pedestrians',align='center')
ax.bar(labels, cyclists, width,  bottom=vehicles+pedestrians, label='cyclists',align='center')

ax.set_ylabel('Absolute occurrence in the first 10000 elements')
ax.set_title('Class distribution over tfrecords')
plt.xticks(rotation='vertical')
ax.legend()

plt.show()